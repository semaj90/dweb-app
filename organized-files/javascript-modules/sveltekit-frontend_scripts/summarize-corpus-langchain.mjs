#!/usr/bin/env node
/**
 * summarize-corpus-langchain.mjs
 * Builds a vector-aware hierarchical summary of the documents corpus.
 * Steps:
 *  1. Fetch documents + (optionally) their embeddings from Postgres.
 *  2. Chunk text (simple length-based or token-estimate heuristic) -> mini-summaries.
 *  3. Use LangChain LLM (Ollama model) to summarize each chunk concurrently (limit concurrency).
 *  4. Embed each document aggregate summary using real embedding model (Ollama) and cluster vectors (simple k-means-lite) for topic grouping.
 *  5. Produce final JSON + markdown report with: per-topic summary, representative documents, key terms.
 *
 * CLI flags (yargs):
 *  --databaseUrl <url>
 *  --model <ollama model name> (default: gemma3-legal)
 *  --maxDocs <n> limit number of documents (0=all)
 *  --chunkSize <chars> naive char length chunk size (default 4000)
 *  --chunkOverlap <chars> (default 400)
 *  --concurrency <n> LLM requests in flight (default 3)
 *  --topics <n> approximate target topics (default 8)
 *  --outputDir ./scripts/logs
 *  --dryRun just plan without calling LLM
 *
 * Output:
 *  JSON: corpus-summary.json
 *  Markdown: corpus-summary.md
 */
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import postgres from 'postgres';
import fs from 'fs';
import path from 'path';

// Lazy dynamic imports for langchain to avoid cost if dry-run
async function loadLangChain(){
  const core = await import('@langchain/core');
  const community = await import('@langchain/community');
  return { core, community };
}

const argv = yargs(hideBin(process.argv))
  .option('databaseUrl',{ type:'string', default: process.env.DATABASE_URL || `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || '123456'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'legal_ai'}`, describe:'Postgres URL'})
  .option('model',{ type:'string', default: process.env.SUMMARY_MODEL || 'gemma3-legal', describe:'Ollama model name'})
  .option('maxDocs',{ type:'number', default: 0, describe:'Limit number of docs (0 = all)'})
  .option('chunkSize',{ type:'number', default: 4000 })
  .option('chunkOverlap',{ type:'number', default: 400 })
  .option('concurrency',{ type:'number', default: 3 })
  .option('topics',{ type:'number', default: 8 })
  .option('outputDir',{ type:'string', default: path.join(process.cwd(),'scripts','logs') })
  .option('dryRun',{ type:'boolean', default: false })
  .option('embedModel',{ type:'string', default: process.env.EMBED_MODEL || 'nomic-embed-text', describe:'Embedding model'})
  .option('embedUrl',{ type:'string', default: process.env.OLLAMA_ENDPOINT || (process.env.OLLAMA_BASE_URL ? `${process.env.OLLAMA_BASE_URL}/api/embeddings` : 'http://localhost:11434/api/embeddings'), describe:'Embedding endpoint URL'})
  .option('retryAttempts',{ type:'number', default: 3, describe:'LLM summary max retry attempts'})
  .option('retryBaseMs',{ type:'number', default: 750, describe:'Base backoff (ms)'})
  .option('persist',{ type:'boolean', default: true, describe:'Persist result to corpus_summaries table'})
  .help()
  .argv;

const {
  databaseUrl: DATABASE_URL,
  model: MODEL,
  maxDocs: MAX_DOCS,
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
  concurrency: CONCURRENCY,
  topics: TARGET_TOPICS,
  outputDir: OUTPUT_DIR,
  dryRun: DRY_RUN
 } = argv;
const EMBED_MODEL = argv.embedModel;
const EMBED_URL = argv.embedUrl;
const RETRY_ATTEMPTS = argv.retryAttempts;
const RETRY_BASE = argv.retryBaseMs;
const PERSIST = argv.persist;

function estimateTokens(str){ return Math.ceil(str.length / 4); }

function chunkText(text){
  const chunks = [];
  let i = 0;
  while(i < text.length){
    const slice = text.slice(i, i + CHUNK_SIZE);
    chunks.push(slice);
    i += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

async function retry(fn, attempts, base){
  let lastErr;
  for(let i=0;i<attempts;i++){
    try { return await fn(); } catch(e){ lastErr = e; const delay = base * 2**i + Math.random()*150; await new Promise(r=>setTimeout(r, delay)); }
  }
  throw lastErr;
}

async function summarizeChunks(llm, chunks){
  const results = [];
  let active = 0; let index = 0; let resolved = 0;
  return await new Promise((resolve,reject)=>{
    const next = () => {
      if(resolved === chunks.length) return resolve(results);
      while(active < CONCURRENCY && index < chunks.length){
        const currentIndex = index++;
        const chunk = chunks[currentIndex];
        active++;
        (async () => {
          let summary = '[dry-run summary placeholder]';
          if(!DRY_RUN){
            try {
              const prompt = `Provide a concise legal-domain oriented summary (max 120 words) highlighting key issues, parties, obligations, and any risks. Text:\n"""\n${chunk.slice(0, 6000)}\n"""\nSummary:`;
              summary = await retry(async ()=>{
                let out = await llm.invoke(prompt);
                if(typeof out === 'object' && out?.content){
                  out = Array.isArray(out.content) ? out.content.map(p=>p?.text||'').join('\n') : out.content;
                }
                return out;
              }, RETRY_ATTEMPTS, RETRY_BASE);
            } catch(e){ summary = `[summary failed after retries: ${e.message}]`; }
          }
          results[currentIndex] = { summary, charLength: chunk.length, tokenEstimate: estimateTokens(chunk) };
          active--; resolved++;
          next();
        })();
      }
    };
    next();
  });
}

function cosine(a,b){
  let dot=0,na=0,nb=0; const len=Math.min(a.length,b.length);
  for(let i=0;i<len;i++){ const x=a[i], y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) || 1);
}

async function embedSummaries(summaries){
  const out = [];
  let i=0; let active=0; const lim = CONCURRENCY;
  return await new Promise(resolve=>{
    const next=()=>{
      if(out.length === summaries.length) return resolve(out);
      while(active < lim && i < summaries.length){
        const idx = i++; active++;
        (async()=>{
          let vec=[];
          if(!DRY_RUN){
            try {
              const body = JSON.stringify({ model: EMBED_MODEL, prompt: summaries[idx].summary });
              const res = await retry(()=>fetch(EMBED_URL,{ method:'POST', headers:{'Content-Type':'application/json'}, body }), RETRY_ATTEMPTS, RETRY_BASE);
              if(!res.ok) throw new Error(`HTTP ${res.status}`);
              const json = await res.json();
              vec = Array.isArray(json.embedding)? json.embedding : json.data?.[0]?.embedding || [];
            } catch(e){
              // fallback hash embedding
              const fallback = new Array(256).fill(0);
              const words = summaries[idx].summary.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
              for(const w of words){ let h=0; for(let j=0;j<w.length;j++) h = (h*31 + w.charCodeAt(j)) & 0xffffffff; const pos = Math.abs(h)%256; fallback[pos]++; }
              const norm = Math.sqrt(fallback.reduce((s,v)=>s+v*v,0))||1; vec = fallback.map(v=>v/norm);
            }
          } else {
            vec = new Array(32).fill(0); // tiny placeholder
          }
          out[idx] = { ...summaries[idx], vector: vec };
          active--; next();
        })();
      }
    };
    next();
  });
}

function clusterSummaries(points, k){
  if(points.length <= k) return points.map((s,i)=>({ topic:`Topic ${i+1}`, summaries:[s] }));
  let centroids = points.slice(0,k).map(p=>p.vector.slice());
  for(let iter=0; iter<5; iter++){
    // Assign
    for(const p of points){
      let best=-1, bestScore=-Infinity;
      for(let i=0;i<centroids.length;i++){
        const score = cosine(p.vector, centroids[i]);
        if(score > bestScore){ bestScore=score; best=i; }
      }
      p.cluster = best;
    }
    // Recompute
    for(let i=0;i<centroids.length;i++){
      const clusterPoints = points.filter(p=>p.cluster===i);
      if(!clusterPoints.length) continue;
      const dim = clusterPoints[0].vector.length;
      const acc = new Array(dim).fill(0);
      for(const p of clusterPoints){ for(let j=0;j<dim;j++) acc[j]+=p.vector[j]; }
      const norm = Math.sqrt(acc.reduce((s,v)=>s+v*v,0))||1;
      centroids[i] = acc.map(v=>v/norm);
    }
  }
  const groups = [];
  for(let i=0;i<k;i++){
    groups.push({ topic:`Topic ${i+1}`, summaries: points.filter(p=>p.cluster===i) });
  }
  return groups;
}

async function main(){
  console.log('🧾 Corpus summarization (LangChain)');
  console.log(JSON.stringify({ MODEL, MAX_DOCS, CHUNK_SIZE, CHUNK_OVERLAP, CONCURRENCY, TARGET_TOPICS, DRY_RUN, EMBED_MODEL, EMBED_URL, PERSIST },null,2));
  if(!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR,{recursive:true});

  const sql = postgres(DATABASE_URL, { max:4 });
  const baseDocs = MAX_DOCS ? await sql`SELECT id, filename, content FROM documents ORDER BY id ASC LIMIT ${MAX_DOCS}` : await sql`SELECT id, filename, content FROM documents ORDER BY id ASC`;
  console.log(`Fetched documents: ${baseDocs.length}`);
  const docs = baseDocs.map(d=>({ id:d.id, filename:d.filename || `doc-${d.id}`, content:(d.content||'').slice(0, 200000) }));

  let lc;
  let llm;
  if(!DRY_RUN){
    lc = await loadLangChain();
    const { Ollama } = await import('@langchain/community/llms/ollama');
    llm = new Ollama({ model: MODEL, baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434' });
  }

  const docSummaries = [];
  for(const doc of docs){
    const chunks = chunkText(doc.content);
    const chunkSummaries = await summarizeChunks(llm, chunks);
    const aggregate = chunkSummaries.map(cs=>cs.summary).join('\n');
    docSummaries.push({ id: doc.id, filename: doc.filename, chunkCount: chunks.length, summary: aggregate });
    console.log(` summarized doc ${doc.id} chunks=${chunks.length}`);
  }

  // Flatten all chunk-level summaries for clustering
  const flatSummaries = docSummaries.map(d=>({ docId:d.id, filename:d.filename, summary:d.summary }));
  const embedded = await embedSummaries(flatSummaries);
  const groups = clusterSummaries(embedded, TARGET_TOPICS);

  function topKeywords(text, n=12){
    const freq = new Map();
    for(const w of text.toLowerCase().split(/[^a-z0-9]+/).filter(x=>x.length>4)){ freq.set(w,(freq.get(w)||0)+1); }
    return [...freq.entries()].sort((a,b)=>b[1]-a[1]).slice(0,n).map(([w])=>w);
  }

  const topics = groups.map(g=>{
    const merged = g.summaries.map(s=>s.summary).join('\n');
    return {
      topic: g.topic,
      documentCount: new Set(g.summaries.map(s=>s.docId)).size,
      representativeDocuments: [...new Set(g.summaries.slice(0,5).map(s=>s.filename))],
      keywords: topKeywords(merged),
      combinedSummary: merged.slice(0,4000)
    };
  });

  const output = { generatedAt: new Date().toISOString(), model: MODEL, docs: docSummaries.length, topics, parameters:{ MODEL, MAX_DOCS, CHUNK_SIZE, CHUNK_OVERLAP, CONCURRENCY, TARGET_TOPICS, DRY_RUN, EMBED_MODEL }, persist: PERSIST };
  const jsonFile = path.join(OUTPUT_DIR, 'corpus-summary.json');
  fs.writeFileSync(jsonFile, JSON.stringify(output,null,2));
  console.log('📄 JSON written:', jsonFile);

  const mdFile = path.join(OUTPUT_DIR, 'corpus-summary.md');
  const md = ['# Corpus Summary', '', `Generated: ${output.generatedAt}`, '', '## Topics', ...topics.map(t=>`### ${t.topic}\nDocuments: ${t.documentCount}\nKeywords: ${t.keywords.join(', ')}\n\n${t.combinedSummary}\n`)].join('\n');
  fs.writeFileSync(mdFile, md);
  console.log('📝 Markdown written:', mdFile);

  if(PERSIST && !DRY_RUN){
    try {
      await sql`CREATE TABLE IF NOT EXISTS corpus_summaries (id SERIAL PRIMARY KEY, generated_at TIMESTAMPTZ NOT NULL, model TEXT, embed_model TEXT, docs_count INT, topics JSONB, parameters JSONB, created_at TIMESTAMPTZ DEFAULT now())`;
      await sql`INSERT INTO corpus_summaries (generated_at, model, embed_model, docs_count, topics, parameters) VALUES (${output.generatedAt}, ${MODEL}, ${EMBED_MODEL}, ${docSummaries.length}, ${sql.json(topics)}, ${sql.json(output.parameters)})`;
      console.log('💾 Persisted corpus summary row');
    } catch(e){ console.error('⚠️  Failed to persist corpus summary:', e.message); }
  }
  await sql.end();
  console.log('✅ Summarization complete');
}
main().catch(e=>{ console.error('❌ Summarization failed', e); process.exit(1); });
