// rag/retrieval.js - Hybrid retrieval + ranking pipeline scaffold
import { Client as PgClient } from 'pg';
import neo4j from 'neo4j-driver';
import Fastify from 'fastify';

const POSTGRES_DSN = process.env.RAG_DATABASE_URL || process.env.DATABASE_URL || 'postgresql://postgres:password@127.0.0.1:5432/postgres';
const TABLE = process.env.RAG_TABLE || 'legal_documents';
const VECTOR_COL = process.env.RAG_VECTOR_COL || 'embedding';
const MAX_DIM = Number(process.env.RAG_DIM || 768);

let pg = null; let pgAvailable = true;
try { pg = new PgClient({ connectionString: POSTGRES_DSN }); await pg.connect(); }
catch(e){ pgAvailable = false; console.warn('[RAG] Postgres unavailable:', e.message); }

const BEST_PRACTICE_KEYWORDS = ['liability','damages','warranty','confidential','performance','security','compliance','termination','jurisdiction'];
let neoAvailable=false; let neoDriver=null; let neoSession=null;
// Metrics state
const metrics = { start: Date.now(), queries:0, lastMs:0, avgMs:0, emptyResults:0 };
try {
  const uri = process.env.NEO4J_URI || 'bolt://localhost:7687';
  const user = process.env.NEO4J_USER || 'neo4j';
  const pass = process.env.NEO4J_PASSWORD || 'password';
  neoDriver = neo4j.driver(uri, neo4j.auth.basic(user, pass));
  neoSession = neoDriver.session();
  neoAvailable = true;
} catch(e){ /* ignore */ }

function cosineSQL(vector){
  const placeholders = vector.map((_,i)=> `$${i+1}`).join(',');
  return {
    text: `SELECT id, doc_id, chunk_id, text, 1 - (${VECTOR_COL} <=> vector[${placeholders}]) AS similarity, created_at\n           FROM ${TABLE}\n           WHERE ${VECTOR_COL} IS NOT NULL\n           ORDER BY ${VECTOR_COL} <=> vector[${placeholders}] ASC\n           LIMIT $${vector.length+1}`,
    values: [...vector, ...vector, 40]
  };
}

export async function baseVectorSearch(queryEmbedding){
  if (!pgAvailable) return [];
  try { const stmt = cosineSQL(queryEmbedding); const res = await pg.query(stmt); return res.rows.map(r=>({ id:r.id, doc_id:r.doc_id, chunk_id:r.chunk_id, text:r.text, semantic:Number(r.similarity||0), created_at:r.created_at })); }
  catch(e){ console.warn('[RAG] vector search failed:', e.message); return []; }
}

async function entityOverlap(docIds){
  if (!neoAvailable || !docIds.length) return {};
  try {
    const res = await neoSession.run('MATCH (d:Document)-[:MENTIONS]->(e:Entity) WHERE d.doc_id IN $ids RETURN d.doc_id AS id, collect(distinct e.name) AS ents',{ ids: docIds });
    const map={}; res.records.forEach(r=>{ map[r.get('id')] = r.get('ents'); });
    return map;
  } catch { return {}; }
}

export async function rerank(results){
  const now = Date.now();
  const docIds = [...new Set(results.map(r=> r.doc_id).filter(Boolean))];
  const entMap = await entityOverlap(docIds);
  return results.map(r=>{
    const ageHours = r.created_at ? Math.max(1,(now - Date.parse(r.created_at))/3600000) : 1;
    const recencyBoost = 1/ageHours;
    const lower = (r.text||'').toLowerCase();
    let overlap = 0; for (const kw of BEST_PRACTICE_KEYWORDS){ if (lower.includes(kw)) overlap++; }
    const overlapBoost = 1 + overlap / BEST_PRACTICE_KEYWORDS.length;
    const ents = entMap[r.doc_id] || [];
    const entityDensity = ents.length ? Math.min(1, ents.length / 12) : 0;
    const high_score = r.semantic * 0.6 + recencyBoost * 0.2 + overlapBoost * 0.05 + entityDensity * 0.15;
    return { ...r, recencyBoost, overlapBoost, entityDensity, high_score };
  }).sort((a,b)=> b.high_score - a.high_score);
}

export function buildPrompt(query, ranked){
  const contextBlocks = ranked.slice(0,6).map((r,i)=>`[C${i}] high_score=${r.high_score.toFixed(4)}\n${r.text.slice(0,500)}`).join('\n\n');
  return [
    'You are a legal domain assistant. Cite sources as [C#]. If insufficient context, state limitation.',
    `User Query: ${query}`,
    'Context Chunks:',
    contextBlocks,
    'Instructions: Provide a concise, legally careful answer with bullet points when helpful, cite relevant chunk ids, and include a short risk note if ambiguity exists.'
  ].join('\n\n');
}

export async function ensureSchema(){
  if (!pgAvailable) return;
  const ddl = `CREATE EXTENSION IF NOT EXISTS vector;\nCREATE TABLE IF NOT EXISTS ${TABLE} (\n  id bigserial primary key,\n  doc_id text,\n  chunk_id int,\n  text text,\n  ${VECTOR_COL} vector(${MAX_DIM}),\n  created_at timestamptz default now()\n);`;
  try { await pg.query(ddl); } catch(e){ console.warn('[RAG] ensureSchema failed:', e.message); }
}
await ensureSchema();

export async function retrieve({ queryEmbedding }){
  const t0 = Date.now();
  const base = await baseVectorSearch(queryEmbedding);
  const ranked = await rerank(base);
  const out = ranked.slice(0,10);
  const dt = Date.now()-t0;
  metrics.queries++;
  metrics.lastMs = dt;
  metrics.avgMs = metrics.avgMs + (dt - metrics.avgMs)/metrics.queries;
  if (!out.length) metrics.emptyResults++;
  return out;
}
export function degradedFlags(){ return { pgAvailable, neoAvailable }; }
export function retrievalMetrics(){
  const uptimeSec = (Date.now()-metrics.start)/1000;
  return { service:'retrieval', uptimeSec, ...metrics, pgAvailable, neoAvailable, timestamp:new Date().toISOString() };
}

if (process.env.RAG_RETRIEVAL_METRICS_PORT){
  const port = Number(process.env.RAG_RETRIEVAL_METRICS_PORT);
  const srv = Fastify({ logger:false });
  srv.get('/metrics', async ()=> retrievalMetrics());
  srv.listen({ port, host:'0.0.0.0' }).then(()=> console.log('[RAG] retrieval metrics on', port)).catch(e=> console.warn('Retrieval metrics failed:', e.message));
}
