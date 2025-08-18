#!/usr/bin/env node
// Recommendation Aggregator: summarizes JSONL ingest + history and optionally persists to PostgreSQL
// Sources:
//  - logs/recommendation-ingest.jsonl (case pipeline events)
//  - logs/recommendations-history.jsonl (daemon fetch events)

const fs = require('fs');
const path = require('path');
let Client;
try { ({ Client } = require('pg')); } catch(_) { /* pg optional */ }

const INGEST_FILE = process.env.REC_INGEST_FILE || path.join('logs','recommendation-ingest.jsonl');
const HISTORY_FILE = process.env.REC_HISTORY_FILE || path.join('logs','recommendations-history.jsonl');

async function readJSONL(file){
  try {
    if (!fs.existsSync(file)) return [];
    const data = fs.readFileSync(file,'utf8');
    return data.split(/\n+/).filter(Boolean).map(l=>{try{return JSON.parse(l);}catch(_){return null;}}).filter(Boolean);
  } catch(e){
    console.warn('⚠️ Failed reading', file, e.message);
    return [];
  }
}

function aggregateRecommendationIngest(ingestEntries, historyEntries){
  const topRecommendations = new Map();
  const byModel = new Map();
  const stages = new Map();
  let recCount = 0;

  for (const e of ingestEntries){
    if (e.stage) stages.set(e.stage, (stages.get(e.stage)||0)+1);
    const recs = Array.isArray(e.recommendations?.recommendations) ? e.recommendations.recommendations : (Array.isArray(e.recommendations)? e.recommendations : []);
    for (const r of recs){
      if (typeof r === 'string'){
        topRecommendations.set(r, (topRecommendations.get(r)||0)+1);
      } else if (r && r.issue){
        topRecommendations.set(r.issue, (topRecommendations.get(r.issue)||0)+1);
      }
      recCount++;
    }
    if (e.model) byModel.set(e.model, (byModel.get(e.model)||0)+1);
  }
  // Also incorporate historyEntries textual pattern
  for (const h of historyEntries){
    const recs = h.recommendations || [];
    for (const r of recs){
      if (r.issue) topRecommendations.set(r.issue, (topRecommendations.get(r.issue)||0)+1);
    }
  }

  const topArray = Array.from(topRecommendations.entries()).sort((a,b)=>b[1]-a[1]).slice(0,20).map(([text,count])=>({text,count}));
  const summary = {
    generated_at: new Date().toISOString(),
    total_ingest_entries: ingestEntries.length,
    total_history_entries: historyEntries.length,
    total_recommendations_counted: recCount,
    top_recommendations: topArray,
    by_model: Object.fromEntries(byModel),
    by_stage: Object.fromEntries(stages)
  };
  return summary;
}

async function persistSummary(summary){
  const outFile = path.join('logs','recommendation-aggregate.json');
  try {
    fs.mkdirSync('logs', { recursive: true });
    fs.writeFileSync(outFile, JSON.stringify(summary,null,2));
  } catch(e){ console.warn('⚠️ Failed to write aggregate file:', e.message); }
}

async function persistToPostgres(ingestEntries){
  if (!Client) { console.warn('ℹ️ pg module not installed, skipping DB persistence'); return; }
  const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';
  const client = new Client({ connectionString });
  try {
    await client.connect();
    await client.query(`CREATE TABLE IF NOT EXISTS recommendation_ingest (
      id SERIAL PRIMARY KEY,
      ts timestamptz NOT NULL,
      case_id text,
      stage text,
      model text,
      recommendations jsonb,
      parsed boolean,
      indexed boolean,
      solution jsonb,
      raw jsonb,
      UNIQUE (ts, case_id, stage)
    );`);
    const insertText = `INSERT INTO recommendation_ingest (ts, case_id, stage, model, recommendations, parsed, indexed, solution, raw)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        ON CONFLICT DO NOTHING`;
    for (const e of ingestEntries){
      const ts = e.ts || e.timestamp || new Date().toISOString();
      await client.query(insertText,[ts, e.caseId||e.case_id||null, e.stage||null, e.model||null, JSON.stringify(e.recommendations||null), e.parsed||false, e.indexed||false, JSON.stringify(e.solution||null), JSON.stringify(e)]);
    }
  } catch(e){
    console.warn('⚠️ Postgres persistence failed:', e.message);
  } finally { await client.end().catch(()=>{}); }
}

async function aggregateAndPersist(){
  const ingestEntries = await readJSONL(INGEST_FILE);
  const historyEntries = await readJSONL(HISTORY_FILE);
  const summary = aggregateRecommendationIngest(ingestEntries, historyEntries);
  await persistSummary(summary);
  if (process.env.REC_PG_PERSIST === 'true') {
    await persistToPostgres(ingestEntries);
  }
  return summary;
}

if (require.main === module){
  aggregateAndPersist().then(s=>{
    console.log('📊 Recommendation Summary Generated');
    console.log(JSON.stringify(s,null,2));
  });
}

module.exports = { aggregateRecommendationIngest, aggregateAndPersist, persistToPostgres };
