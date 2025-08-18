// rag/entity_enricher.js - naive entity extraction and graph enrichment
import { Client as PgClient } from 'pg';
import neo4j from 'neo4j-driver';
import Redis from 'ioredis';

const POSTGRES_DSN = process.env.RAG_DATABASE_URL || process.env.DATABASE_URL || 'postgresql://postgres:password@127.0.0.1:5432/postgres';
const pg = new PgClient({ connectionString: POSTGRES_DSN });
await pg.connect().catch(e=>{ console.warn('PG connect failed', e.message); process.exit(0); });

const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
const driver = neo4j.driver(process.env.NEO4J_URI || 'bolt://localhost:7687', neo4j.auth.basic(process.env.NEO4J_USER || 'neo4j', process.env.NEO4J_PASSWORD || 'password'));
const session = driver.session();

function extractEntities(text){
  const seen = new Set(); const ents=[];
  const stop = new Set(['The','This','That','There','Where','Here','Such','Shall','Party','Agreement','Section','Clause','Any','Each','Other','Including']);
  const regex = /\b([A-Z][A-Za-z]{3,})\b/g; let m; let cap=0;
  while((m=regex.exec(text)) && cap<80){ const w=m[1]; if (stop.has(w)) continue; if(!seen.has(w)){ seen.add(w); ents.push(w); cap++; } }
  return ents;
}

async function processDoc(docId){
  const res = await pg.query('SELECT text FROM legal_documents WHERE doc_id=$1 LIMIT 200',[docId]);
  const corpus = res.rows.map(r=> r.text).join(' ');
  const ents = extractEntities(corpus).slice(0,40);
  if (!ents.length) return;
  const tx = session.beginTransaction();
  try {
    await tx.run('MERGE (d:Document {doc_id:$id}) SET d.last_enriched=timestamp()', { id:docId });
    for (const e of ents){
      await tx.run('MERGE (ent:Entity {name:$n}) MERGE (d:Document {doc_id:$id})-[:MENTIONS]->(ent)', { n:e, id:docId });
    }
    await tx.commit();
  } catch(e){ console.warn('enrich failed', e.message); try { await tx.rollback(); } catch{} }
  await redis.sadd('rag_entity_enriched_docs', docId);
  console.log('[entity-enricher] enriched', docId, 'ents=', ents.length);
}

async function loop(){
  while(true){
    try {
      const enriched = await redis.smembers('rag_entity_enriched_docs');
      const res = await pg.query('SELECT DISTINCT doc_id FROM legal_documents WHERE doc_id IS NOT NULL LIMIT 50');
      for (const row of res.rows){
        if (enriched.includes(row.doc_id)) continue;
        await processDoc(row.doc_id);
      }
    } catch(e){ console.warn('enricher loop error', e.message); }
    await new Promise(r=> setTimeout(r,5000));
  }
}
loop();
