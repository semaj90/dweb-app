#!/usr/bin/env node
/**
 * verify-embeddings-dim.mjs
 * Verifies that all embeddings in legal_embeddings match the configured dimension.
 * Reports anomalies (shorter/longer) and basic stats.
 */
import postgres from 'postgres';
import fs from 'fs';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const argv = yargs(hideBin(process.argv))
  .option('targetDim', { type:'number', default: parseInt(process.env.EMBEDDING_DIM || process.env.VECTOR_DIM || '768',10), describe:'Target embedding dimension' })
  .option('databaseUrl', { type:'string', default: process.env.DATABASE_URL || `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || '123456'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'legal_ai'}`, describe:'Postgres URL' })
  .option('sampleLimit', { type:'number', default: parseInt(process.env.SAMPLE_LIMIT || '2000',10), describe:'Rows to sample (0=all)' })
  .option('exitOnAnomaly', { type:'boolean', default: true, describe:'Non-zero exit code if anomalies detected' })
  .help()
  .argv;

const TARGET_DIM = argv.targetDim;
const DATABASE_URL = argv.databaseUrl;
const SAMPLE_LIMIT = argv.sampleLimit;
const EXIT_ON_ANOMALY = argv.exitOnAnomaly;

async function main(){
  const sql = postgres(DATABASE_URL, { max: 4 });
  console.log('🔎 Verifying embedding dimensions');
  console.log('DB:', DATABASE_URL.replace(/:[^:@]*@/,'://****@'));
  console.log('Target dimension:', TARGET_DIM);

  const [{ count: docCount }] = await sql`SELECT COUNT(*)::int FROM documents`;
  const [{ count: embCount }] = await sql`SELECT COUNT(*)::int FROM legal_embeddings`;
  console.log(`Documents: ${docCount} | Embedding rows: ${embCount}`);

  // Sample all (or cap for large datasets)
  let rows;
  if(SAMPLE_LIMIT === 0){
    rows = await sql`SELECT id, document_id, embedding FROM legal_embeddings`;
  } else {
    rows = await sql`SELECT id, document_id, embedding FROM legal_embeddings LIMIT ${SAMPLE_LIMIT}`;
  }

  let tooShort = [], tooLong = [], ok = 0;
  for(const r of rows){
    const len = r.embedding?.length || 0;
    if(len === TARGET_DIM) ok++; else if(len < TARGET_DIM) tooShort.push({ id:r.id, documentId:r.document_id, len }); else tooLong.push({ id:r.id, documentId:r.document_id, len });
  }

  console.log(`Checked rows: ${rows.length}`);
  console.log(`OK: ${ok}`);
  console.log(`Too short: ${tooShort.length}`);
  console.log(`Too long: ${tooLong.length}`);

  const report = { generatedAt: new Date().toISOString(), targetDim: TARGET_DIM, totals:{ documents: docCount, embeddings: embCount, sampled: rows.length }, ok, tooShort, tooLong };
  const outDir = new URL('./logs/', import.meta.url).pathname.replace(/%20/g,' ');
  if(!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive:true });
  const outFile = outDir + 'embedding-dimension-report.json';
  fs.writeFileSync(outFile, JSON.stringify(report,null,2));
  console.log('📄 Report saved:', outFile);

  if(tooShort.length || tooLong.length){
    console.log('⚠️  Dimension anomalies detected. Run backfill script to fix.');
    if(EXIT_ON_ANOMALY) process.exitCode = 2;
  } else {
    console.log('✅ All sampled embeddings match target dimension.');
  }
  await sql.end();
}
main().catch(e=>{ console.error('❌ Verification failed:', e); process.exit(1); });
