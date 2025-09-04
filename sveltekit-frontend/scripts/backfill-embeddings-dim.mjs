#!/usr/bin/env node
/**
 * backfill-embeddings-dim.mjs
 * Paginated streaming backfill: scans legal_embeddings in ascending id order without loading entire table.
 * Pads/truncates embeddings to target dimension. Creates rolling JSONL backup for safety.
 * Supports CLI flags via yargs:
 *   --targetDim 768 --pageSize 1000 --limit 0 --dryRun true --databaseUrl postgres://... --sinceId 0 --progressInterval 5000 --stopOn first|never
 */
import postgres from 'postgres';
import fs from 'fs';
import path from 'path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const argv = yargs(hideBin(process.argv))
  .option('targetDim', { type: 'number', default: parseInt(process.env.EMBEDDING_DIM || process.env.VECTOR_DIM || '768',10), describe: 'Target embedding dimension' })
  .option('pageSize', { type: 'number', default: parseInt(process.env.PAGE_SIZE || '1000',10), describe: 'Rows fetched per page' })
  .option('batchSize', { type: 'number', default: parseInt(process.env.BATCH_SIZE || '500',10), describe: 'Rows updated per transaction batch' })
  .option('limit', { type: 'number', default: 0, describe: 'Max rows to scan (0 = all)' })
  .option('dryRun', { type: 'boolean', default: (process.env.DRY_RUN || 'true').toLowerCase()==='true', describe: 'Do not persist updates' })
  .option('databaseUrl', { type: 'string', default: process.env.DATABASE_URL || `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || '123456'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'legal_ai'}`, describe: 'Postgres connection URL' })
  .option('sinceId', { type: 'number', default: 0, describe: 'Resume scanning after this id' })
  .option('progressInterval', { type: 'number', default: 5000, describe: 'Progress log interval in ms' })
  .option('stopOn', { type: 'string', choices:['first','never'], default: 'never', describe: 'Stop after first anomaly (useful for quick check)' })
  .help()
  .argv;

const TARGET_DIM = argv.targetDim;
const PAGE_SIZE = argv.pageSize;
const BATCH_SIZE = argv.batchSize;
const LIMIT = argv.limit;
const DRY_RUN = argv.dryRun;
const DATABASE_URL = argv.databaseUrl;
const SINCE_ID = argv.sinceId;
const PROGRESS_INTERVAL = argv.progressInterval;
const STOP_ON = argv.stopOn;

function normalize(vec){
  if(!Array.isArray(vec)) return [];
  if(vec.length === TARGET_DIM) return vec;
  if(vec.length > TARGET_DIM) return vec.slice(0, TARGET_DIM);
  return vec.concat(Array(TARGET_DIM - vec.length).fill(0));
}

async function main(){
  console.log('🔧 Streaming Backfill (dimensions)');
  console.log(JSON.stringify({ TARGET_DIM, PAGE_SIZE, BATCH_SIZE, LIMIT, DRY_RUN, SINCE_ID, STOP_ON }, null,2));

  const sql = postgres(DATABASE_URL, { max: 4 });
  const backupDir = path.join(path.dirname(new URL(import.meta.url).pathname), 'logs');
  if(!fs.existsSync(backupDir)) fs.mkdirSync(backupDir,{recursive:true});
  const backupFile = path.join(backupDir, `embedding-backfill-backup-${Date.now()}.jsonl`);
  const summary = { scanned:0, anomalies:0, updated:0, started:new Date().toISOString(), targetDim: TARGET_DIM };

  let lastId = SINCE_ID;
  let page = 0;
  let pageAnomalies = [];
  let lastProgress = Date.now();
  let stop = false;

  while(!stop){
    page++;
    const rows = await sql`SELECT id, document_id, embedding FROM legal_embeddings WHERE id > ${lastId} ORDER BY id ASC LIMIT ${PAGE_SIZE}`;
    if(rows.length === 0) break;
    lastId = rows[rows.length-1].id;
    summary.scanned += rows.length;

    for(const r of rows){
      if(LIMIT && summary.scanned > LIMIT){ stop = true; break; }
      if(!Array.isArray(r.embedding)) continue; // skip null/invalid
      if(r.embedding.length !== TARGET_DIM){
        summary.anomalies++;
        const fixed = normalize(r.embedding);
        pageAnomalies.push({ id:r.id, documentId:r.document_id, originalLength:r.embedding.length, fixedLength: fixed.length, diff: fixed.length - r.embedding.length, original: r.embedding });
        fs.appendFileSync(backupFile, JSON.stringify({ id:r.id, documentId:r.document_id, embedding:r.embedding })+'\n');
        if(STOP_ON==='first'){ stop = true; break; }
      }
    }

    if(!DRY_RUN && pageAnomalies.length){
      // chunk updates in BATCH_SIZE transactions
      for(let i=0;i<pageAnomalies.length;i+=BATCH_SIZE){
        const batch = pageAnomalies.slice(i,i+BATCH_SIZE);
        await sql.begin(async tx => {
          for(const a of batch){
            const fixed = normalize(a.original);
            await tx`UPDATE legal_embeddings SET embedding = ${fixed} WHERE id = ${a.id}`;
            summary.updated++;
          }
        });
      }
      console.log(`Page ${page} updated ${pageAnomalies.length} anomalies (cumulative updated ${summary.updated})`);
    }
    pageAnomalies = [];

    if(Date.now() - lastProgress > PROGRESS_INTERVAL){
      lastProgress = Date.now();
      console.log(`Progress: scanned=${summary.scanned} anomalies=${summary.anomalies} updated=${summary.updated} lastId=${lastId}`);
    }
  }

  summary.completed = new Date().toISOString();
  summary.durationSeconds = (Date.now() - Date.parse(summary.started))/1000;
  summary.backupFile = backupFile;
  summary.dryRun = DRY_RUN;
  const summaryFile = path.join(backupDir, 'embedding-backfill-summary.json');
  fs.writeFileSync(summaryFile, JSON.stringify(summary,null,2));
  console.log('📄 Summary saved:', summaryFile);
  console.log('🎉 Backfill finished');
  if(DRY_RUN) console.log('Dry run: no database changes were applied.');
  await sql.end();
}
main().catch(e=>{ console.error('❌ Backfill failed:', e); process.exit(1); });
