#!/usr/bin/env node
/**
 * post-batch-selftest.mjs
 * Uses batch-upload log to pick a random successful document, extracts keywords and tests /api/rag/search.
 */
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const argv = yargs(hideBin(process.argv))
  .option('frontendPort', { type:'number', default: parseInt(process.env.FRONTEND_PORT || '5173',10), describe:'Frontend port / API fallback' })
  .option('apiBase', { type:'string', default: process.env.RAG_BASE_URL, describe:'Override API base URL' })
  .option('logFile', { type:'string', default: path.join(process.cwd(), 'scripts','logs','batch-upload-log.jsonl'), describe:'Batch upload log file path' })
  .option('limit', { type:'number', default: 5, describe:'Search topK limit' })
  .option('threshold', { type:'number', default: 0.6, describe:'Minimum similarity score expected' })
  .option('requireOriginal', { type:'boolean', default: true, describe:'Fail if original documentId not in topK results' })
  .option('searchTypes', { type:'string', default: 'semantic,hybrid', describe:'Comma separated search types to test' })
  .help()
  .argv;

const FRONTEND_PORT = String(argv.frontendPort);
const API_BASE = argv.apiBase || `http://localhost:${FRONTEND_PORT}`;
const LOG_FILE = argv.logFile;
const TOP_K = argv.limit;
const THRESHOLD = argv.threshold;
const REQUIRE_ORIGINAL = argv.requireOriginal;
const SEARCH_TYPES = argv.searchTypes.split(',').map(s=>s.trim()).filter(Boolean);

function randomSample(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

async function main(){
  console.log('🧪 Post-batch self test');
  if(!fs.existsSync(LOG_FILE)){
    console.error('Log file not found:', LOG_FILE);
    process.exit(1);
  }
  const lines = fs.readFileSync(LOG_FILE,'utf8').split(/\r?\n/).filter(Boolean);
  const entries = lines.map(l=>{ try{return JSON.parse(l);}catch{return null;} }).filter(e=>e && e.success && e.response && e.response.documentId);
  if(!entries.length){
    console.error('No successful entries with documentId found in log.');
    process.exit(2);
  }
  const chosen = randomSample(entries);
  console.log('Chosen document:', chosen.response.documentId, chosen.filename);

  // Derive basic keywords from filename
  const baseWords = chosen.filename.replace(/[^a-zA-Z0-9 ]/g,' ').split(/\s+/).filter(w=>w.length>3).slice(0,5);
  const query = baseWords.join(' ') || 'legal contract';
  console.log('Query:', query);

  async function doSearch(searchType){
    const res = await fetch(`${API_BASE}/api/rag/search`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query, searchType, limit:TOP_K, threshold:THRESHOLD })
    });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`Search ${searchType} failed ${res.status}: ${t}`);
    }
    return res.json();
  }
  let failures = 0;
  for(const st of SEARCH_TYPES){
    try {
      const result = await doSearch(st);
      const list = result.results || [];
      console.log(`${st} top results:`);
      list.slice(0,TOP_K).forEach((r,i)=>{
        console.log(`  ${i+1}. ${r.filename||r.id} score=${r.similarity?.toFixed?.(3)}`);
      });
      if(REQUIRE_ORIGINAL){
        const foundIndex = list.findIndex(r => r.documentId === chosen.response.documentId || r.id === chosen.response.documentId);
        if(foundIndex === -1){
          console.error(`❌ Original documentId ${chosen.response.documentId} NOT found in top ${TOP_K} (${st})`);
          failures++;
        } else {
          console.log(`✅ Original document found at rank ${foundIndex+1} (${st})`);
        }
      }
      if(list.length){
        const topScore = list[0].similarity || list[0].score;
        if(typeof topScore === 'number' && topScore < THRESHOLD){
          console.error(`❌ Top score ${topScore.toFixed(3)} below threshold ${THRESHOLD}`);
          failures++;
        }
      }
    } catch(e){
      console.error(`${st} search error:`, e.message);
      failures++;
    }
  }

  if(failures){
    console.error(`❌ Self test completed with ${failures} failure(s)`);
    process.exitCode = 3;
  } else {
    console.log('✅ Self test passed');
  }
}
main().catch(e=>{ console.error('❌ Self test failed:', e); process.exit(1); });
