/**
 * Batch Legal PDF Processor (Node.js Compatible)
 * Uploads and processes all PDF documents from the lawpdfs folder
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import FormData from 'form-data';
import pLimit from 'p-limit';
import { createSpinner } from 'nanospinner';

// Lazy optional dependency load for p-retry (avoid hard crash if not installed yet)
let pRetry; // function wrapper
async function withRetry(fn, options) {
  if (!pRetry) {
    try {
      const mod = await import('p-retry');
      pRetry = mod.default || mod; // handle CJS/ESM
    } catch (err) {
      // Fallback simple manual retry
      return simpleRetry(fn, options?.retries || 3, options?.minTimeout || 500);
    }
  }
  return pRetry(fn, options);
}

async function simpleRetry(fn, retries = 3, delayMs = 500) {
  let lastErr;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try { return await fn({ attempt }); } catch (e) {
      lastErr = e;
      if (attempt < retries) await sleep(delayMs * attempt);
    }
  }
  throw lastErr;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration (env overridable)
const FRONTEND_PORT = process.env.FRONTEND_PORT || process.env.PORT || '5173';
const API_BASE = process.env.RAG_BASE_URL || `http://localhost:${FRONTEND_PORT}`;
const RAG_API_URL = process.env.RAG_PROCESS_URL || `${API_BASE}/api/rag/process`;
const DRY_RUN = (process.env.DRY_RUN || 'true').toLowerCase() === 'true';
const LIMIT = parseInt(process.env.LIMIT || '5', 10);
const CONCURRENCY = parseInt(process.env.CONCURRENCY || '2', 10);
const LAWPDFS_FOLDER = path.join(__dirname, '../../lawpdfs');
const LOG_DIR = path.join(__dirname, '../logs');
const LOG_FILE = path.join(LOG_DIR, 'batch-upload-log.jsonl');
const SUMMARY_FILE = path.join(LOG_DIR, 'batch-upload-summary.json');
const RESUME = (process.env.RESUME || 'true').toLowerCase() === 'true';
const RETRIES = parseInt(process.env.RETRIES || '3', 10);
const RETRY_MIN_TIMEOUT = parseInt(process.env.RETRY_MIN_TIMEOUT || '750', 10);

// Simple delay function
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Load already processed docs for resume
function loadProcessedMaps() {
  if (!RESUME || !fs.existsSync(LOG_FILE)) return { filenames: new Set(), docIds: new Set() };
  try {
    const lines = fs.readFileSync(LOG_FILE, 'utf8').split(/\r?\n/).filter(Boolean);
    const filenames = new Set();
    const docIds = new Set();
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.success) {
          if (entry.filename) filenames.add(entry.filename);
          if (entry.response?.documentId) docIds.add(entry.response.documentId);
          else if (entry.response?.id) docIds.add(entry.response.id);
        }
      } catch (_) { /* ignore */ }
    }
    return { filenames, docIds };
  } catch (e) { return { filenames: new Set(), docIds: new Set() }; }
}

function ensureLogDir() {
  if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
}

function appendLog(entry) {
  ensureLogDir();
  fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + '\n');
}

async function uploadDocument(filePath, filename) {
  try {
    console.log(`📄 Processing: ${filename}`);
    const size = fs.statSync(filePath).size;
    console.log(`   📁 File path: ${filePath}`);
    console.log(`   📊 File size: ${size} bytes`);

    if (DRY_RUN) {
      await sleep(300);
      console.log('✅ DRY-RUN: Skipped real upload (set DRY_RUN=false to enable)');
      return { success: true, filename, dryRun: true };
    }

    const fd = new FormData();
    fd.append('file', fs.createReadStream(filePath), { filename, contentType: 'application/pdf' });
    fd.append('source', 'batch-script');
    fd.append('original_name', filename);
    const controller = new AbortController();
    const timeout = setTimeout(()=>controller.abort(), 30000);
  const res = await fetch(RAG_API_URL, { method: 'POST', body: fd, signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Upload failed ${res.status} ${res.statusText}: ${text.slice(0,200)}`);
    }
    const json = await res.json().catch(()=>({}));
    console.log(`✅ Uploaded: ${filename}`);
    if (json.documentId || json.id) console.log(`   🆔 ID: ${json.documentId || json.id}`);
    if (json.embeddingGenerated) console.log('   � Embeddings generated');
  return { success: true, filename, response: json };

  } catch (error) {
    console.log(`❌ Failed to process: ${filename}`);
    console.log(`   Error: ${error.message}`);
  return { success: false, filename, error: error.message };
  }
}

async function main() {
  console.log('🚀 Starting Legal PDF Batch Processing');
  console.log(`📁 Source folder: ${LAWPDFS_FOLDER}`);
  console.log(`⚙️  Config: LIMIT=${LIMIT} DRY_RUN=${DRY_RUN} CONCURRENCY=${CONCURRENCY} RETRIES=${RETRIES} RESUME=${RESUME}`);

  // Check if folder exists
  if (!fs.existsSync(LAWPDFS_FOLDER)) {
    console.log(`❌ Folder not found: ${LAWPDFS_FOLDER}`);
    process.exit(1);
  }

  // Get all PDF files
  const files = fs.readdirSync(LAWPDFS_FOLDER)
    .filter(file => file.toLowerCase().endsWith('.pdf'))
    .sort();

  console.log(`📄 Found ${files.length} PDF documents to process`);
  console.log('');

  const results = {
    successful: [],
    failed: [],
    startTime: new Date()
  };

  // Process first LIMIT files as a test
  const processed = loadProcessedMaps();
  let testFiles = files.slice(0, LIMIT);
  if (RESUME) {
    const before = testFiles.length;
    testFiles = testFiles.filter(f => !processed.filenames.has(f));
    const skipped = before - testFiles.length;
    if (skipped > 0) console.log(`⏩ Resume (by filename) skipped ${skipped}`);
  }
  console.log(`🧪 Processing first ${testFiles.length} files (LIMIT=${LIMIT}, DRY_RUN=${DRY_RUN})`);
  console.log('');

  const limit = pLimit(CONCURRENCY);
  let completed = 0;
  const startSpinner = createSpinner('Processing batch').start();
  const tasks = testFiles.map((filename) => limit(async () => {
    const filePath = path.join(LAWPDFS_FOLDER, filename);
    const startedAt = Date.now();
    let attemptResult;
    try {
      attemptResult = await withRetry(() => uploadDocument(filePath, filename), {
        retries: RETRIES,
        minTimeout: RETRY_MIN_TIMEOUT,
        onFailedAttempt: (err) => {
          console.log(`   🔄 Retry ${err.attemptNumber}/${err.retriesLeft + err.attemptNumber} for ${filename}: ${err.message}`);
        }
      });
    } catch (finalErr) {
      attemptResult = { success: false, filename, error: finalErr.message };
    }
    const durationMs = Date.now() - startedAt;
  const logEntry = { timestamp: new Date().toISOString(), filename, success: attemptResult.success, durationMs, dryRun: DRY_RUN };
  if (attemptResult.response) logEntry.response = attemptResult.response;
  if (attemptResult.error) logEntry.error = attemptResult.error;
    appendLog(logEntry);
    if (attemptResult.success) results.successful.push({ ...attemptResult, durationMs }); else results.failed.push({ ...attemptResult, durationMs });
    completed++;
    const pct = Math.round((completed / testFiles.length) * 100);
    const barWidth = 20;
    const filled = Math.round((pct / 100) * barWidth);
    const bar = '█'.repeat(filled) + '░'.repeat(barWidth - filled);
    startSpinner.update({ text: `Processing ${completed}/${testFiles.length} ${pct}% |${bar}|` });
  }));
  await Promise.all(tasks);
  startSpinner.success({ text: `Completed ${testFiles.length} files` });

  // Display results
  const endTime = new Date();
  const totalTime = ((endTime - results.startTime) / 1000).toFixed(1);

  console.log('📊 TEST BATCH PROCESSING COMPLETE');
  console.log('');
  console.log(`⏰ Total time: ${totalTime} seconds`);
  console.log(`✅ Successful: ${results.successful.length}`);
  console.log(`❌ Failed: ${results.failed.length}`);
  const avgMs = results.successful.length ? Math.round(results.successful.reduce((a,b)=>a+b.durationMs,0)/results.successful.length) : 0;
  console.log(`⚡ Avg success duration: ${avgMs} ms`);

  // Summary table
  function pad(str,len){ str=String(str); return str.length>=len?str.slice(0,len):str+' '.repeat(len-str.length);}
  const header = ['File','Success','Time(ms)','DocID'];
  const colWidths = [30,8,10,20];
  console.log('\n┌' + colWidths.map(w=>'─'.repeat(w+2)).join('┬') + '┐');
  function row(cols){ console.log('│' + cols.map((c,i)=>' '+pad(c,colWidths[i])+' ').join('│') + '│'); }
  row(header);
  console.log('├' + colWidths.map(w=>'─'.repeat(w+2)).join('┼') + '┤');
  [...results.successful, ...results.failed].slice(0,50).forEach(r=>{
    const docId = r.response?.documentId || r.response?.id || '';
    row([r.filename, r.success?'yes':'no', r.durationMs || '', docId]);
  });
  console.log('└' + colWidths.map(w=>'─'.repeat(w+2)).join('┴') + '┘');

  // Persist summary JSON
  ensureLogDir();
  const summary = { totalFiles: files.length, attempted: testFiles.length, success: results.successful.length, failed: results.failed.length, avgSuccessMs: avgMs, totalTimeSeconds: Number(totalTime), config: { LIMIT, DRY_RUN, CONCURRENCY, RETRIES, RESUME } };
  fs.writeFileSync(SUMMARY_FILE, JSON.stringify(summary,null,2));
  console.log(`📄 Summary saved: ${SUMMARY_FILE}`);

  console.log('');
  console.log('📝 Available documents for RAG processing:');
  files.forEach((file, index) => {
    console.log(`   ${index + 1}. ${file}`);
  });

  console.log('');
  console.log('🎯 Next steps:');
  console.log(`   1. Use the web interface at ${API_BASE}/upload-test (or simple-upload-test if available)`);
  console.log('   2. Upload PDFs manually to test the real database integration');
  console.log('   3. Search the uploaded documents using semantic search');
  console.log('   4. Run full batch processing when ready');

  console.log('');
  console.log('🎉 Ready for legal document processing!');
}

// Run the processor
main().catch(error => {
  console.log(`❌ Fatal error: ${error.message}`);
  process.exit(1);
});
