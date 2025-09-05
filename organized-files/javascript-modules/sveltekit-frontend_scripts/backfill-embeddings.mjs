#!/usr/bin/env node
// Backfill embeddings for legal_documents and document_sections
// Uses postgres.js + real Ollama nomic-embed-text model with graceful fallback to placeholder
// Now includes rate limiting and connection error handling for production use
import postgres from 'postgres';
import crypto from 'crypto';
// Allow opting into connecting as the Postgres superuser in local dev.
// Set PG_SUPERUSER=1 to prefer the "postgres" user, or supply DB_USER/DB_PASS/DB_HOST/DB_PORT/DB_NAME.
if (!process.env.DATABASE_URL) {
  const useSuper = process.env.PG_SUPERUSER === '1';
  const user = useSuper ? (process.env.DB_USER || 'postgres') : (process.env.DB_USER || 'legal_admin');
  const pass = process.env.DB_PASS || (useSuper ? 'postgres' : '123456');
  const host = process.env.DB_HOST || 'localhost';
  const port = process.env.DB_PORT || '5432';
  const name = process.env.DB_NAME || 'legal_ai_db';
  process.env.DATABASE_URL = `postgresql://${user}:${encodeURIComponent(pass)}@${host}:${port}/${name}`;
}
// Prefer explicit DATABASE_URL; allow opting into the postgres superuser via PG_SUPERUSER
const DATABASE_URL =
  process.env.DATABASE_URL ||
  (process.env.PG_SUPERUSER === '1'
    ? 'postgresql://postgres:postgres@localhost:5432/legal_ai_db'
    : 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db');
const EMBED_DIM = 384; // Keep in sync with migrations & schema
let BATCH_SIZE = parseInt(process.env.BACKFILL_BATCH || '50', 10);
const MIN_BATCH_SIZE = 5;
const RATE_LIMIT_DELAY = parseInt(process.env.OLLAMA_DELAY_MS || '100', 10); // Delay between requests
const TEXT_TRUNCATE_CHARS = parseInt(process.env.TEXT_TRUNCATE_CHARS || '8000', 10);
const MEMORY_SOFT_LIMIT_MB = parseInt(process.env.MEMORY_SOFT_LIMIT_MB || '1400', 10); // adapt below this
const MEMORY_HARD_LIMIT_MB = parseInt(process.env.MEMORY_HARD_LIMIT_MB || '1800', 10); // abort if exceeded
const ENABLE_GC = process.env.ENABLE_GC === '1' && global.gc;

const sql = postgres(DATABASE_URL, { prepare: true });

// Simple sleep function for rate limiting
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function generateEmbedding(text) {
  // Use real Ollama nomic-embed-text model
  const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';

  try {
    const response = await fetch(`${OLLAMA_URL}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        input: text.substring(0, 2048) // Limit token length
      })
    });

    if (!response.ok) {
      console.warn(`⚠️ Ollama API error (${response.status}), falling back to placeholder`);
      return generatePlaceholderEmbedding(text);
    }

    const data = await response.json();

    if (!data.embedding || !Array.isArray(data.embedding)) {
      console.warn('⚠️ Invalid embedding response, falling back to placeholder');
      return generatePlaceholderEmbedding(text);
    }

    return data.embedding;
  } catch (error) {
    console.warn(`⚠️ Ollama connection failed: ${error.message}, using placeholder`);
    return generatePlaceholderEmbedding(text);
  }
}

function generatePlaceholderEmbedding(text) {
  // Deterministic pseudo-embedding fallback for when Ollama is unavailable
  const hash = crypto.createHash('sha256').update(text).digest();
  const floats = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) {
    floats[i] = (hash[i % hash.length] / 255) * 2 - 1; // scale to [-1,1]
  }
  // L2 normalize
  let norm = 0;
  for (let i = 0; i < EMBED_DIM; i++) norm += floats[i] * floats[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < EMBED_DIM; i++) floats[i] /= norm;
  return Array.from(floats);
}

async function logMemory(prefix='mem') {
  const m = process.memoryUsage();
  const rss = (m.rss/1024/1024).toFixed(1);
  const heap = (m.heapUsed/1024/1024).toFixed(1);
  console.log(`  ${prefix} RSS=${rss}MB heap=${heap}MB batch=${BATCH_SIZE}`);
  return { rss: parseFloat(rss), heap: parseFloat(heap) };
}

async function backfill(table, idCol, textCol, embedCol) {
  console.log(`\n▶ Backfilling ${table}.${embedCol} (source: ${textCol}) batch=${BATCH_SIZE}`);
  const pending = await sql`
    SELECT ${sql(idCol)} as id, ${sql(textCol)} as content
    FROM ${sql(table)}
    WHERE ${sql(embedCol)} IS NULL
      AND ${sql(textCol)} IS NOT NULL
    ORDER BY ${sql(idCol)}
    LIMIT ${BATCH_SIZE};`;
  if (pending.length === 0) {
    console.log('✅ No rows needing embeddings.');
    return 0;
  }
  console.log(` • Processing batch of ${pending.length}`);

  let processed = 0;
  for (const row of pending) {
    try {
      const source = (row.content || '').slice(0, TEXT_TRUNCATE_CHARS);
      const emb = await generateEmbedding(source);
      // Build vector literal once (avoid large parameter array overhead)
      const literal = `'[${emb.join(',')}]'`;
      await sql.unsafe(`UPDATE ${table} SET ${embedCol} = ${literal}::vector WHERE ${idCol} = $1`, [row.id]);
      processed++;
    } catch (err) {
      console.warn(`   ⚠️ Row ${row.id} failed: ${err.message}`);
    }

    if (RATE_LIMIT_DELAY > 0) await sleep(RATE_LIMIT_DELAY);

    if (processed % 5 === 0) {
      const { rss } = await logMemory('progress');
      if (rss > MEMORY_SOFT_LIMIT_MB && BATCH_SIZE > MIN_BATCH_SIZE) {
        BATCH_SIZE = Math.max(MIN_BATCH_SIZE, Math.floor(BATCH_SIZE * 0.6));
        console.log(`  🔧 Reduced batch size to ${BATCH_SIZE} due to memory usage.`);
      }
      if (rss > MEMORY_HARD_LIMIT_MB) {
        throw new Error(`Memory hard limit (${MEMORY_HARD_LIMIT_MB}MB) exceeded`);
      }
      if (ENABLE_GC) {
        try { global.gc(); } catch { /* ignore */ }
      }
    }
  }
  console.log(` ✅ Batch applied (${processed} rows)`);
  return processed;
}

(async () => {
  try {
    // Ensure extension exists
    await sql`CREATE EXTENSION IF NOT EXISTS vector;`;
    let total = 0;
    while (true) {
      const beforeLoop = Date.now();
      let changed = 0;
      changed += await backfill('legal_documents', 'id', 'full_text', 'embedding');
      changed += await backfill('document_sections', 'id', 'content', 'embedding');
      total += changed;
      if (changed === 0) break;
      const loopMs = Date.now() - beforeLoop;
      console.log(`  ⏱ Loop processed ${changed} rows in ${loopMs}ms (total ${total})`);
    }
    console.log(`\n🏁 Backfill complete. Rows updated: ${total}`);
  } catch (e) {
    console.error('❌ Backfill error:', e.message);
    process.exit(1);
  } finally {
    await sql.end({ timeout: 5 });
  }
})();
