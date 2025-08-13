#!/usr/bin/env node
import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'
import fetch from 'node-fetch'
// Work around pdf-parse CJS debug harness by importing the wrapper
import pdf from 'pdf-parse/lib/pdf-parse.js'
import { Pool } from 'pg'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Config
const PG_DSN = process.env.PG_CONN_STRING || process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
const OLLAMA_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434'
const RAG_URL = process.env.RAG_URL || 'http://localhost:8093'
const MODEL = process.env.EMBED_MODEL || 'nomic-embed-text'
const SEED_LIMIT = process.env.SEED_LIMIT ? parseInt(process.env.SEED_LIMIT, 10) : 0
const SEED_PDF_DIR = process.env.SEED_PDF_DIR

const candidates = [
  SEED_PDF_DIR,
  path.resolve(__dirname, '..', 'lawpdfs'),
  path.resolve(__dirname, '..', 'sample-data', 'legal-documents', 'pdf'),
  path.resolve(__dirname, '..', 'ai-summary-service', 'uploads')
].filter(Boolean)

function isPdf(file) { return file.toLowerCase().endsWith('.pdf') }

function looksLikePdf(buf) {
  if (!buf || buf.length < 4) return false
  const hdr = buf.subarray(0, 4).toString('utf8')
  return hdr.startsWith('%PDF')
}

function stripHtml(input) {
  const s = input.toString('utf8')
  // Remove scripts/styles
  let out = s.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
  out = out.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
  // Replace tags with spaces
  out = out.replace(/<[^>]+>/g, ' ')
  // Decode basic entities
  out = out.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
  // Collapse whitespace
  out = out.replace(/\s+/g, ' ').trim()
  return out
}

async function exists(p) {
  try { await fs.access(p); return true } catch { return false }
}

async function walk(dir) {
  const out = []
  const entries = await fs.readdir(dir, { withFileTypes: true })
  for (const e of entries) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) out.push(...(await walk(full)))
    else if (isPdf(full)) out.push(full)
  }
  return out
}

async function getEmbedViaRag(text) {
  const resp = await fetch(`${RAG_URL}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts: [text], model: MODEL })
  })
  if (!resp.ok) throw new Error(`rag-kratos /embed ${resp.status}`)
  const data = await resp.json()
  const vec = data?.vectors?.[0]
  if (!Array.isArray(vec)) throw new Error('no vectors from rag-kratos')
  return vec
}

async function getEmbedViaOllama(text, model) {
  const m = model || MODEL
  const resp = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: m, input: text })
  })
  if (!resp.ok) throw new Error(`ollama embeddings ${resp.status}`)
  const data = await resp.json()
  const vec = data?.embedding || data?.data?.[0]?.embedding
  if (!Array.isArray(vec)) throw new Error('no embedding vector from ollama')
  return vec
}

async function tryOllamaModels(text) {
  const candidates = [
    MODEL,
    'nomic-embed-text',
    'all-minilm',
    'bge-m3',
    'mxbai-embed-large',
    'gte-small',
    'gte-base',
    'gte-large'
  ].filter(Boolean)
  const tried = new Set()
  for (const m of candidates) {
    if (tried.has(m)) continue
    tried.add(m)
    try {
      const v = await getEmbedViaOllama(text, m)
      if (Array.isArray(v) && v.length > 0) {
        return { vec: v, model: m }
      }
    } catch (e) {
      // continue
    }
  }
  throw new Error('no embedding model produced a non-empty vector')
}

async function getEmbedViaXenova(text) {
  // Dynamic import so script works even if package not installed
  const { pipeline } = await import('@xenova/transformers')
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
  const result = await extractor(text, { pooling: 'mean', normalize: true })
  const arr = Array.from(result.data)
  return arr
}

function toVectorLiteral(vec) {
  // pgvector accepts text input like "[1,2,3]"; we'll cast to vector in SQL
  return `[${vec.join(',')}]`
}

function ensureDim(vec, target) {
  if (!Array.isArray(vec)) return []
  if (typeof target !== 'number' || target <= 0) return vec
  if (vec.length === target) return vec
  if (vec.length > target) return vec.slice(0, target)
  // pad with zeros
  const out = vec.slice()
  while (out.length < target) out.push(0)
  return out
}

async function main() {
  const srcDirs = []
  for (const c of candidates) {
    if (await exists(c)) srcDirs.push(c)
  }
  if (srcDirs.length === 0) {
    console.error('No seed directory found. Set SEED_PDF_DIR or create ./lawpdfs or use ./sample-data/legal-documents/pdf')
    process.exit(1)
  }
  console.log(`Seeding PDFs from:`)
  srcDirs.forEach((d) => console.log(' -', d))
  const all = new Set()
  for (const d of srcDirs) {
    for (const f of await walk(d)) all.add(f)
  }
  const pdfs = Array.from(all)
  if (pdfs.length === 0) {
    console.log('No PDF files found to seed.')
    return
  }
  if (SEED_LIMIT > 0 && pdfs.length > SEED_LIMIT) {
    console.log(`Limiting to first ${SEED_LIMIT} files (of ${pdfs.length}) for this run`)
  }

  const pool = new Pool({ connectionString: PG_DSN })
  const client = await pool.connect()
  let ok = 0, fail = 0
  try {
    await client.query('CREATE EXTENSION IF NOT EXISTS vector')
  const files = SEED_LIMIT > 0 ? pdfs.slice(0, SEED_LIMIT) : pdfs
  for (const f of files) {
      try {
        const buf = await fs.readFile(f)
        let text = ''
        if (looksLikePdf(buf)) {
          try {
            const parsed = await pdf(buf)
            text = (parsed.text || '').trim()
          } catch (e) {
            console.warn('pdf parse failed, trying html/text fallback:', f, e.message)
          }
        }
        if (!text) {
          // Fallback: try interpret as HTML or plain text
          const sample = buf.subarray(0, Math.min(buf.length, 4096)).toString('utf8')
          if (sample.toLowerCase().includes('<html') || sample.toLowerCase().includes('<!doctype')) {
            text = stripHtml(buf)
          } else {
            // assume utf8 text-ish
            text = buf.toString('utf8')
          }
          text = (text || '').trim()
        }
        if (!text) { console.warn('skip empty/unreadable file:', f); continue }
        const snippet = text.length > 16000 ? text.slice(0, 16000) : text
        let vec
        let chosenModel = 'rag-kratos'
        try {
          vec = await getEmbedViaRag(snippet)
        } catch (e) {
          console.warn('rag-kratos embed failed, trying local ollama models:', e.message)
          const r = await tryOllamaModels(snippet)
          vec = r.vec
          chosenModel = r.model
        }
        if (!Array.isArray(vec) || vec.length === 0) {
          try {
            vec = await getEmbedViaXenova(snippet)
            chosenModel = 'xenova:all-MiniLM-L6-v2'
          } catch (e) {
            console.warn('xenova fallback failed:', e.message)
          }
          if (!Array.isArray(vec) || vec.length === 0) {
            console.warn('skip due to empty embedding vector:', f)
            continue
          }
        }
        const vlit = toVectorLiteral(vec)
        const title = path.basename(f)
        const caseId = 'SEED:LAWPDFS'
        const metadata = { source: path.relative(path.resolve(__dirname, '..'), f), seededAt: new Date().toISOString(), model: chosenModel }
        try {
          await client.query(
            `INSERT INTO legal_documents (title, content, case_id, embedding, metadata)
             VALUES ($1, $2, $3, $4::vector, $5)`,
            [title, text, caseId, vlit, metadata]
          )
        } catch (e) {
          // Handle dimension mismatch by parsing expected dimension and retrying
          const msg = String(e.message || e)
          const m1 = msg.match(/expected\s+(\d+)\s+dimensions/i) || msg.match(/\b(\d+)\s+dimensions\b/i)
          if (m1) {
            const target = parseInt(m1[1], 10)
            if (Number.isFinite(target) && target > 0) {
              const adj = ensureDim(vec, target)
              if (adj.length === target) {
                const vlit2 = toVectorLiteral(adj)
                await client.query(
                  `INSERT INTO legal_documents (title, content, case_id, embedding, metadata)
                   VALUES ($1, $2, $3, $4::vector, $5)`,
                  [title, text, caseId, vlit2, metadata]
                )
              } else {
                throw e
              }
            } else {
              throw e
            }
          } else {
            throw e
          }
        }
        ok++
        if (ok % 5 === 0) console.log(`Inserted ${ok}/${pdfs.length}`)
      } catch (e) {
        fail++
        console.error('Failed:', f, e.message)
      }
    }
    console.log(`Seed complete. inserted=${ok} failed=${fail}`)
    // Optional analyze to update stats
    await client.query('ANALYZE legal_documents')
  } finally {
    client.release()
    await pool.end()
  }
}

main().catch((e) => { console.error(e); process.exit(1) })
