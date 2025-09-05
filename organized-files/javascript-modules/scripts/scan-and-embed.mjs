#!/usr/bin/env node
import { QdrantClient } from '@qdrant/js-client-rest'
import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const SRC_DIR = process.env.SCAN_DIR || path.join(__dirname, '..', 'data', 'dev-docs')
const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333'
const COLLECTION = process.env.QDRANT_COLLECTION || 'dev_embeddings'
const MODEL = process.env.EMBED_MODEL || 'nomic-embed-text'

async function embed(text) {
  const resp = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, input: text })
  })
  if (!resp.ok) throw new Error(`Ollama embeddings failed: ${resp.status}`)
  const data = await resp.json()
  const vector = data?.embedding || data?.data?.[0]?.embedding
  if (!Array.isArray(vector)) throw new Error('No embedding vector in response')
  return vector
}

async function ensureCollection(client, size) {
  try {
    await client.getCollection(COLLECTION)
    return
  } catch {}
  await client.createCollection(COLLECTION, {
    vectors: { size, distance: 'Cosine' }
  })
}

async function walk(dir) {
  const out = []
  const entries = await fs.readdir(dir, { withFileTypes: true })
  for (const e of entries) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) out.push(...(await walk(full)))
    else out.push(full)
  }
  return out
}

async function main() {
  console.log(`Scanning: ${SRC_DIR}`)
  const files = await walk(SRC_DIR)
  if (files.length === 0) {
    console.log('No files found to embed.')
    return
  }

  const client = new QdrantClient({ url: QDRANT_URL })

  // Probe first file for vector size
  const sampleText = await fs.readFile(files[0], 'utf8')
  const sampleVec = await embed(sampleText.slice(0, 4000))
  await ensureCollection(client, sampleVec.length)

  let count = 0
  for (const f of files) {
    try {
      const content = await fs.readFile(f, 'utf8')
      const vector = await embed(content.slice(0, 4000))
      const id = `${path.relative(SRC_DIR, f)}:${Date.now()}`
      const payload = {
        path: path.relative(SRC_DIR, f),
        mtime: (await fs.stat(f)).mtime.toISOString(),
        size: (await fs.stat(f)).size,
        model: MODEL
      }
      await client.upsert(COLLECTION, {
        points: [ { id, vector, payload } ]
      })
      count++
      if (count % 10 === 0) console.log(`Indexed ${count}/${files.length}`)
    } catch (e) {
      console.error('Failed to index', f, e.message)
    }
  }
  console.log(`Done. Indexed ${count} files into ${COLLECTION}.`)
}

main().catch((e) => { console.error(e); process.exit(1) })
