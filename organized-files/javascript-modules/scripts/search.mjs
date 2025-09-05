#!/usr/bin/env node
import { QdrantClient } from '@qdrant/js-client-rest'

const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333'
const COLLECTION = process.env.QDRANT_COLLECTION || 'dev_embeddings'
const MODEL = process.env.EMBED_MODEL || 'nomic-embed-text'

async function embedQuery(text) {
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

async function main() {
  const query = process.argv.slice(2).join(' ') || 'contract liability terms'
  const limit = Number(process.env.LIMIT || 5)

  const client = new QdrantClient({ url: QDRANT_URL })
  const vector = await embedQuery(query)

  const res = await client.search(COLLECTION, {
    vector,
    limit,
    with_payload: true
  })

  console.log(JSON.stringify(res, null, 2))
}

main().catch((e) => { console.error(e); process.exit(1) })
