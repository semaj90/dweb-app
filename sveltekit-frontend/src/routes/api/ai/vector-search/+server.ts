
import { json } from '@sveltejs/kit';
import { getEmbeddingRepository } from '$lib/server/embedding/embedding-repository';

// Minimal vector search endpoint leveraging pgvector embedding repository
export async function POST({ request }) {
  try {
    const { query, limit = 8, model } = await request.json();
    if (!query || typeof query !== 'string') return json({ error: 'query required' }, { status: 400 });
    const repo = getEmbeddingRepository();
    const results = await repo.querySimilar(query, { limit, model });
    return json({ results, count: results.length });
  } catch (e) {
    console.error('vector-search error', e);
    return json({ error: 'internal error' }, { status: 500 });
  }
}

export async function GET() {
  return json({ status: 'ok', message: 'POST { query, limit?, model? }' });
}
