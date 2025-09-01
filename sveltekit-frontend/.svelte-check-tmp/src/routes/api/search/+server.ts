import type { RequestHandler } from './$types';
import { db } from '$lib/db';
import { sql } from 'drizzle-orm';

async function getEmbedding(text: string): Promise<number[]> {
  const response = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'nomic-embed-text', prompt: text })
  });
  const data = await response.json();
  return data.embedding;
}

export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) return new Response('Unauthorized', { status: 401 });
  
  const { query } = await request.json();
  const queryEmbedding = await getEmbedding(query);
  
  const results = await db.execute(sql`
    SELECT id, filename, content, summary,
           1 - (embedding <=> ${JSON.stringify(queryEmbedding)}::vector) as similarity
    FROM documents 
    WHERE user_id = ${locals.user.id}
    ORDER BY embedding <=> ${JSON.stringify(queryEmbedding)}::vector
    LIMIT 10
  `);

  return new Response(JSON.stringify(results.rows));
};
