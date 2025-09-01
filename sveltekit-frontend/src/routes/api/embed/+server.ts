import type { RequestHandler } from './$types.js';
import { db } from '$lib/db';
import { documents } from '$lib/db/schema';
import crypto from "crypto";

async function getEmbedding(text: string): Promise<number[]> {
  const response = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'nomic-embed-text',
      prompt: text
    })
  });
  const data = await response.json();
  return data.embedding;
}

export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) return new Response('Unauthorized', { status: 401 });
  
  const { title, content } = await request.json();
  const embedding = await getEmbedding(content);
  
  const doc = await db.insert(documents).values({
    id: crypto.randomUUID(),
    filename: title,
    content,
    embedding: JSON.stringify(embedding),
    user_id: parseInt(locals.user.id)
  }).returning();

  return new Response(JSON.stringify(doc[0]));
};
