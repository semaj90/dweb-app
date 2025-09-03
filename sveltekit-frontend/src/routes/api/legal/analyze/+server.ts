import type { RequestHandler } from './$types';
import { gemma3Client } from '$lib/gemma3Client';
import { ai_interactions as aiInteractions } from '$lib/server/db/schema-postgres';
import { db } from '$lib/server/db/drizzle';

export const POST: RequestHandler = async ({ request, locals }) => {
  const { prompt, documentId, caseId } = await request.json();
  const start = Date.now();

  const response = await gemma3Client.generate(prompt);
  const responseTime = Date.now() - start;

  await db.insert(aiInteractions).values({
    userId: locals.user.id,
    caseId,
    prompt,
    response: response.text,
    model: 'gemma3',
    responseTime,
    metadata: { documentId }
  });

  return new Response(JSON.stringify({ response: response.text }));
};