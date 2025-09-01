import type { RequestHandler } from './$types';
import { embeddingService } from '$lib/services/embedding';
import { hybridSemanticSearch } from '$lib/database/connection';

export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) return new Response('Unauthorized', { status: 401 });
  
  const { query, caseId } = await request.json();
  const embedding = await embeddingService.generateEmbedding(query);
  
  const results = await hybridSemanticSearch(query, embedding, {
    limit: 20,
    threshold: 0.7,
    caseId,
    userId: locals.user.id
  });

  return new Response(JSON.stringify(results));
};