// Enhanced Vector Search API - Auto-generated
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { vectorService } from '$lib/server/vector/EnhancedVectorService.js';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options = {} } = await request.json();
    
    if (!query) throw error(400, 'Query required');
    
    const results = await vectorService.hybridSearch(query, options);
    
    return json({
      success: true,
      query,
      results,
      count: results.length,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('Vector search error:', err);
    throw error(500, `Search failed: ${err.message}`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const query = url.searchParams.get('q');
  if (!query) {
    return json({
      message: 'Vector Search API',
      usage: 'POST /api/vector/search with { query, options }',
      health: await vectorService.healthCheck()
    });
  }
  
  const results = await vectorService.hybridSearch(query);
  return json({ success: true, query, results });
};