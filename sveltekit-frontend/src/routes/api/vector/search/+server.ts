// src/routes/api/vector/search/+server.ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { vectorService } from '$lib/server/vector/vectorService.js';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { 
      query, 
      limit = 10, 
      threshold = 0.7, 
      filter = {}, 
      searchType = 'vector',
      includeMetadata = true 
    } = await request.json();
    
    if (!query || typeof query !== 'string') {
      throw error(400, 'Query parameter is required and must be a string');
    }
    
    let results;
    
    switch (searchType) {
      case 'vector':
        results = await vectorService.search(query, {
          limit,
          threshold,
          filter,
          includeMetadata
        });
        break;
        
      case 'hybrid':
        results = await vectorService.hybridSearch(query, {
          limit,
          threshold,
          filter,
          keywordWeight: 0.3,
          vectorWeight: 0.7
        });
        break;
        
      default:
        throw error(400, 'Invalid search type. Use "vector" or "hybrid"');
    }
    
    return json({
      success: true,
      query,
      searchType,
      resultsCount: results.length,
      results,
      timestamp: new Date().toISOString()
    });
    
  } catch (err) {
    console.error('Vector search error:', err);
    
    if (err instanceof Error && err.message.includes('Qdrant')) {
      throw error(503, 'Vector search service unavailable');
    }
    
    throw error(500, `Search failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get('q');
    const limit = parseInt(url.searchParams.get('limit') || '10');
    const searchType = url.searchParams.get('type') || 'vector';
    
    if (!query) {
      return json({
        message: 'Vector search endpoint',
        methods: ['GET', 'POST'],
        parameters: {
          q: 'Search query (required)',
          limit: 'Number of results (default: 10)',
          type: 'Search type: vector or hybrid (default: vector)'
        },
        example: '/api/vector/search?q=criminal+case&limit=5&type=hybrid'
      });
    }
    
    let results;
    
    switch (searchType) {
      case 'vector':
        results = await vectorService.search(query, { limit });
        break;
      case 'hybrid':
        results = await vectorService.hybridSearch(query, { limit });
        break;
      default:
        throw error(400, 'Invalid search type. Use "vector" or "hybrid"');
    }
    
    return json({
      success: true,
      query,
      searchType,
      resultsCount: results.length,
      results,
      timestamp: new Date().toISOString()
    });
    
  } catch (err) {
    console.error('Vector search error:', err);
    throw error(500, `Search failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};