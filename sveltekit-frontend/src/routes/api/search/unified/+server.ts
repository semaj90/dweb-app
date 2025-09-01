import { json } from '@sveltejs/kit';
import { z } from 'zod';
import { globalSearch } from '$lib/services/search-service';
import { hybridVectorService } from '$lib/services/hybrid-vector-operations';
import { unifiedSearch } from '$lib/services/unified-loki-fuzzy-search.js';
import type { SearchResult } from '$lib/components/search/types';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


const unifiedSearchSchema = z.object({
  query: z.string().min(1).max(1000),
  categories: z.array(z.enum(['cases', 'evidence', 'precedents', 'statutes', 'criminals', 'documents', 'services', 'components'])).optional(),
  enableVectorSearch: z.boolean().default(true),
  aiSuggestions: z.boolean().default(true),
  maxResults: z.number().min(1).max(100).default(20),
  similarityThreshold: z.number().min(0).max(1).default(0.7),
  includeMetadata: z.boolean().default(true)
});

/**
 * Reusable handler for the unified search flow.
 * Accepts validated search params and returns a Response via json(...)
 */
async function handleUnifiedSearch(searchParams: z.infer<typeof unifiedSearchSchema>, locals: any) {
  const { query, categories, enableVectorSearch, maxResults, similarityThreshold } = searchParams;

  let results: SearchResult[] = [];
  const startTime = Date.now();

  // 1. Unified Loki.js Fuzzy Search (NEW: Primary search with legal context)
  let lokiResults: SearchResult[] = [];
  try {
    const lokiSearchResults = await unifiedSearch.search(query, {
      threshold: 0.4,
      limit: Math.ceil(maxResults / 2),
      legalContext: categories?.includes('cases') || categories?.includes('evidence') ? 'legal' : undefined,
      userId: locals.user?.id
    });

    lokiResults = lokiSearchResults.map(result => ({
      id: `loki-${result.id}`,
      title: result.content.title || result.content.name || 'Untitled',
      type: result.type === 'legal_document' ? 'document' : result.type,
      content: typeof result.content === 'string' ? result.content :
        result.content.description || result.content.content || '',
      score: result.score,
      metadata: {
        source: `loki-${result.source}`,
        legalContext: result.metadata?.legalContext,
        confidence: result.metadata?.confidence,
        lastAccessed: result.metadata?.lastAccessed,
        ...result.metadata
      }
    }));
  } catch (lokiError) {
    console.warn('Loki fuzzy search failed, continuing with other search methods:', lokiError);
  }

  // 2. Service and Component Search (using Fuse.js)
  const serviceResults = await globalSearch(query, { limit: Math.ceil(maxResults / 3) });
  const mappedServiceResults: SearchResult[] = serviceResults.map(result => ({
    id: `service-${result.id}`,
    title: result.title,
    type: result.category === 'goBinaries' ? 'service' : 'component',
    content: result.content || result.description || '',
    score: 1 - result.score, // Convert Fuse score to similarity score
    metadata: {
      category: result.category,
      tags: result.tags || [],
      port: result.port,
      status: result.status
    },
    highlights: result.highlights
  }));

  // 2. Vector Search for Legal Entities (if enabled)
  let vectorResults: SearchResult[] = [];
  if (enableVectorSearch) {
    try {
      // Generate embedding for the query
      const { legalNLP } = await import('$lib/services/sentence-transformer');
      const queryEmbedding = await legalNLP.embedText(query);

      // Perform hybrid vector search across cases, evidence, documents
      const hybridResults = await hybridVectorService.hybridVectorSearch(
        queryEmbedding,
        {
          threshold: similarityThreshold,
          limit: Math.ceil(maxResults / 2),
          useQdrant: true,
          usePgVector: true,
          hybridWeights: { pgvector: 0.6, qdrant: 0.4 }
        }
      );

      vectorResults = hybridResults.map(result => ({
        id: result.id,
        title: result.title || 'Untitled Document',
        type: result.type || 'document',
        content: result.content || result.excerpt || '',
        score: result.similarity,
        similarity: result.similarity,
        metadata: {
          ...result.metadata,
          source: result.source || 'vector_db'
        },
        highlights: result.highlights,
        createdAt: result.created_at || result.createdAt
      }));
    } catch (vectorError) {
      console.warn('Vector search failed, continuing with service search only:', vectorError);
    }
  }

  // 3. Combine and rank results (now including Loki.js fuzzy search)
  results = [...lokiResults, ...mappedServiceResults, ...vectorResults];

  // 4. Filter by categories if specified
  if (categories && categories.length > 0) {
    results = results.filter(result =>
      categories.includes(result.type as any) ||
      categories.some(cat => result.metadata?.category?.includes(cat))
    );
  }

  // 5. Sort by relevance score and limit results
  results = results
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, maxResults);

  // 6. Generate search metadata
  const processingTime = Date.now() - startTime;
  const searchMetadata = {
    query,
    categories: categories || ['all'],
    totalResults: results.length,
    processingTime,
    vectorSearchUsed: enableVectorSearch,
    lokiFuzzySearchUsed: lokiResults.length > 0,
    aiEnhanced: searchParams.aiSuggestions,
    sourceBreakdown: {
      loki: lokiResults.length,
      services: mappedServiceResults.length,
      vector: vectorResults.length
    }
  };

  return json({
    success: true,
    results,
    metadata: searchMetadata,
    suggestions: [] // TODO: Add AI-generated suggestions
  });
}

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Parse and validate request
    const body = await request.json();
    const searchParams = unifiedSearchSchema.parse(body);
    return await handleUnifiedSearch(searchParams, locals);
  } catch (error: any) {
    console.error('Unified search API error:', error);

    if (error instanceof z.ZodError) {
      return json(
        {
          success: false,
          error: 'Invalid search parameters',
          details: error.errors
        },
        { status: 400 }
      );
    }

    return json(
      {
        success: false,
        error: 'Search failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const query = url.searchParams.get('q');
    const categories = url.searchParams.get('categories')?.split(',') || [];
    const limit = parseInt(url.searchParams.get('limit') || '20');
    const threshold = parseFloat(url.searchParams.get('threshold') || '0.7');
    const vectorSearch = url.searchParams.get('vector') !== 'false';

    if (!query) {
      return json({ success: false, error: 'Query parameter (q) required' }, { status: 400 });
    }

    // Build body from GET parameters and validate using the same schema
    const body = {
      query,
      categories: categories.length > 0 ? categories : undefined,
      enableVectorSearch: vectorSearch,
      maxResults: limit,
      similarityThreshold: threshold,
      aiSuggestions: true,
      includeMetadata: true
    };

    const searchParams = unifiedSearchSchema.parse(body);
    return await handleUnifiedSearch(searchParams, locals);

  } catch (error: any) {
    console.error('Unified search GET API error:', error);

    if (error instanceof z.ZodError) {
      return json(
        {
          success: false,
          error: 'Invalid search parameters',
          details: error.errors
        },
        { status: 400 }
      );
    }

    return json(
      { success: false, error: 'Search failed' },
      { status: 500 }
    );
  }
};