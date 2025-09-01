/**
 * Gallery Search API - Advanced Search and Filtering
 * Provides comprehensive search capabilities across all gallery content
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/database';
import { evidence, cases, users } from '$lib/server/db/schema';
import { eq, desc, asc, and, or, like, ilike, gte, lte, inArray, sql } from 'drizzle-orm';
import { URL } from "url";

interface SearchFilters {
  query?: string;
  types?: string[];
  categories?: string[];
  caseIds?: string[];
  userIds?: string[];
  tags?: string[];
  dateFrom?: string;
  dateTo?: string;
  fileSizeMin?: number;
  fileSizeMax?: number;
  fileTypes?: string[];
  hasOCR?: boolean;
  hasEmbedding?: boolean;
  isProcessed?: boolean;
  isPublic?: boolean;
  contentSearch?: boolean; // Search in OCR text and content
}

interface SearchOptions {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  includeMetadata?: boolean;
  includeContent?: boolean;
}

interface SearchResult {
  id: string;
  type: string;
  title: string;
  description?: string;
  fileName: string;
  fileType: string;
  fileSize: number;
  url: string;
  thumbnailUrl?: string;
  uploadedAt: string;
  caseId?: string;
  caseTitle?: string;
  tags: string[];
  metadata?: Record<string, any>;
  relevanceScore?: number;
  matchedFields: string[];
  snippet?: string;
}

interface SearchResponse {
  results: SearchResult[];
  totalCount: number;
  searchTime: number;
  facets: {
    types: Array<{ name: string; count: number }>;
    fileTypes: Array<{ name: string; count: number }>;
    cases: Array<{ id: string; title: string; count: number }>;
    tags: Array<{ name: string; count: number }>;
    dateRanges: Array<{ range: string; count: number }>;
  };
  suggestions?: string[];
  pagination: {
    page: number;
    pageSize: number;
    totalPages: number;
  };
}

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const { filters, options }: { filters: SearchFilters; options: SearchOptions } = await request.json();
    
    const startTime = Date.now();
    
    // Default options
    const page = options.page || 1;
    const pageSize = Math.min(options.pageSize || 20, 100);
    const sortBy = options.sortBy || 'uploadedAt';
    const sortOrder = options.sortOrder || 'desc';
    const includeMetadata = options.includeMetadata !== false;
    const includeContent = options.includeContent || false;

    // Build the base query
    const baseQuery = db
      .select({
        id: evidence.id,
        title: evidence.title,
        description: evidence.description,
        fileName: evidence.fileName,
        originalFileName: evidence.originalFileName,
        fileType: evidence.fileType,
        fileSize: evidence.fileSize,
        filePath: evidence.filePath,
        uploadedAt: evidence.uploadedAt,
        processedAt: evidence.processedAt,
        caseId: evidence.caseId,
        caseTitle: cases.title,
        tags: evidence.tags,
        metadata: includeMetadata ? evidence.metadata : sql`NULL`,
        ocrText: includeContent ? evidence.ocrText : sql`NULL`,
        contentText: includeContent ? evidence.contentText : sql`NULL`,
        embedding: evidence.embedding,
        isPublic: evidence.isPublic
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id));

    // Build WHERE conditions
    const conditions = await buildSearchConditions(filters);
    
    if (conditions.length > 0) {
      baseQuery.where(and(...conditions));
    }

    // Count total results
    const countQuery = db
      .select({ count: sql<number>`count(*)` })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id));
    
    if (conditions.length > 0) {
      countQuery.where(and(...conditions));
    }

    const [countResult, searchResults] = await Promise.all([
      countQuery.execute(),
      executeSearchQuery(baseQuery, sortBy, sortOrder, page, pageSize)
    ]);

    const totalCount = countResult[0]?.count || 0;

    // Process results and calculate relevance scores
    const processedResults = await Promise.all(
      searchResults.map(item => processSearchResult(item, filters))
    );

    // Sort by relevance if we have a search query
    if (filters.query) {
      processedResults.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
    }

    // Generate facets
    const facets = await generateFacets(filters);

    // Generate search suggestions
    const suggestions = await generateSuggestions(filters.query);

    const searchTime = Date.now() - startTime;

    const response: SearchResponse = {
      results: processedResults,
      totalCount,
      searchTime,
      facets,
      suggestions,
      pagination: {
        page,
        pageSize,
        totalPages: Math.ceil(totalCount / pageSize)
      }
    };

    return json(response, {
      headers: {
        'X-Search-Time': `${searchTime}ms`,
        'X-Total-Results': totalCount.toString(),
        'Cache-Control': 'public, max-age=120' // Cache for 2 minutes
      }
    });

  } catch (err) {
    console.error('Search error:', err);
    throw error(500, `Search failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

async function buildSearchConditions(filters: SearchFilters) {
  const conditions = [];

  // Text search across multiple fields
  if (filters.query) {
    const searchTerm = `%${filters.query}%`;
    conditions.push(
      or(
        ilike(evidence.title, searchTerm),
        ilike(evidence.description, searchTerm),
        ilike(evidence.fileName, searchTerm),
        ilike(evidence.originalFileName, searchTerm),
        filters.contentSearch ? ilike(evidence.ocrText, searchTerm) : sql`FALSE`,
        filters.contentSearch ? ilike(evidence.contentText, searchTerm) : sql`FALSE`,
        ilike(cases.title, searchTerm)
      )
    );
  }

  // File type filters
  if (filters.fileTypes && filters.fileTypes.length > 0) {
    conditions.push(
      or(...filters.fileTypes.map(type => ilike(evidence.fileType, `%${type}%`)))
    );
  }

  // Case filters
  if (filters.caseIds && filters.caseIds.length > 0) {
    conditions.push(inArray(evidence.caseId, filters.caseIds));
  }

  // Date range filters
  if (filters.dateFrom) {
    conditions.push(gte(evidence.uploadedAt, new Date(filters.dateFrom)));
  }

  if (filters.dateTo) {
    conditions.push(lte(evidence.uploadedAt, new Date(filters.dateTo)));
  }

  // File size filters
  if (filters.fileSizeMin) {
    conditions.push(gte(evidence.fileSize, filters.fileSizeMin));
  }

  if (filters.fileSizeMax) {
    conditions.push(lte(evidence.fileSize, filters.fileSizeMax));
  }

  // Processing status filters
  if (filters.hasOCR !== undefined) {
    if (filters.hasOCR) {
      conditions.push(sql`${evidence.ocrText} IS NOT NULL AND ${evidence.ocrText} != ''`);
    } else {
      conditions.push(sql`${evidence.ocrText} IS NULL OR ${evidence.ocrText} = ''`);
    }
  }

  if (filters.hasEmbedding !== undefined) {
    if (filters.hasEmbedding) {
      conditions.push(sql`${evidence.embedding} IS NOT NULL`);
    } else {
      conditions.push(sql`${evidence.embedding} IS NULL`);
    }
  }

  if (filters.isProcessed !== undefined) {
    if (filters.isProcessed) {
      conditions.push(sql`${evidence.processedAt} IS NOT NULL`);
    } else {
      conditions.push(sql`${evidence.processedAt} IS NULL`);
    }
  }

  // Public/private filter
  if (filters.isPublic !== undefined) {
    conditions.push(eq(evidence.isPublic, filters.isPublic));
  }

  // Tag filters
  if (filters.tags && filters.tags.length > 0) {
    conditions.push(
      or(...filters.tags.map(tag => 
        sql`${evidence.tags} @> ${JSON.stringify([tag])}`
      ))
    );
  }

  return conditions;
}

async function executeSearchQuery(query: any, sortBy: string, sortOrder: string, page: number, pageSize: number) {
  // Apply sorting
  const orderColumn = getOrderColumn(sortBy);
  if (sortOrder === 'desc') {
    query.orderBy(desc(orderColumn));
  } else {
    query.orderBy(asc(orderColumn));
  }

  // Apply pagination
  const offset = (page - 1) * pageSize;
  query.limit(pageSize).offset(offset);

  return await query.execute();
}

function getOrderColumn(sortBy: string) {
  switch (sortBy) {
    case 'title':
      return evidence.title;
    case 'fileSize':
      return evidence.fileSize;
    case 'fileType':
      return evidence.fileType;
    case 'processedAt':
      return evidence.processedAt;
    case 'caseTitle':
      return cases.title;
    default:
      return evidence.uploadedAt;
  }
}

async function processSearchResult(item: any, filters: SearchFilters): Promise<SearchResult> {
  // Calculate relevance score
  const relevanceScore = calculateRelevanceScore(item, filters);
  
  // Extract matched fields
  const matchedFields = getMatchedFields(item, filters);
  
  // Generate snippet if content search is enabled
  const snippet = filters.contentSearch ? generateSnippet(item, filters.query) : undefined;

  return {
    id: item.id,
    type: determineItemType(item.fileType),
    title: item.title || item.fileName || 'Untitled',
    description: item.description || undefined,
    fileName: item.fileName || '',
    fileType: item.fileType || 'unknown',
    fileSize: item.fileSize || 0,
    url: `/api/files/evidence/${item.id}`,
    thumbnailUrl: generateThumbnailUrl(item.filePath, item.fileType),
    uploadedAt: item.uploadedAt?.toISOString() || new Date().toISOString(),
    caseId: item.caseId || undefined,
    caseTitle: item.caseTitle || undefined,
    tags: Array.isArray(item.tags) ? item.tags : [],
    metadata: item.metadata || undefined,
    relevanceScore,
    matchedFields,
    snippet
  };
}

function calculateRelevanceScore(item: any, filters: SearchFilters): number {
  if (!filters.query) return 0;

  let score = 0;
  const query = filters.query.toLowerCase();

  // Title match (highest weight)
  if (item.title?.toLowerCase().includes(query)) {
    score += 10;
    if (item.title?.toLowerCase().startsWith(query)) score += 5;
  }

  // Filename match
  if (item.fileName?.toLowerCase().includes(query)) {
    score += 7;
  }

  // Description match
  if (item.description?.toLowerCase().includes(query)) {
    score += 5;
  }

  // Case title match
  if (item.caseTitle?.toLowerCase().includes(query)) {
    score += 4;
  }

  // Content match
  if (filters.contentSearch) {
    if (item.ocrText?.toLowerCase().includes(query)) score += 3;
    if (item.contentText?.toLowerCase().includes(query)) score += 3;
  }

  // Tag match
  if (Array.isArray(item.tags)) {
    for (const tag of item.tags) {
      if (tag.toLowerCase().includes(query)) {
        score += 6;
      }
    }
  }

  return score;
}

function getMatchedFields(item: any, filters: SearchFilters): string[] {
  if (!filters.query) return [];

  const matchedFields = [];
  const query = filters.query.toLowerCase();

  if (item.title?.toLowerCase().includes(query)) matchedFields.push('title');
  if (item.fileName?.toLowerCase().includes(query)) matchedFields.push('fileName');
  if (item.description?.toLowerCase().includes(query)) matchedFields.push('description');
  if (item.caseTitle?.toLowerCase().includes(query)) matchedFields.push('caseTitle');
  
  if (filters.contentSearch) {
    if (item.ocrText?.toLowerCase().includes(query)) matchedFields.push('ocrText');
    if (item.contentText?.toLowerCase().includes(query)) matchedFields.push('contentText');
  }

  if (Array.isArray(item.tags)) {
    for (const tag of item.tags) {
      if (tag.toLowerCase().includes(query)) {
        matchedFields.push('tags');
        break;
      }
    }
  }

  return matchedFields;
}

function generateSnippet(item: any, query?: string): string | undefined {
  if (!query) return undefined;

  const text = item.contentText || item.ocrText || item.description || '';
  if (!text) return undefined;

  const index = text.toLowerCase().indexOf(query.toLowerCase());
  if (index === -1) return undefined;

  const start = Math.max(0, index - 50);
  const end = Math.min(text.length, index + query.length + 50);
  
  let snippet = text.slice(start, end);
  
  // Highlight the query term
  const regex = new RegExp(`(${query})`, 'gi');
  snippet = snippet.replace(regex, '<mark>$1</mark>');

  if (start > 0) snippet = '...' + snippet;
  if (end < text.length) snippet = snippet + '...';

  return snippet;
}

async function generateFacets(filters: SearchFilters) {
  try {
    // Get facet counts (simplified for now)
    const [typeFacets, fileTypeFacets, caseFacets, tagFacets] = await Promise.all([
      getTypeFacets(),
      getFileTypeFacets(),
      getCaseFacets(),
      getTagFacets()
    ]);

    return {
      types: typeFacets,
      fileTypes: fileTypeFacets,
      cases: caseFacets,
      tags: tagFacets,
      dateRanges: [
        { range: 'Last 24 hours', count: 0 },
        { range: 'Last week', count: 0 },
        { range: 'Last month', count: 0 },
        { range: 'Last year', count: 0 }
      ]
    };
  } catch (error) {
    console.error('Facet generation error:', error);
    return {
      types: [],
      fileTypes: [],
      cases: [],
      tags: [],
      dateRanges: []
    };
  }
}

async function getTypeFacets() {
  // TODO: Implement proper type facet counting
  return [
    { name: 'Evidence', count: 0 },
    { name: 'Document', count: 0 },
    { name: 'Image', count: 0 },
    { name: 'Video', count: 0 },
    { name: 'Audio', count: 0 }
  ];
}

async function getFileTypeFacets() {
  // TODO: Implement file type facet counting
  return [
    { name: 'PDF', count: 0 },
    { name: 'Image', count: 0 },
    { name: 'Word', count: 0 },
    { name: 'Excel', count: 0 },
    { name: 'Video', count: 0 }
  ];
}

async function getCaseFacets() {
  try {
    const caseCounts = await db
      .select({
        id: cases.id,
        title: cases.title,
        count: sql<number>`count(${evidence.id})`
      })
      .from(cases)
      .leftJoin(evidence, eq(cases.id, evidence.caseId))
      .groupBy(cases.id, cases.title)
      .execute();

    return caseCounts.map(c => ({
      id: c.id,
      title: c.title || 'Untitled',
      count: c.count || 0
    }));
  } catch (error) {
    return [];
  }
}

async function getTagFacets() {
  // TODO: Implement tag facet counting from JSONB array
  return [];
}

async function generateSuggestions(query?: string): Promise<string[]> {
  if (!query || query.length < 2) return [];

  // Simple suggestion generation (could be improved with proper search index)
  const suggestions = [
    'contract analysis',
    'evidence photos',
    'court documents',
    'legal briefs',
    'case files',
    'witness statements',
    'financial records',
    'email correspondence'
  ];

  return suggestions
    .filter(s => s.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 5);
}

function determineItemType(fileType?: string): string {
  if (!fileType) return 'document';
  
  if (fileType.startsWith('image/')) return 'image';
  if (fileType.startsWith('video/')) return 'video';
  if (fileType.startsWith('audio/')) return 'audio';
  if (fileType.includes('pdf')) return 'document';
  
  return 'document';
}

function generateThumbnailUrl(filePath: string | null, fileType: string | null): string | undefined {
  if (!filePath || !fileType) return undefined;
  
  if (fileType.startsWith('image/')) {
    const pathParts = filePath.split('/');
    const fileName = pathParts.pop();
    const dir = pathParts.join('/');
    return `${dir}/thumb_${fileName}`;
  }
  
  // Return type-specific icons for non-images
  const typeIconMap: Record<string, string> = {
    'application/pdf': '/icons/pdf-thumbnail.svg',
    'video/': '/icons/video-thumbnail.svg',
    'audio/': '/icons/audio-thumbnail.svg',
    'document': '/icons/document-thumbnail.svg'
  };

  for (const [type, icon] of Object.entries(typeIconMap)) {
    if (fileType.includes(type)) return icon;
  }
  
  return '/icons/file-thumbnail.svg';
}

// GET endpoint for simple search
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const query = url.searchParams.get('q') || '';
    const type = url.searchParams.get('type') || undefined;
    const caseId = url.searchParams.get('caseId') || undefined;
    const page = parseInt(url.searchParams.get('page') || '1');
    const pageSize = parseInt(url.searchParams.get('pageSize') || '20');

    const filters: SearchFilters = {
      query,
      types: type ? [type] : undefined,
      caseIds: caseId ? [caseId] : undefined,
      contentSearch: true
    };

    const options: SearchOptions = {
      page,
      pageSize,
      sortBy: 'uploadedAt',
      sortOrder: 'desc',
      includeContent: false
    };

    // Reuse POST logic
    const request = new Request('', {
      method: 'POST',
      body: JSON.stringify({ filters, options }),
      headers: { 'content-type': 'application/json' }
    });

    return await POST({ request, locals } as any);

  } catch (err) {
    console.error('GET search error:', err);
    throw error(500, `Search failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};