import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from "$lib/database/postgres-enhanced.js";
import { legalDocuments } from "$lib/database/schema/legal-documents.js";
import { vectorSearchService, embeddingUtils } from "$lib/database/vector-operations.js";
import { sql, desc, asc, and, or, eq, ilike, inArray, count } from "drizzle-orm";
import { z } from 'zod';

// Search parameters schema
const searchParamsSchema = z.object({
  query: z.string().min(1).max(1000),
  searchType: z.enum(['semantic', 'text', 'hybrid']).default('hybrid'),
  documentTypes: z.array(z.enum(['contract', 'motion', 'evidence', 'correspondence', 'brief', 'regulation', 'case_law'])).optional(),
  jurisdictions: z.array(z.string()).optional(),
  practiceAreas: z.array(z.enum(['corporate', 'litigation', 'intellectual_property', 'employment', 'real_estate', 'criminal', 'family', 'tax', 'immigration', 'environmental'])).optional(),
  isConfidential: z.boolean().optional(),
  dateRange: z.object({
    start: z.string().pipe(z.coerce.date()),
    end: z.string().pipe(z.coerce.date())
  }).optional(),
  limit: z.number().min(1).max(100).default(20),
  offset: z.number().min(0).default(0),
  threshold: z.number().min(0).max(1).default(0.7),
  includeContent: z.boolean().default(false),
  includeAnalysis: z.boolean().default(false),
  sortBy: z.enum(['relevance', 'date', 'title', 'size']).default('relevance'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
});

/**
 * Document Search API Endpoint
 * Supports semantic search, text search, and hybrid search with advanced filtering
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const searchParams = searchParamsSchema.parse(body);

    const {
      query,
      searchType,
      documentTypes,
      jurisdictions,
      practiceAreas,
      isConfidential,
      dateRange,
      limit,
      offset,
      threshold,
      includeContent,
      includeAnalysis,
      sortBy,
      sortOrder
    } = searchParams;

    let results: unknown[] = [];
    let totalCount = 0;

    if (searchType === 'semantic' || searchType === 'hybrid') {
      // Generate embedding for the search query
      const queryEmbedding = await generateSearchEmbedding(query);

      // Prepare filter options
      const filterOptions = {
        limit: limit + offset, // Get more for pagination
        threshold,
        filter: {
          documentType: documentTypes,
          jurisdiction: jurisdictions,
          practiceArea: practiceAreas,
          isConfidential,
          dateRange
        }
      };

      if (searchType === 'semantic') {
        // Pure semantic search
        const semanticResults = await vectorSearchService.searchDocuments(queryEmbedding, filterOptions);
        results = semanticResults.slice(offset, offset + limit);
        totalCount = semanticResults.length;
      } else {
        // Hybrid search (semantic + text)
        const titleEmbedding = await generateSearchEmbedding(query.split(' ').slice(0, 10).join(' ')); // Shorter query for title
        const hybridResults = await vectorSearchService.hybridSearch(
          queryEmbedding,
          titleEmbedding,
          { ...filterOptions, contentWeight: 0.7, titleWeight: 0.3 }
        );
        results = hybridResults.slice(offset, offset + limit);
        totalCount = hybridResults.length;
      }

      // Convert to standard format
      results = results.map(result => ({
        ...result.item,
        similarity: result.similarity,
        rank: result.rank
      }));

    } else {
      // Text-based search using PostgreSQL full-text search
      const textResults = await performTextSearch(searchParams);
      results = textResults.documents;
      totalCount = textResults.total;
    }

    // Apply sorting if not using relevance (semantic results are already sorted by relevance)
    if (sortBy !== 'relevance') {
      results = applySorting(results, sortBy, sortOrder);
    }

    // Format response
    const formattedResults = results.map(doc => ({
      id: doc.id,
      title: doc.title,
      documentType: doc.documentType,
      jurisdiction: doc.jurisdiction,
      practiceArea: doc.practiceArea,
      fileName: doc.fileName,
      fileSize: doc.fileSize,
      mimeType: doc.mimeType,
      processingStatus: doc.processingStatus,
      isConfidential: doc.isConfidential,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
      similarity: doc.similarity || null,
      rank: doc.rank || null,
      // Conditionally include content and analysis
      content: includeContent ? doc.content : null,
      analysisResults: includeAnalysis ? doc.analysisResults : null,
      // Add snippet for text search
      snippet: searchType !== 'semantic' ? generateSnippet(doc.content, query) : null,
    }));

    return json({
      success: true,
      results: formattedResults,
      pagination: {
        total: totalCount,
        limit,
        offset,
        hasMore: offset + limit < totalCount,
        page: Math.floor(offset / limit) + 1,
        totalPages: Math.ceil(totalCount / limit)
      },
      searchMetadata: {
        query,
        searchType,
        threshold: searchType === 'text' ? null : threshold,
        filters: {
          documentTypes,
          jurisdictions,
          practiceAreas,
          isConfidential,
          dateRange
        },
        executionTime: Date.now() - Date.now(), // Would track actual execution time
      }
    });

  } catch (error: unknown) {
    console.error("Search error:", error);

    if (error instanceof z.ZodError) {
      return json({
        success: false,
        error: "Invalid search parameters",
        details: error.errors,
      }, { status: 400 });
    }

    return json({
      success: false,
      error: error?.message || "Search failed",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Get search suggestions and autocomplete
 */
export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get("q");
    const type = url.searchParams.get("type") || "suggestions";

    if (!query || query.length < 2) {
      return json({
        success: true,
        suggestions: [],
        message: "Query too short for suggestions"
      });
    }

    if (type === "suggestions") {
      // Get title suggestions based on partial text match
      const suggestions = await db
        .select({
          id: legalDocuments.id,
          title: legalDocuments.title,
          documentType: legalDocuments.documentType
        })
        .from(legalDocuments)
        .where(
          and(
            ilike(legalDocuments.title, `%${query}%`),
            eq(legalDocuments.processingStatus, 'completed')
          )
        )
        .limit(10)
        .orderBy(desc(legalDocuments.updatedAt));

      return json({
        success: true,
        suggestions: suggestions.map(s => ({
          id: s.id,
          title: s.title,
          type: s.documentType
        }))
      });

    } else if (type === "filters") {
      // Get available filter values
      const filters = await getFilterOptions();
      
      return json({
        success: true,
        filters
      });

    } else {
      return json({
        success: false,
        error: "Invalid suggestion type"
      }, { status: 400 });
    }

  } catch (error: unknown) {
    console.error("Search suggestions error:", error);

    return json({
      success: false,
      error: "Failed to get search suggestions",
    }, { status: 500 });
  }
};

/**
 * Perform text-based search using PostgreSQL full-text search
 */
async function performTextSearch(params: z.infer<typeof searchParamsSchema>) {
  const {
    query,
    documentTypes,
    jurisdictions,
    practiceAreas,
    isConfidential,
    dateRange,
    limit,
    offset,
    includeContent
  } = params;

  // Build filter conditions
  const filterConditions = [];
  
  // Add text search condition
  filterConditions.push(
    or(
      sql`to_tsvector('english', ${legalDocuments.content}) @@ plainto_tsquery('english', ${query})`,
      sql`to_tsvector('english', ${legalDocuments.title}) @@ plainto_tsquery('english', ${query})`
    )
  );

  // Add other filters
  if (documentTypes && documentTypes.length > 0) {
    filterConditions.push(inArray(legalDocuments.documentType, documentTypes));
  }
  
  if (jurisdictions && jurisdictions.length > 0) {
    filterConditions.push(inArray(legalDocuments.jurisdiction, jurisdictions));
  }
  
  if (practiceAreas && practiceAreas.length > 0) {
    filterConditions.push(inArray(legalDocuments.practiceArea, practiceAreas));
  }
  
  if (isConfidential !== undefined) {
    filterConditions.push(eq(legalDocuments.isConfidential, isConfidential));
  }
  
  if (dateRange) {
    filterConditions.push(
      and(
        sql`${legalDocuments.createdAt} >= ${dateRange.start}`,
        sql`${legalDocuments.createdAt} <= ${dateRange.end}`
      )
    );
  }

  // Get total count
  const [countResult] = await db
    .select({ count: count() })
    .from(legalDocuments)
    .where(and(...filterConditions));

  // Get documents
  const documents = await db
    .select({
      id: legalDocuments.id,
      title: legalDocuments.title,
      content: includeContent ? legalDocuments.content : sql`''`.as('content'),
      documentType: legalDocuments.documentType,
      jurisdiction: legalDocuments.jurisdiction,
      practiceArea: legalDocuments.practiceArea,
      fileName: legalDocuments.fileName,
      fileSize: legalDocuments.fileSize,
      mimeType: legalDocuments.mimeType,
      processingStatus: legalDocuments.processingStatus,
      isConfidential: legalDocuments.isConfidential,
      analysisResults: legalDocuments.analysisResults,
      createdAt: legalDocuments.createdAt,
      updatedAt: legalDocuments.updatedAt,
      // Calculate text search rank
      rank: sql`ts_rank(to_tsvector('english', ${legalDocuments.content}), plainto_tsquery('english', ${query}))`.as('rank')
    })
    .from(legalDocuments)
    .where(and(...filterConditions))
    .orderBy(desc(sql`ts_rank(to_tsvector('english', ${legalDocuments.content}), plainto_tsquery('english', ${query}))`))
    .limit(limit)
    .offset(offset);

  return {
    documents,
    total: countResult.count
  };
}

/**
 * Apply sorting to search results
 */
function applySorting(results: unknown[], sortBy: string, sortOrder: string) {
  const sortFn = (a: unknown, b: unknown) => {
    let valueA, valueB;

    switch (sortBy) {
      case 'date':
        valueA = new Date(a.createdAt).getTime();
        valueB = new Date(b.createdAt).getTime();
        break;
      case 'title':
        valueA = a.title.toLowerCase();
        valueB = b.title.toLowerCase();
        break;
      case 'size':
        valueA = a.fileSize || 0;
        valueB = b.fileSize || 0;
        break;
      default:
        return 0;
    }

    if (valueA < valueB) return sortOrder === 'asc' ? -1 : 1;
    if (valueA > valueB) return sortOrder === 'asc' ? 1 : -1;
    return 0;
  };

  return [...results].sort(sortFn);
}

/**
 * Generate a snippet from content around the search query
 */
function generateSnippet(content: string, query: string, maxLength: number = 200): string {
  if (!content) return '';

  const queryWords = query.toLowerCase().split(/\s+/);
  const contentLower = content.toLowerCase();
  
  // Find the first occurrence of any query word
  let earliestIndex = content.length;
  for (const word of queryWords) {
    const index = contentLower.indexOf(word);
    if (index !== -1 && index < earliestIndex) {
      earliestIndex = index;
    }
  }

  if (earliestIndex === content.length) {
    // No query words found, return beginning of content
    return content.substring(0, maxLength) + (content.length > maxLength ? '...' : '');
  }

  // Extract snippet around the found word
  const start = Math.max(0, earliestIndex - maxLength / 2);
  const end = Math.min(content.length, start + maxLength);
  
  let snippet = content.substring(start, end);
  
  if (start > 0) snippet = '...' + snippet;
  if (end < content.length) snippet = snippet + '...';
  
  return snippet;
}

/**
 * Get available filter options for the search interface
 */
async function getFilterOptions() {
  const [documentTypes, jurisdictions, practiceAreas] = await Promise.all([
    // Get distinct document types
    db
      .selectDistinct({ value: legalDocuments.documentType })
      .from(legalDocuments)
      .where(eq(legalDocuments.processingStatus, 'completed')),
    
    // Get distinct jurisdictions
    db
      .selectDistinct({ value: legalDocuments.jurisdiction })
      .from(legalDocuments)
      .where(eq(legalDocuments.processingStatus, 'completed')),
    
    // Get distinct practice areas
    db
      .selectDistinct({ value: legalDocuments.practiceArea })
      .from(legalDocuments)
      .where(
        and(
          eq(legalDocuments.processingStatus, 'completed'),
          sql`${legalDocuments.practiceArea} IS NOT NULL`
        )
      )
  ]);

  return {
    documentTypes: documentTypes.map(d => d.value).filter(Boolean),
    jurisdictions: jurisdictions.map(j => j.value).filter(Boolean),
    practiceAreas: practiceAreas.map(p => p.value).filter(Boolean)
  };
}

/**
 * Generate search embedding using your embedding service
 */
async function generateSearchEmbedding(text: string): Promise<number[]> {
  // This would integrate with your embedding service (Ollama, OpenAI, etc.)
  // For now, return a placeholder 384-dimensional vector
  // In production, this would call your actual embedding service
  return Array(384).fill(0).map(() => Math.random() - 0.5);
}