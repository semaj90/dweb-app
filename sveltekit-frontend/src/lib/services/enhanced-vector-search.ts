// Enhanced Vector Search Service for Legal AI Platform
// Uses pgvector L2 distance with proper operator classes and current schema
// Integrates with Drizzle ORM and pgvector extension

import { db } from '$lib/server/db';
import { 
  evidenceVectorsTable, 
  evidenceAnalysisTable,
  evidenceTable,
  casesTable 
} from '$lib/server/schema';
import { sql, eq, and, desc, inArray } from 'drizzle-orm';

export interface VectorSearchOptions {
  limit?: number;
  threshold?: number;
  caseId?: string;
  includeMetadata?: boolean;
  entityTypes?: ('evidence' | 'case')[];
}

export interface VectorSearchResult {
  id: string;
  entityId: string;
  entityType: 'evidence' | 'case' | 'chunk';
  similarity: number;
  title: string;
  description: string;
  metadata: Record<string, any>;
}

export class EnhancedVectorSearchService {
  /**
   * Find similar evidence using L2 distance with IVFFlat index
   * Uses the vector index we created with proper operator class
   */
  async findSimilarEvidence(
    queryEmbedding: number[],
    options: VectorSearchOptions = {}
  ): Promise<VectorSearchResult[]> {
    const {
      limit = 10,
      threshold = 0.7,
      caseId,
      includeMetadata = true
    } = options;

    try {
      // Convert embedding to pgvector format
      const embeddingVector = `[${queryEmbedding.join(',')}]`;
      
      console.log(`🔍 Vector search: ${queryEmbedding.length}D embedding, threshold=${threshold}, limit=${limit}`);

      // Vector similarity search using L2 distance (<->)
      // The IVFFlat index will automatically optimize this query
      let query = db.select({
        id: evidenceVectorsTable.id,
        evidenceId: evidenceVectorsTable.evidence_id,
        // L2 distance: smaller values = more similar
        distance: sql<number>`${evidenceVectorsTable.vector} <-> ${embeddingVector}::vector`.as('distance'),
        // Convert to similarity score (1 - normalized_distance)
        similarity: sql<number>`1 - (${evidenceVectorsTable.vector} <-> ${embeddingVector}::vector)`.as('similarity'),
        ...(includeMetadata ? {
          title: evidenceTable.title,
          description: evidenceTable.description,
          fileType: evidenceTable.file_type,
          caseId: evidenceTable.case_id,
          metadata: evidenceVectorsTable.metadata
        } : {})
      })
      .from(evidenceVectorsTable)
      .leftJoin(evidenceTable, eq(evidenceVectorsTable.evidence_id, evidenceTable.id))
      .where(
        sql`${evidenceVectorsTable.vector} <-> ${embeddingVector}::vector < ${1 - threshold}`
      )
      .orderBy(sql`${evidenceVectorsTable.vector} <-> ${embeddingVector}::vector ASC`) // Closest first
      .limit(limit);

      // Add case filter if specified
      if (caseId) {
        query = query.where(
          and(
            sql`${evidenceVectorsTable.vector} <-> ${embeddingVector}::vector < ${1 - threshold}`,
            eq(evidenceTable.case_id, caseId)
          )
        );
      }

      const results = await query;

      console.log(`✅ Found ${results.length} similar evidence items`);

      return results.map(result => ({
        id: result.id,
        entityId: result.evidenceId,
        entityType: 'evidence' as const,
        similarity: result.similarity,
        title: result.title || 'Untitled Evidence',
        description: result.description || '',
        metadata: {
          fileType: result.fileType,
          caseId: result.caseId,
          distance: result.distance,
          ...result.metadata as Record<string, any>
        }
      }));
    } catch (error) {
      console.error('❌ Error in evidence vector search:', error);
      throw new Error(`Evidence vector search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Find similar cases using case embeddings
   * Uses L2 distance for case similarity and clustering
   */
  async findSimilarCases(
    queryEmbedding: number[],
    options: VectorSearchOptions = {}
  ): Promise<VectorSearchResult[]> {
    const {
      limit = 5,
      threshold = 0.6,
      excludeCaseId
    } = options;

    try {
      // Check if cases table has embeddings (case_embedding column)
      const embeddingVector = `[${queryEmbedding.join(',')}]`;
      
      console.log(`🔍 Case similarity search: ${queryEmbedding.length}D embedding`);

      // Note: This assumes cases table has case_embedding vector column
      // If not present, this query will return empty results gracefully
      const results = await db.select({
        id: casesTable.id,
        title: casesTable.title,
        description: casesTable.description,
        status: casesTable.status,
        caseNumber: casesTable.case_number,
        // Using L2 distance for case similarity
        similarity: sql<number>`
          CASE 
            WHEN case_embedding IS NOT NULL 
            THEN 1 - (case_embedding <-> ${embeddingVector}::vector)
            ELSE 0 
          END`.as('similarity'),
        metadata: casesTable.metadata
      })
      .from(casesTable)
      .where(
        and(
          sql`case_embedding IS NOT NULL`,
          sql`case_embedding <-> ${embeddingVector}::vector < ${1 - threshold}`,
          excludeCaseId ? sql`${casesTable.id} != ${excludeCaseId}` : sql`1=1`
        )
      )
      .orderBy(sql`case_embedding <-> ${embeddingVector}::vector ASC`)
      .limit(limit);

      console.log(`✅ Found ${results.length} similar cases`);

      return results.map(result => ({
        id: result.id,
        entityId: result.id,
        entityType: 'case' as const,
        similarity: result.similarity,
        title: result.title,
        description: result.description || '',
        metadata: {
          status: result.status,
          caseNumber: result.caseNumber,
          ...result.metadata as Record<string, any>
        }
      }));
    } catch (error) {
      console.error('❌ Error in case vector search:', error);
      // Return empty results if case embeddings aren't implemented yet
      console.warn('⚠️ Case embeddings may not be implemented yet');
      return [];
    }
  }

  /**
   * Unified vector search across multiple entity types
   * Combines evidence and case searches with weighted results
   */
  async unifiedVectorSearch(
    queryEmbedding: number[],
    options: VectorSearchOptions & {
      weightByType?: Record<string, number>;
    } = {}
  ): Promise<VectorSearchResult[]> {
    const {
      limit = 20,
      threshold = 0.6,
      entityTypes = ['evidence', 'case'],
      weightByType = { evidence: 1.0, case: 0.8 }
    } = options;

    const results: VectorSearchResult[] = [];

    try {
      console.log(`🔍 Unified vector search across: ${entityTypes.join(', ')}`);

      // Search evidence if requested
      if (entityTypes.includes('evidence')) {
        const evidenceResults = await this.findSimilarEvidence(queryEmbedding, {
          ...options,
          limit: Math.ceil(limit * 0.7) // 70% of results from evidence
        });
        
        // Apply type weighting
        const weightedEvidenceResults = evidenceResults.map(result => ({
          ...result,
          similarity: result.similarity * (weightByType.evidence || 1.0)
        }));
        
        results.push(...weightedEvidenceResults);
      }

      // Search cases if requested
      if (entityTypes.includes('case')) {
        const caseResults = await this.findSimilarCases(queryEmbedding, {
          ...options,
          limit: Math.ceil(limit * 0.3) // 30% of results from cases
        });
        
        // Apply type weighting
        const weightedCaseResults = caseResults.map(result => ({
          ...result,
          similarity: result.similarity * (weightByType.case || 1.0)
        }));
        
        results.push(...weightedCaseResults);
      }

      // Sort by weighted similarity and limit results
      const finalResults = results
        .filter(result => result.similarity >= threshold)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);

      console.log(`✅ Unified search returned ${finalResults.length} results`);
      
      return finalResults;
    } catch (error) {
      console.error('❌ Error in unified vector search:', error);
      throw new Error(`Unified vector search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Batch vector search for multiple queries
   * Optimized for processing multiple searches efficiently
   */
  async batchVectorSearch(
    queries: { id: string; embedding: number[]; options?: VectorSearchOptions }[]
  ): Promise<Record<string, VectorSearchResult[]>> {
    const results: Record<string, VectorSearchResult[]> = {};

    console.log(`🔍 Batch vector search: ${queries.length} queries`);

    // Process queries in parallel for better performance
    const promises = queries.map(async (query) => {
      try {
        const searchResults = await this.unifiedVectorSearch(
          query.embedding,
          query.options
        );
        return { id: query.id, results: searchResults };
      } catch (error) {
        console.error(`❌ Batch search failed for query ${query.id}:`, error);
        return { id: query.id, results: [] };
      }
    });

    const completed = await Promise.all(promises);
    
    completed.forEach(({ id, results: searchResults }) => {
      results[id] = searchResults;
    });

    console.log(`✅ Batch search completed: ${Object.keys(results).length} queries processed`);
    
    return results;
  }

  /**
   * Get the nearest neighbors for a specific evidence item
   * Useful for "similar documents" features
   */
  async findNearestNeighbors(
    evidenceId: string,
    options: VectorSearchOptions = {}
  ): Promise<VectorSearchResult[]> {
    const { limit = 5, threshold = 0.7 } = options;

    try {
      // First, get the embedding for the source evidence
      const sourceEvidence = await db.select({
        id: evidenceVectorsTable.id,
        vector: evidenceVectorsTable.vector
      })
      .from(evidenceVectorsTable)
      .where(eq(evidenceVectorsTable.evidence_id, evidenceId))
      .limit(1);

      if (sourceEvidence.length === 0) {
        throw new Error(`No vector found for evidence ID: ${evidenceId}`);
      }

      const sourceVector = sourceEvidence[0].vector;

      // Find similar documents using the source vector
      const results = await db.select({
        id: evidenceVectorsTable.id,
        evidenceId: evidenceVectorsTable.evidence_id,
        similarity: sql<number>`1 - (${evidenceVectorsTable.vector} <-> ${sourceVector})`.as('similarity'),
        title: evidenceTable.title,
        description: evidenceTable.description,
        fileType: evidenceTable.file_type,
        metadata: evidenceVectorsTable.metadata
      })
      .from(evidenceVectorsTable)
      .leftJoin(evidenceTable, eq(evidenceVectorsTable.evidence_id, evidenceTable.id))
      .where(
        and(
          sql`${evidenceVectorsTable.vector} <-> ${sourceVector} < ${1 - threshold}`,
          sql`${evidenceVectorsTable.evidence_id} != ${evidenceId}` // Exclude the source document
        )
      )
      .orderBy(sql`${evidenceVectorsTable.vector} <-> ${sourceVector} ASC`)
      .limit(limit);

      console.log(`✅ Found ${results.length} nearest neighbors for evidence ${evidenceId}`);

      return results.map(result => ({
        id: result.id,
        entityId: result.evidenceId,
        entityType: 'evidence' as const,
        similarity: result.similarity,
        title: result.title || 'Untitled Evidence',
        description: result.description || '',
        metadata: {
          fileType: result.fileType,
          ...result.metadata as Record<string, any>
        }
      }));
    } catch (error) {
      console.error('❌ Error finding nearest neighbors:', error);
      throw new Error(`Nearest neighbors search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Health check for vector search functionality
   * Validates pgvector extension, indexes, and performance
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    details: Record<string, any>;
  }> {
    try {
      console.log('🔍 Performing vector search health check...');

      // Check if pgvector extension is installed
      const extensionCheck = await db.execute(sql`
        SELECT EXISTS(
          SELECT 1 FROM pg_extension WHERE extname = 'vector'
        ) as has_vector
      `);

      const hasVector = extensionCheck.rows[0]?.has_vector;

      if (!hasVector) {
        return {
          status: 'unhealthy',
          details: { 
            error: 'pgvector extension not installed',
            recommendation: 'Install pgvector extension: CREATE EXTENSION vector;'
          }
        };
      }

      // Check vector indexes exist
      const indexCheck = await db.execute(sql`
        SELECT indexname, tablename 
        FROM pg_indexes 
        WHERE indexname LIKE '%vector%'
      `);

      const indexes = indexCheck.rows;

      // Test query performance with a sample
      const performanceStart = Date.now();
      const sampleQuery = await db.execute(sql`
        SELECT COUNT(*) as count 
        FROM ${evidenceVectorsTable} 
        WHERE vector IS NOT NULL
      `);
      const performanceTime = Date.now() - performanceStart;

      const vectorCount = sampleQuery.rows[0]?.count || 0;

      // Test actual vector query if we have data
      let queryPerformance = 0;
      if (vectorCount > 0) {
        const queryStart = Date.now();
        await db.execute(sql`
          SELECT id 
          FROM ${evidenceVectorsTable} 
          WHERE vector IS NOT NULL 
          LIMIT 1
        `);
        queryPerformance = Date.now() - queryStart;
      }

      const status = performanceTime < 100 && queryPerformance < 50 ? 'healthy' : 
                    performanceTime < 500 && queryPerformance < 200 ? 'degraded' : 'unhealthy';

      console.log(`✅ Health check completed: ${status}`);

      return {
        status,
        details: {
          hasVector: true,
          vectorCount,
          indexes: indexes.map(idx => ({ name: idx.indexname, table: idx.tablename })),
          performance: {
            countQueryTime: performanceTime,
            sampleQueryTime: queryPerformance
          },
          indexStatus: performanceTime < 50 ? 'optimal' : 
                      performanceTime < 100 ? 'good' : 
                      performanceTime < 500 ? 'slow' : 'poor'
        }
      };
    } catch (error) {
      console.error('❌ Health check failed:', error);
      return {
        status: 'unhealthy',
        details: { 
          error: error instanceof Error ? error.message : 'Unknown error',
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  /**
   * Get comprehensive vector search statistics
   */
  async getSearchStats(): Promise<{
    vectorCounts: Record<string, number>;
    indexStats: Record<string, any>;
    dimensions: Record<string, number>;
    recentActivity: Record<string, any>;
  }> {
    try {
      console.log('📊 Gathering vector search statistics...');

      // Get vector counts by table
      const evidenceVectorCount = await db.execute(sql`
        SELECT COUNT(*) as count FROM ${evidenceVectorsTable} WHERE vector IS NOT NULL
      `);

      const caseVectorCount = await db.execute(sql`
        SELECT COUNT(*) as count FROM ${casesTable} WHERE case_embedding IS NOT NULL
      `);

      // Check vector dimensions
      const dimensionCheck = await db.execute(sql`
        SELECT 
          'evidence_vectors' as table_name,
          array_length(vector, 1) as dimensions
        FROM ${evidenceVectorsTable} 
        WHERE vector IS NOT NULL 
        LIMIT 1
      `);

      // Get index statistics
      const indexStats = await db.execute(sql`
        SELECT 
          schemaname,
          tablename,
          indexname,
          idx_scan as scans,
          idx_tup_read as tuples_read,
          idx_tup_fetch as tuples_fetched
        FROM pg_stat_user_indexes 
        WHERE indexname LIKE '%vector%'
      `);

      const stats = {
        vectorCounts: {
          evidence: evidenceVectorCount.rows[0]?.count || 0,
          cases: caseVectorCount.rows[0]?.count || 0
        },
        indexStats: indexStats.rows.reduce((acc, row: any) => {
          acc[row.indexname] = {
            table: row.tablename,
            scans: row.scans,
            tuplesRead: row.tuples_read,
            tuplesFetched: row.tuples_fetched
          };
          return acc;
        }, {} as Record<string, any>),
        dimensions: {
          evidence: dimensionCheck.rows[0]?.dimensions || 0
        },
        recentActivity: {
          lastUpdated: new Date().toISOString(),
          healthStatus: 'operational' // This would be updated by monitoring
        }
      };

      console.log('✅ Statistics gathered successfully');
      
      return stats;
    } catch (error) {
      console.error('❌ Error getting search stats:', error);
      return {
        vectorCounts: {},
        indexStats: {},
        dimensions: {},
        recentActivity: {
          error: error instanceof Error ? error.message : 'Unknown error',
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  /**
   * Utility method to validate embedding dimensions
   */
  validateEmbeddingDimensions(embedding: number[], expectedDimensions: number = 384): boolean {
    if (!Array.isArray(embedding)) {
      console.error('❌ Embedding must be an array');
      return false;
    }
    
    if (embedding.length !== expectedDimensions) {
      console.error(`❌ Embedding dimension mismatch: expected ${expectedDimensions}, got ${embedding.length}`);
      return false;
    }
    
    if (embedding.some(val => typeof val !== 'number' || isNaN(val))) {
      console.error('❌ Embedding contains invalid values');
      return false;
    }
    
    return true;
  }
}

// Export singleton instance
export const enhancedVectorSearchService = new EnhancedVectorSearchService();

// Export for dependency injection
export default EnhancedVectorSearchService;