/**
 * Qdrant Vector Database Integration with PostgreSQL Sync
 * Seamless vector operations between PostgreSQL pgvector and Qdrant
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import { drizzle } from 'drizzle-orm/node-postgres';
import postgres from 'postgres';
import { eq, and, sql } from 'drizzle-orm';
import crypto from "crypto";
import {
  legalDocuments,
  cases,
  vectorOperations,
  qdrantCollections,
  createQdrantPoint,
  type LegalDocument,
  type Case,
  type VectorOperation,
  type VectorMetadata
} from './enhanced-unified-schema';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface QdrantConfig {
  host: string;
  port: number;
  apiKey?: string;
  timeout?: number;
}

export interface PostgreSQLConfig {
  connectionString: string;
  max?: number;
  idle_timeout?: number;
}

// ============================================================================
// QDRANT-POSTGRESQL INTEGRATION SERVICE
// ============================================================================

export class QdrantPostgreSQLService {
  private qdrant: QdrantClient;
  private postgres: postgres.Sql;
  private db: ReturnType<typeof drizzle>;

  constructor(
    qdrantConfig: QdrantConfig,
    postgresConfig: PostgreSQLConfig
  ) {
    // Initialize Qdrant client
    this.qdrant = new QdrantClient({
      url: `http://${qdrantConfig.host}:${qdrantConfig.port}`,
      apiKey: qdrantConfig.apiKey,
    });

    // Initialize PostgreSQL connection
    this.postgres = postgres(postgresConfig.connectionString, {
      max: postgresConfig.max || 10,
      idle_timeout: postgresConfig.idle_timeout || 20,
      types: {
        vector: {
          to: 1184,
          from: [1184],
          serialize: (x: number[]) => {
            if (Array.isArray(x)) {
              return `[${x.join(',')}]`;
            }
            return x || '[]';
          },
          parse: (x: string) => {
            if (typeof x === 'string' && x.startsWith('[') && x.endsWith(']')) {
              return x.slice(1, -1).split(',').map(Number);
            }
            return [];
          },
        },
      },
    });

    this.db = drizzle(this.postgres, {
      schema: {
        legalDocuments,
        cases,
        vectorOperations,
        qdrantCollections,
      },
    });
  }

  // ============================================================================
  // COLLECTION MANAGEMENT
  // ============================================================================

  async ensureCollection(
    collectionName: string,
    vectorSize: number = 384,
    distance: 'Cosine' | 'Dot' | 'Euclidean' = 'Cosine'
  ): Promise<void> {
    try {
      // Check if collection exists in Qdrant
      const collections = await this.qdrant.getCollections();
      const exists = collections.collections.some(c => c.name === collectionName);

      if (!exists) {
        // Create collection in Qdrant
        await this.qdrant.createCollection(collectionName, {
          vectors: {
            size: vectorSize,
            distance,
          },
          optimizers_config: {
            default_segment_number: 2,
            memmap_threshold: 20000,
            indexing_threshold: 20000,
          },
          hnsw_config: {
            m: 16,
            ef_construct: 64,
            full_scan_threshold: 10000,
          },
        });

        console.log(`✅ Created Qdrant collection: ${collectionName}`);
      }

      // Ensure collection record in PostgreSQL
      await this.db
        .insert(qdrantCollections)
        .values({
          name: collectionName,
          vectorSize,
          distance,
          status: 'active',
        })
        .onConflictDoUpdate({
          target: qdrantCollections.name,
          set: {
            vectorSize,
            distance,
            updatedAt: new Date(),
          },
        });

    } catch (error: any) {
      console.error(`❌ Failed to ensure collection ${collectionName}:`, error);
      throw error;
    }
  }

  // ============================================================================
  // DOCUMENT VECTOR OPERATIONS
  // ============================================================================

  async syncDocumentToQdrant(documentId: string): Promise<boolean> {
    const operationId = crypto.randomUUID();

    try {
      // Create operation record
      await this.db.insert(vectorOperations).values({
        id: operationId,
        operationType: 'sync',
        entityType: 'document',
        entityId: documentId,
        status: 'processing',
      });

      // Get document with embeddings
      const document = await this.db
        .select()
        .from(legalDocuments)
        .where(eq(legalDocuments.id, documentId))
        .limit(1);

      if (!document.length) {
        throw new Error(`Document ${documentId} not found`);
      }

      const doc = document[0];
      if (!doc.contentEmbedding) {
        throw new Error(`Document ${documentId} has no content embedding`);
      }

      // Ensure collection exists
      await this.ensureCollection(doc.qdrantCollection || 'legal_documents');

      // Create Qdrant point
      const point = createQdrantPoint(documentId, doc.contentEmbedding, {
        title: doc.title,
        document_type: doc.documentType,
        practice_area: doc.practiceArea,
        case_id: doc.caseId,
        user_id: doc.userId,
        created_at: doc.createdAt?.toISOString(),
        metadata: doc.metadata,
      });

      // Upsert to Qdrant
      await this.qdrant.upsert(doc.qdrantCollection || 'legal_documents', {
        points: [point],
      });

      // Update document with Qdrant sync info
      await this.db
        .update(legalDocuments)
        .set({
          qdrantId: documentId,
          lastSyncedToQdrant: new Date(),
          updatedAt: new Date(),
        })
        .where(eq(legalDocuments.id, documentId));

      // Update operation as completed
      await this.db
        .update(vectorOperations)
        .set({
          status: 'completed',
          qdrantSynced: true,
          qdrantSyncedAt: new Date(),
          completedAt: new Date(),
        })
        .where(eq(vectorOperations.id, operationId));

      console.log(`✅ Synced document ${documentId} to Qdrant`);
      return true;

    } catch (error: any) {
      console.error(`❌ Failed to sync document ${documentId}:`, error);

      // Update operation as failed
      await this.db
        .update(vectorOperations)
        .set({
          status: 'failed',
          error: error.message,
          completedAt: new Date(),
        })
        .where(eq(vectorOperations.id, operationId));

      return false;
    }
  }

  // ============================================================================
  // VECTOR SEARCH OPERATIONS
  // ============================================================================

  async hybridSearch(
    query: string,
    queryEmbedding: number[],
    options: {
      collection?: string;
      limit?: number;
      threshold?: number;
      filter?: Record<string, any>;
      usePostgreSQL?: boolean;
      useQdrant?: boolean;
    } = {}
  ): Promise<{
    results: Array<{
      id: string;
      score: number;
      document: LegalDocument;
      source: 'postgresql' | 'qdrant';
    }>;
    performance: {
      postgresqlTime?: number;
      qdrantTime?: number;
      totalTime: number;
    };
  }> {
    const startTime = Date.now();
    const {
      collection = 'legal_documents',
      limit = 10,
      threshold = 0.7,
      filter = {},
      usePostgreSQL = true,
      useQdrant = true,
    } = options;

    const results: Array<{
      id: string;
      score: number;
      document: LegalDocument;
      source: 'postgresql' | 'qdrant';
    }> = [];

    let postgresqlTime: number | undefined;
    let qdrantTime: number | undefined;

    // PostgreSQL search
    if (usePostgreSQL) {
      const pgStart = Date.now();
      
      try {
        const pgResults = await this.postgres`
          SELECT *, 
                 (1 - (content_embedding <=> ${JSON.stringify(queryEmbedding)}::vector)) as similarity
          FROM legal_documents
          WHERE (1 - (content_embedding <=> ${JSON.stringify(queryEmbedding)}::vector)) >= ${threshold}
            AND deleted_at IS NULL
            AND status = 'active'
          ORDER BY content_embedding <=> ${JSON.stringify(queryEmbedding)}::vector
          LIMIT ${limit}
        `;

        postgresqlTime = Date.now() - pgStart;

        for (const row of pgResults) {
          results.push({
            id: row.id,
            score: row.similarity,
            document: row as LegalDocument,
            source: 'postgresql',
          });
        }

      } catch (error: any) {
        console.error('PostgreSQL search error:', error);
      }
    }

    // Qdrant search
    if (useQdrant) {
      const qdrantStart = Date.now();
      
      try {
        const qdrantResults = await this.qdrant.search(collection, {
          vector: queryEmbedding,
          limit,
          score_threshold: threshold,
          with_payload: true,
          filter: Object.keys(filter).length > 0 ? { must: Object.entries(filter).map(([key, value]) => ({
            key,
            match: { value },
          })) } : undefined,
        });

        qdrantTime = Date.now() - qdrantStart;

        // Get corresponding PostgreSQL records
        const qdrantIds = qdrantResults.map(r => r.id.toString());
        
        if (qdrantIds.length > 0) {
          const pgDocuments = await this.db
            .select()
            .from(legalDocuments)
            .where(sql`${legalDocuments.id} = ANY(${qdrantIds})`);

          const docMap = new Map(pgDocuments.map(doc => [doc.id, doc]));

          for (const result of qdrantResults) {
            const document = docMap.get(result.id.toString());
            if (document) {
              results.push({
                id: result.id.toString(),
                score: result.score,
                document,
                source: 'qdrant',
              });
            }
          }
        }

      } catch (error: any) {
        console.error('Qdrant search error:', error);
      }
    }

    // Deduplicate and sort results
    const uniqueResults = new Map();
    for (const result of results) {
      const existing = uniqueResults.get(result.id);
      if (!existing || result.score > existing.score) {
        uniqueResults.set(result.id, result);
      }
    }

    const finalResults = Array.from(uniqueResults.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return {
      results: finalResults,
      performance: {
        postgresqlTime,
        qdrantTime,
        totalTime: Date.now() - startTime,
      },
    };
  }

  // ============================================================================
  // BATCH OPERATIONS
  // ============================================================================

  async batchSyncToQdrant(
    entityType: 'document' | 'case',
    batchSize: number = 100
  ): Promise<{
    synced: number;
    failed: number;
    errors: string[];
  }> {
    const results = { synced: 0, failed: 0, errors: [] as string[] };

    try {
      let offset = 0;
      let hasMore = true;

      while (hasMore) {
        // Get batch of documents that need syncing
        const batch = await this.db
          .select()
          .from(legalDocuments)
          .where(
            and(
              sql`${legalDocuments.contentEmbedding} IS NOT NULL`,
              sql`${legalDocuments.lastSyncedToQdrant} IS NULL OR ${legalDocuments.updatedAt} > ${legalDocuments.lastSyncedToQdrant}`,
              sql`${legalDocuments.deletedAt} IS NULL`
            )
          )
          .limit(batchSize)
          .offset(offset);

        if (batch.length === 0) {
          hasMore = false;
          break;
        }

        // Process batch
        for (const document of batch) {
          const success = await this.syncDocumentToQdrant(document.id);
          if (success) {
            results.synced++;
          } else {
            results.failed++;
            results.errors.push(`Failed to sync document ${document.id}`);
          }
        }

        offset += batchSize;
        
        // Add small delay to prevent overwhelming the services
        await new Promise(resolve => setTimeout(resolve, 100));
      }

    } catch (error: any) {
      results.errors.push(`Batch sync error: ${error.message}`);
    }

    return results;
  }

  // ============================================================================
  // HEALTH CHECK AND MONITORING
  // ============================================================================

  async healthCheck(): Promise<{
    postgresql: boolean;
    qdrant: boolean;
    collections: string[];
    syncStatus: {
      totalDocuments: number;
      syncedDocuments: number;
      pendingSyncs: number;
    };
  }> {
    let postgresql = false;
    let qdrant = false;
    let collections: string[] = [];
    
    // Check PostgreSQL
    try {
      await this.postgres`SELECT 1`;
      postgresql = true;
    } catch (error: any) {
      console.error('PostgreSQL health check failed:', error);
    }

    // Check Qdrant
    try {
      const collectionsResponse = await this.qdrant.getCollections();
      qdrant = true;
      collections = collectionsResponse.collections.map(c => c.name);
    } catch (error: any) {
      console.error('Qdrant health check failed:', error);
    }

    // Get sync status
    const syncStatus = { totalDocuments: 0, syncedDocuments: 0, pendingSyncs: 0 };
    
    try {
      const totalResult = await this.postgres`
        SELECT COUNT(*) as count 
        FROM legal_documents 
        WHERE deleted_at IS NULL AND content_embedding IS NOT NULL
      `;
      syncStatus.totalDocuments = parseInt(totalResult[0].count);

      const syncedResult = await this.postgres`
        SELECT COUNT(*) as count 
        FROM legal_documents 
        WHERE deleted_at IS NULL 
          AND content_embedding IS NOT NULL 
          AND last_synced_to_qdrant IS NOT NULL
      `;
      syncStatus.syncedDocuments = parseInt(syncedResult[0].count);

      syncStatus.pendingSyncs = syncStatus.totalDocuments - syncStatus.syncedDocuments;

    } catch (error: any) {
      console.error('Sync status check failed:', error);
    }

    return {
      postgresql,
      qdrant,
      collections,
      syncStatus,
    };
  }

  // ============================================================================
  // CLEANUP
  // ============================================================================

  async close(): Promise<void> {
    await this.postgres.end();
  }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export const createQdrantService = (;
  qdrantConfig?: Partial<QdrantConfig>,
  postgresConfig?: Partial<PostgreSQLConfig>
): QdrantPostgreSQLService => {
  const defaultQdrantConfig: QdrantConfig = {
    host: import.meta.env.QDRANT_HOST || 'localhost',
    port: parseInt(import.meta.env.QDRANT_PORT || '6333'),
    apiKey: import.meta.env.QDRANT_API_KEY,
    ...qdrantConfig,
  };

  const defaultPostgresConfig: PostgreSQLConfig = {
    connectionString: import.meta.env.DATABASE_URL || 
      `postgresql://${import.meta.env.DATABASE_USER || 'legal_admin'}:${import.meta.env.DATABASE_PASSWORD || '123456'}@${import.meta.env.DATABASE_HOST || 'localhost'}:${import.meta.env.DATABASE_PORT || '5432'}/${import.meta.env.DATABASE_NAME || 'legal_ai_db'}`,
    ...postgresConfig,
  };

  return new QdrantPostgreSQLService(defaultQdrantConfig, defaultPostgresConfig);
};

export default QdrantPostgreSQLService;