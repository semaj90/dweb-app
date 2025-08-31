
// Enhanced Database Module Index
// Provides centralized access to all database services with proper TypeScript support

// Enhanced PostgreSQL with Drizzle ORM
export { 
  db, 
  dbManager, 
  dbUtils, 
  queryClient,
  schema,
  legalDocuments,
  contentEmbeddings,
  searchSessions,
  embeddings
} from './postgres-enhanced';

// Enhanced Qdrant Vector Database
export { 
  qdrantManager,
  EnhancedQdrantManager
} from './qdrant-enhanced';

// Enhanced Legal AI Orchestrator
export { 
  legalOrchestrator,
  EnhancedLegalOrchestrator
} from '../agents/orchestrator-enhanced';

// Database Schema Types (consolidated export)
export type {
  LegalDocument,
  ContentEmbedding,
  NewContentEmbedding,
  SearchSession,
  NewSearchSession,
  EmbeddingRecord,
  NewEmbeddingRecord
} from './schema/legal-documents';

// Import NewLegalDocument from the correct schema  
import type {
  NewLegalDocument
} from '../server/db/schema-postgres';

// Re-export for external use
export type { NewLegalDocument };

// Qdrant Types
export type {
  QdrantPoint,
  QdrantSearchResult,
  DocumentUpsertRequest
} from './qdrant-enhanced';

// Orchestration Types
export type {
  OrchestrationRequest,
  OrchestrationResponse
} from '../agents/orchestrator-enhanced';

// Database initialization helper
export async function initializeDatabase(): Promise<{
  postgres: boolean;
  qdrant: boolean;
  errors: string[];
}> {
  const results = {
    postgres: false,
    qdrant: false,
    errors: [] as string[]
  };

  try {
    // Initialize PostgreSQL
    const { dbUtils } = await import('./postgres-enhanced.js');
    results.postgres = await dbUtils.initialize();
    if (!results.postgres) {
      results.errors.push('PostgreSQL initialization failed');
    }
  } catch (error: any) {
    results.errors.push(`PostgreSQL error: ${error.message}`);
  }

  try {
    // Initialize Qdrant
    const { qdrantManager } = await import('./qdrant-enhanced.js');
    results.qdrant = await qdrantManager.connect();
    if (!results.qdrant) {
      results.errors.push('Qdrant connection failed');
    }
  } catch (error: any) {
    results.errors.push(`Qdrant error: ${error.message}`);
  }

  return results;
}

// Health check for all database services
export async function getDatabaseHealth(): Promise<{
  postgres: {
    connected: boolean;
    responseTime?: number;
    error?: string;
  };
  qdrant: {
    connected: boolean;
    collection: string;
    vectorCount?: number;
    error?: string;
  };
  overall: 'healthy' | 'degraded' | 'unhealthy';
}> {
  const health = {
    postgres: { connected: false } as {
      connected: boolean;
      responseTime?: number;
      error?: string;
    },
    qdrant: { connected: false, collection: '' } as {
      connected: boolean;
      collection: string;
      vectorCount?: number;
      error?: string;
    },
    overall: 'unhealthy' as 'healthy' | 'degraded' | 'unhealthy'
  };

  try {
    const { dbManager } = await import('./postgres-enhanced.js');
    health.postgres = await dbManager.healthCheck();
  } catch (error: any) {
    health.postgres = { connected: false, error: error.message };
  }

  try {
    const { qdrantManager } = await import('./qdrant-enhanced.js');
    health.qdrant = await qdrantManager.getHealthStatus();
  } catch (error: any) {
    health.qdrant = { connected: false, collection: 'legal_documents', error: error.message };
  }

  // Determine overall health
  if (health.postgres.connected && health.qdrant.connected) {
    health.overall = 'healthy';
  } else if (health.postgres.connected || health.qdrant.connected) {
    health.overall = 'degraded';
  } else {
    health.overall = 'unhealthy';
  }

  return health;
}

// Database utilities
export const databaseUtils = {
  /**
   * Migrate a document from old schema to new enhanced schema
   */
  async migrateDocument(oldDocument: any): Promise<NewLegalDocument> {
    return {
      title: oldDocument.title || 'Untitled Document',
      content: oldDocument.content || '',
      document_type: oldDocument.documentType || oldDocument.document_type || 'general',
      jurisdiction: oldDocument.jurisdiction || 'federal',
      analysis_results: oldDocument.metadata || {}
    };
  },

  /**
   * Validate embedding dimensions
   */
  validateEmbedding(embedding: any): embedding is number[] {
    return Array.isArray(embedding) && 
           embedding.length > 0 && 
           embedding.every(val => typeof val === 'number');
  },

  /**
   * Serialize embedding for database storage
   */
  serializeEmbedding(embedding: number[]): string {
    return JSON.stringify(embedding);
  },

  /**
   * Deserialize embedding from database
   */
  deserializeEmbedding(embeddingStr: string | null): number[] {
    if (!embeddingStr) return [];
    try {
      const parsed = JSON.parse(embeddingStr);
      return this.validateEmbedding(parsed) ? parsed : [];
    } catch {
      return [];
    }
  },

  /**
   * Generate unique document ID
   */
  generateDocumentId(): string {
    return `doc_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  },

  /**
   * Calculate similarity between embeddings
   */
  calculateCosineSimilarity(embedding1: number[], embedding2: number[]): number {
    if (embedding1.length !== embedding2.length) {
      throw new Error('Embeddings must have the same dimensions');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      norm1 += embedding1[i] * embedding1[i];
      norm2 += embedding2[i] * embedding2[i];
    }

    if (norm1 === 0 || norm2 === 0) {
      return 0;
    }

    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }
};

// Legacy compatibility - re-export from old locations
export { db as postgres } from './postgres-enhanced';
export { qdrantManager as qdrant } from './qdrant-enhanced';

// Default export
export default {
  initializeDatabase,
  getDatabaseHealth,
  databaseUtils
};
