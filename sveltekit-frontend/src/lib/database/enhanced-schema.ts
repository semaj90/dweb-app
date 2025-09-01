
/**
 * Enhanced Database Schema with Advanced pgvector Integration
 * Optimized for LangChain-Ollama workflows with nomic-embed-text
 */

import { 
  pgTable, 
  text, 
  timestamp, 
  integer, 
  boolean, 
  json, 
  uuid, 
  varchar, 
  real, 
  index, 
  customType,
  type PgColumn
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import { relations } from "drizzle-orm/relations";

// Custom vector type for pgvector
const vector = customType<{ data: number[]; driverData: string }>({
  dataType(config) {
    return `vector(${(config as any)?.dimensions ?? 1536})`;
  },
  toDriver(value: number[]): string {
    return `[${value.join(',')}]`;
  },
  fromDriver(value: string): number[] {
    return value.slice(1, -1).split(',').map(Number);
  }
});

// Core tables (existing)
export const users = pgTable('users', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  email: varchar('email', { length: 255 }).notNull().unique(),
  passwordHash: text('password_hash').notNull(),
  firstName: varchar('first_name', { length: 100 }),
  lastName: varchar('last_name', { length: 100 }),
  role: varchar('role', { length: 50 }).notNull().default('user'),
  isActive: boolean('is_active').default(true),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  status: varchar('status', { length: 50 }).default('active'),
  priority: varchar('priority', { length: 20 }).default('medium'),
  caseNumber: varchar('case_number', { length: 100 }).unique(),
  createdBy: uuid('created_by').references(() => users.id),
  assignedTo: uuid('assigned_to').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Enhanced Documents table with nomic-embed-text (768 dimensions)
export const documents = pgTable('documents', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  caseId: uuid('case_id').references(() => cases.id),
  title: varchar('title', { length: 255 }).notNull(),
  filename: varchar('filename', { length: 255 }),
  fileType: varchar('file_type', { length: 50 }),
  fileSize: integer('file_size'),
  content: text('content'),
  extractedText: text('extracted_text'),
  // Using 768 dimensions for nomic-embed-text
  embedding: vector('embedding', { dimensions: 384 }),
  metadata: json('metadata'),
  tags: json('tags').default(sql`'[]'::json`),
  isIndexed: boolean('is_indexed').default(false),
  source: varchar('source', { length: 100 }).default('upload'),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => ({
  // Optimized indexes for vector operations
  embeddingIdx: index('documents_embedding_idx').using('ivfflat', table.embedding.op('vector_cosine_ops')),
  caseIdx: index('documents_case_idx').on(table.caseId),
  contentIdx: index('documents_content_idx').using('gin', sql`to_tsvector('english', ${table.content})`)
}));

// Document chunks for optimized retrieval
export const documentChunks = pgTable('document_chunks', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  documentId: uuid('document_id').references(() => documents.id).notNull(),
  chunkIndex: integer('chunk_index').notNull(),
  content: text('content').notNull(),
  // 768-dimensional embeddings for nomic-embed-text
  embedding: vector('embedding', { dimensions: 384 }).notNull(),
  startIndex: integer('start_index'),
  endIndex: integer('end_index'),
  tokenCount: integer('token_count'),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow()
}, (table) => ({
  // Optimized vector search index
  embeddingIdx: index('document_chunks_embedding_idx').using('ivfflat', table.embedding.op('vector_cosine_ops')),
  documentIdx: index('document_chunks_document_idx').on(table.documentId),
  chunkIdx: index('document_chunks_chunk_idx').on(table.documentId, table.chunkIndex)
}));

// Enhanced evidence table
export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  caseId: uuid('case_id').references(() => cases.id),
  documentId: uuid('document_id').references(() => documents.id),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  evidenceType: varchar('evidence_type', { length: 50 }),
  hash: varchar('hash', { length: 256 }),
  chainOfCustody: json('chain_of_custody').default(sql`'[]'::json`),
  isAdmissible: boolean('is_admissible'),
  admissibilityNotes: text('admissibility_notes'),
  tags: json('tags').default(sql`'[]'::json`),
  // Enhanced AI analysis with embedding support
  aiAnalysis: json('ai_analysis'),
  // Vector embedding for semantic search
  embedding: vector('embedding', { dimensions: 384 }),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => ({
  embeddingIdx: index('evidence_embedding_idx').using('ivfflat', table.embedding.op('vector_cosine_ops')),
  caseIdx: index('evidence_case_idx').on(table.caseId)
}));

// Enhanced search index with optimized vector operations
export const searchIndex = pgTable('search_index', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  entityType: varchar('entity_type', { length: 50 }).notNull(),
  entityId: uuid('entity_id').notNull(),
  content: text('content').notNull(),
  // 768-dimensional embeddings for nomic-embed-text
  embedding: vector('embedding', { dimensions: 384 }).notNull(),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => ({
  // High-performance vector index for similarity search
  embeddingIdx: index('search_index_embedding_idx').using('ivfflat', table.embedding.op('vector_cosine_ops')),
  entityIdx: index('search_index_entity_idx').on(table.entityType, table.entityId),
  contentIdx: index('search_index_content_idx').using('gin', sql`to_tsvector('english', ${table.content})`)
}));

// AI chat interactions with conversation context
export const aiInteractions = pgTable('ai_interactions', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  userId: uuid('user_id').references(() => users.id),
  caseId: uuid('case_id').references(() => cases.id),
  sessionId: varchar('session_id', { length: 255 }),
  prompt: text('prompt').notNull(),
  response: text('response').notNull(),
  model: varchar('model', { length: 100 }),
  tokensUsed: integer('tokens_used'),
  responseTime: integer('response_time'),
  confidence: real('confidence'),
  // Context embedding for conversation understanding
  contextEmbedding: vector('context_embedding', { dimensions: 384 }),
  feedback: json('feedback'),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow()
}, (table) => ({
  contextEmbeddingIdx: index('ai_interactions_context_embedding_idx').using('ivfflat', table.contextEmbedding.op('vector_cosine_ops')),
  sessionIdx: index('ai_interactions_session_idx').on(table.sessionId),
  userIdx: index('ai_interactions_user_idx').on(table.userId)
}));

// Vector similarity cache for performance optimization
export const vectorSimilarityCache = pgTable('vector_similarity_cache', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  queryHash: varchar('query_hash', { length: 64 }).notNull().unique(),
  queryEmbedding: vector('query_embedding', { dimensions: 384 }).notNull(),
  results: json('results').notNull(),
  hitCount: integer('hit_count').default(1),
  lastAccessed: timestamp('last_accessed').defaultNow(),
  expiresAt: timestamp('expires_at'),
  createdAt: timestamp('created_at').defaultNow()
}, (table) => ({
  queryHashIdx: index('vector_similarity_cache_hash_idx').on(table.queryHash),
  expiresIdx: index('vector_similarity_cache_expires_idx').on(table.expiresAt)
}));

// Legal knowledge base with semantic embeddings
export const legalKnowledgeBase = pgTable('legal_knowledge_base', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content').notNull(),
  category: varchar('category', { length: 100 }),
  subcategory: varchar('subcategory', { length: 100 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  source: varchar('source', { length: 255 }),
  sourceUrl: text('source_url'),
  citationFormat: text('citation_format'),
  // Semantic embedding for knowledge retrieval
  embedding: vector('embedding', { dimensions: 384 }),
  metadata: json('metadata'),
  isVerified: boolean('is_verified').default(false),
  verifiedBy: uuid('verified_by').references(() => users.id),
  verifiedAt: timestamp('verified_at'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => ({
  embeddingIdx: index('legal_knowledge_base_embedding_idx').using('ivfflat', table.embedding.op('vector_cosine_ops')),
  categoryIdx: index('legal_knowledge_base_category_idx').on(table.category, table.subcategory),
  jurisdictionIdx: index('legal_knowledge_base_jurisdiction_idx').on(table.jurisdiction)
}));

// Embedding processing jobs for background processing
export const embeddingJobs = pgTable('embedding_jobs', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  entityType: varchar('entity_type', { length: 50 }).notNull(),
  entityId: uuid('entity_id').notNull(),
  jobType: varchar('job_type', { length: 50 }).notNull(), // 'embedding', 'reembedding', 'similarity_update'
  status: varchar('status', { length: 20 }).default('pending'), // 'pending', 'processing', 'completed', 'failed'
  progress: integer('progress').default(0), // 0-100
  model: varchar('model', { length: 100 }),
  batchSize: integer('batch_size'),
  priority: integer('priority').default(5), // 1-10, higher is more priority
  retryCount: integer('retry_count').default(0),
  maxRetries: integer('max_retries').default(3),
  error: text('error'),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
}, (table) => ({
  statusIdx: index('embedding_jobs_status_idx').on(table.status),
  entityIdx: index('embedding_jobs_entity_idx').on(table.entityType, table.entityId),
  priorityIdx: index('embedding_jobs_priority_idx').on(table.priority, table.createdAt)
}));

// Define relations
export const documentsRelations = relations(documents, ({ one, many }) => ({
  case: one(cases, {
    fields: [documents.caseId],
    references: [cases.id]
  }),
  creator: one(users, {
    fields: [documents.createdBy],
    references: [users.id]
  }),
  chunks: many(documentChunks),
  evidence: many(evidence)
}));

export const documentChunksRelations = relations(documentChunks, ({ one }) => ({
  document: one(documents, {
    fields: [documentChunks.documentId],
    references: [documents.id]
  })
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({
  case: one(cases, {
    fields: [evidence.caseId],
    references: [cases.id]
  }),
  document: one(documents, {
    fields: [evidence.documentId],
    references: [documents.id]
  }),
  creator: one(users, {
    fields: [evidence.createdBy],
    references: [users.id]
  })
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  creator: one(users, {
    fields: [cases.createdBy],
    references: [users.id]
  }),
  assignee: one(users, {
    fields: [cases.assignedTo],
    references: [users.id]
  }),
  documents: many(documents),
  evidence: many(evidence),
  aiInteractions: many(aiInteractions)
}));

export const aiInteractionsRelations = relations(aiInteractions, ({ one }) => ({
  user: one(users, {
    fields: [aiInteractions.userId],
    references: [users.id]
  }),
  case: one(cases, {
    fields: [aiInteractions.caseId],
    references: [cases.id]
  })
}));

export const legalKnowledgeBaseRelations = relations(legalKnowledgeBase, ({ one }) => ({
  verifier: one(users, {
    fields: [legalKnowledgeBase.verifiedBy],
    references: [users.id]
  })
}));

// Type exports
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;

export type Document = typeof documents.$inferSelect;
export type NewDocument = typeof documents.$inferInsert;

export type DocumentChunk = typeof documentChunks.$inferSelect;
export type NewDocumentChunk = typeof documentChunks.$inferInsert;

export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;

export type SearchIndex = typeof searchIndex.$inferSelect;
export type NewSearchIndex = typeof searchIndex.$inferInsert;

export type AIInteraction = typeof aiInteractions.$inferSelect;
export type NewAIInteraction = typeof aiInteractions.$inferInsert;

export type VectorSimilarityCache = typeof vectorSimilarityCache.$inferSelect;
export type NewVectorSimilarityCache = typeof vectorSimilarityCache.$inferInsert;

export type LegalKnowledgeBase = typeof legalKnowledgeBase.$inferSelect;
export type NewLegalKnowledgeBase = typeof legalKnowledgeBase.$inferInsert;

export type EmbeddingJob = typeof embeddingJobs.$inferSelect;
export type NewEmbeddingJob = typeof embeddingJobs.$inferInsert;

// AI Processing Jobs table
export const aiProcessingJobs = pgTable('ai_processing_jobs', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  type: varchar('type', { length: 50 }).notNull(),
  status: varchar('status', { length: 50 }).default('pending'),
  input: json('input'),
  output: json('output'),
  error: text('error'),
  progress: integer('progress').default(0),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  completedAt: timestamp('completed_at')
});

export type AIProcessingJob = typeof aiProcessingJobs.$inferSelect;
export type NewAIProcessingJob = typeof aiProcessingJobs.$inferInsert;