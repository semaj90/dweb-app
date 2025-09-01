/**
 * Enhanced Unified Schema for SvelteKit Legal AI Platform
 * PostgreSQL + pgvector + Qdrant Integration with Drizzle ORM
 */

import { 
  pgTable, 
  uuid, 
  varchar, 
  text, 
  timestamp, 
  integer, 
  decimal, 
  boolean, 
  jsonb, 
  serial,
  vector,
  index,
  uniqueIndex,
  real
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { relations } from 'drizzle-orm/relations';
import { createId } from '@paralleldrive/cuid2';
import type { z } from 'zod';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface DocumentMetadata {
  keywords?: string[];
  customFields?: Record<string, unknown>;
  confidentialityLevel?: 'public' | 'restricted' | 'confidential' | 'top_secret';
  documentType?: 'contract' | 'brief' | 'evidence' | 'case_file';
  jurisdiction?: string;
  practiceArea?: string;
  caseNumber?: string;
  clientId?: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  tags?: string[];
  aiGenerated?: boolean;
  vectorized?: boolean;
  qdrantId?: string; // For Qdrant vector database sync
}

export interface VectorMetadata {
  modelName: string;
  dimensions: number;
  similarity: 'cosine' | 'dot' | 'euclidean';
  createdAt: string;
  lastSynced?: string;
  qdrantCollection?: string;
}

// ============================================================================
// USER MANAGEMENT TABLES
// ============================================================================

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  hashedPassword: varchar('hashed_password', { length: 255 }),
  username: varchar('username', { length: 100 }),
  firstName: varchar('first_name', { length: 100 }),
  lastName: varchar('last_name', { length: 100 }),
  role: varchar('role', { length: 50 }).default('user').notNull(),
  department: varchar('department', { length: 100 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  permissions: jsonb('permissions').default([]).notNull(),
  isActive: boolean('is_active').default(true).notNull(),
  emailVerified: boolean('email_verified').default(false).notNull(),
  avatarUrl: varchar('avatar_url', { length: 500 }),
  lastLoginAt: timestamp('last_login_at', { mode: 'date' }),
  
  // Legal AI specific fields
  practiceAreas: jsonb('practice_areas').default([]),
  barNumber: varchar('bar_number', { length: 50 }),
  firmName: varchar('firm_name', { length: 200 }),
  
  // Vector embeddings for AI recommendations (384 dimensions for nomic-embed-text)
  profileEmbedding: vector('profile_embedding', { dimensions: 384 }),
  
  // Metadata and timestamps
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
  deletedAt: timestamp('deleted_at', { mode: 'date' }),
}, (table) => ({
  emailIdx: uniqueIndex('users_email_idx').on(table.email),
  usernameIdx: index('users_username_idx').on(table.username),
  roleIdx: index('users_role_idx').on(table.role),
  activeIdx: index('users_active_idx').on(table.isActive),
  profileEmbeddingIdx: index('users_profile_embedding_hnsw_idx')
    .using('hnsw', table.profileEmbedding.op('vector_cosine_ops'))
}));

export const sessions = pgTable('sessions', {
  id: varchar('id', { length: 255 }).primaryKey(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: timestamp('expires_at', { mode: 'date' }).notNull(),
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  sessionContext: jsonb('session_context').default({}),
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  userIdIdx: index('sessions_user_id_idx').on(table.userId),
  expiresAtIdx: index('sessions_expires_at_idx').on(table.expiresAt),
}));

// ============================================================================
// KEYS TABLE (AUTHENTICATION KEYS MANAGEMENT)
// ============================================================================

export const keys = pgTable('keys', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  hashedPassword: varchar('hashed_password', { length: 255 }),
  keyType: varchar('key_type', { length: 50 }).notNull(), // 'password', 'oauth', 'api_key'
  keyId: varchar('key_id', { length: 255 }).notNull(),
  keyValue: text('key_value'), // Encrypted key data
  expiresAt: timestamp('expires_at', { mode: 'date' }),
  isActive: boolean('is_active').default(true).notNull(),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  userIdIdx: index('keys_user_id_idx').on(table.userId),
  keyIdIdx: uniqueIndex('keys_key_id_idx').on(table.keyId),
  keyTypeIdx: index('keys_key_type_idx').on(table.keyType),
  activeIdx: index('keys_active_idx').on(table.isActive),
}));

// ============================================================================
// LEGAL DOCUMENT TABLES WITH VECTOR SUPPORT
// ============================================================================

export const legalDocuments = pgTable('legal_documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 500 }).notNull(),
  content: text('content').notNull(),
  summary: text('summary'),
  
  // Document classification
  documentType: varchar('document_type', { length: 50 }).notNull().default('document'),
  practiceArea: varchar('practice_area', { length: 100 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  caseNumber: varchar('case_number', { length: 100 }),
  
  // File information
  filePath: varchar('file_path', { length: 1000 }),
  fileName: varchar('file_name', { length: 255 }),
  fileSize: integer('file_size'),
  mimeType: varchar('mime_type', { length: 100 }),
  
  // Vector embeddings for semantic search
  titleEmbedding: vector('title_embedding', { dimensions: 384 }),
  contentEmbedding: vector('content_embedding', { dimensions: 384 }),
  summaryEmbedding: vector('summary_embedding', { dimensions: 384 }),
  
  // Qdrant integration
  qdrantId: uuid('qdrant_id'), // UUID for Qdrant point ID
  qdrantCollection: varchar('qdrant_collection', { length: 100 }).default('legal_documents'),
  lastSyncedToQdrant: timestamp('last_synced_to_qdrant', { mode: 'date' }),
  
  // Metadata and relationships
  userId: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'set null' }),
  metadata: jsonb('metadata').default({}),
  
  // Status and visibility
  status: varchar('status', { length: 20 }).default('active').notNull(),
  visibility: varchar('visibility', { length: 20 }).default('private').notNull(),
  
  // AI processing fields
  aiProcessed: boolean('ai_processed').default(false).notNull(),
  confidenceScore: real('confidence_score').default(0),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
  deletedAt: timestamp('deleted_at', { mode: 'date' }),
}, (table) => ({
  titleIdx: index('legal_documents_title_idx').on(table.title),
  documentTypeIdx: index('legal_documents_type_idx').on(table.documentType),
  practiceAreaIdx: index('legal_documents_practice_area_idx').on(table.practiceArea),
  userIdIdx: index('legal_documents_user_id_idx').on(table.userId),
  caseIdIdx: index('legal_documents_case_id_idx').on(table.caseId),
  statusIdx: index('legal_documents_status_idx').on(table.status),
  qdrantIdIdx: uniqueIndex('legal_documents_qdrant_id_idx').on(table.qdrantId),
  
  // Vector similarity indexes for fast semantic search
  titleEmbeddingIdx: index('legal_documents_title_embedding_hnsw_idx')
    .using('hnsw', table.titleEmbedding.op('vector_cosine_ops'))
    .with({ 'm': 16, 'ef_construction': 64 }),
  contentEmbeddingIdx: index('legal_documents_content_embedding_hnsw_idx')
    .using('hnsw', table.contentEmbedding.op('vector_cosine_ops'))
    .with({ 'm': 16, 'ef_construction': 64 }),
  summaryEmbeddingIdx: index('legal_documents_summary_embedding_hnsw_idx')
    .using('hnsw', table.summaryEmbedding.op('vector_cosine_ops'))
    .with({ 'm': 16, 'ef_construction': 64 }),
}));

export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 500 }).notNull(),
  description: text('description'),
  caseNumber: varchar('case_number', { length: 100 }).unique(),
  
  // Case details
  status: varchar('status', { length: 50 }).default('active').notNull(),
  priority: varchar('priority', { length: 20 }).default('medium').notNull(),
  practiceArea: varchar('practice_area', { length: 100 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  court: varchar('court', { length: 200 }),
  
  // Parties and representatives
  clientName: varchar('client_name', { length: 200 }),
  opposingParty: varchar('opposing_party', { length: 200 }),
  assignedAttorney: uuid('assigned_attorney').references(() => users.id),
  
  // Dates
  filingDate: timestamp('filing_date', { mode: 'date' }),
  dueDate: timestamp('due_date', { mode: 'date' }),
  closedDate: timestamp('closed_date', { mode: 'date' }),
  
  // Vector embedding for case similarity
  caseEmbedding: vector('case_embedding', { dimensions: 384 }),
  
  // Qdrant integration
  qdrantId: uuid('qdrant_id'),
  qdrantCollection: varchar('qdrant_collection', { length: 100 }).default('cases'),
  
  // Metadata
  metadata: jsonb('metadata').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  caseNumberIdx: uniqueIndex('cases_case_number_idx').on(table.caseNumber),
  statusIdx: index('cases_status_idx').on(table.status),
  practiceAreaIdx: index('cases_practice_area_idx').on(table.practiceArea),
  assignedAttorneyIdx: index('cases_assigned_attorney_idx').on(table.assignedAttorney),
  caseEmbeddingIdx: index('cases_case_embedding_hnsw_idx')
    .using('hnsw', table.caseEmbedding.op('vector_cosine_ops')),
}));

// ============================================================================
// VECTOR OPERATIONS AND QDRANT SYNC TABLES
// ============================================================================

export const vectorOperations = pgTable('vector_operations', {
  id: uuid('id').primaryKey().defaultRandom(),
  operationType: varchar('operation_type', { length: 50 }).notNull(), // 'embed', 'search', 'sync'
  entityType: varchar('entity_type', { length: 50 }).notNull(), // 'document', 'case', 'user'
  entityId: uuid('entity_id').notNull(),
  
  // Vector operation details
  modelName: varchar('model_name', { length: 100 }).notNull().default('nomic-embed-text'),
  dimensions: integer('dimensions').notNull().default(384),
  similarity: varchar('similarity', { length: 20 }).default('cosine'),
  
  // Performance metrics
  processingTimeMs: integer('processing_time_ms'),
  similarity_score: real('similarity_score'),
  
  // Qdrant sync status
  qdrantSynced: boolean('qdrant_synced').default(false).notNull(),
  qdrantSyncedAt: timestamp('qdrant_synced_at', { mode: 'date' }),
  qdrantError: text('qdrant_error'),
  
  // Status and metadata
  status: varchar('status', { length: 20 }).default('pending').notNull(), // 'pending', 'processing', 'completed', 'failed'
  metadata: jsonb('metadata').default({}),
  error: text('error'),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  completedAt: timestamp('completed_at', { mode: 'date' }),
}, (table) => ({
  operationTypeIdx: index('vector_operations_operation_type_idx').on(table.operationType),
  entityTypeIdx: index('vector_operations_entity_type_idx').on(table.entityType),
  entityIdIdx: index('vector_operations_entity_id_idx').on(table.entityId),
  statusIdx: index('vector_operations_status_idx').on(table.status),
  qdrantSyncedIdx: index('vector_operations_qdrant_synced_idx').on(table.qdrantSynced),
}));

export const qdrantCollections = pgTable('qdrant_collections', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 100 }).notNull().unique(),
  description: text('description'),
  
  // Vector configuration
  vectorSize: integer('vector_size').notNull().default(384),
  distance: varchar('distance', { length: 20 }).default('Cosine').notNull(),
  
  // Collection status
  status: varchar('status', { length: 20 }).default('active').notNull(),
  isOptimized: boolean('is_optimized').default(false).notNull(),
  
  // Statistics
  pointsCount: integer('points_count').default(0).notNull(),
  lastSynced: timestamp('last_synced', { mode: 'date' }),
  
  // Configuration
  config: jsonb('config').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  nameIdx: uniqueIndex('qdrant_collections_name_idx').on(table.name),
  statusIdx: index('qdrant_collections_status_idx').on(table.status),
}));

// ============================================================================
// RELATIONS
// ============================================================================

export const usersRelations = relations(users, ({ many, one }) => ({
  sessions: many(sessions),
  keys: many(keys),
  documents: many(legalDocuments),
  assignedCases: many(cases, { relationName: 'assignedAttorney' }),
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.userId],
    references: [users.id],
  }),
}));

export const keysRelations = relations(keys, ({ one }) => ({
  user: one(users, {
    fields: [keys.userId],
    references: [users.id],
  }),
}));

export const legalDocumentsRelations = relations(legalDocuments, ({ one }) => ({
  user: one(users, {
    fields: [legalDocuments.userId],
    references: [users.id],
  }),
  case: one(cases, {
    fields: [legalDocuments.caseId],
    references: [cases.id],
  }),
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  assignedAttorney: one(users, {
    fields: [cases.assignedAttorney],
    references: [users.id],
  }),
  documents: many(legalDocuments),
}));

// ============================================================================
// HELPER FUNCTIONS FOR QDRANT INTEGRATION
// ============================================================================

export const createQdrantPoint = (documentId: string, embedding: number[], metadata: Record<string, any>) => {
  return {
    id: documentId,
    vector: embedding,
    payload: {
      ...metadata,
      postgres_id: documentId,
      synced_at: new Date().toISOString(),
    },
  };
};

export const getVectorSimilarityQuery = (
  table: typeof legalDocuments,
  embeddingColumn: typeof legalDocuments.contentEmbedding,
  queryVector: number[],
  limit = 10,
  threshold = 0.7
) => {
  return sql`
    SELECT *, 
           (${embeddingColumn} <=> ${JSON.stringify(queryVector)}::vector) as distance,
           (1 - (${embeddingColumn} <=> ${JSON.stringify(queryVector)}::vector)) as similarity
    FROM ${table}
    WHERE (1 - (${embeddingColumn} <=> ${JSON.stringify(queryVector)}::vector)) >= ${threshold}
    ORDER BY ${embeddingColumn} <=> ${JSON.stringify(queryVector)}::vector
    LIMIT ${limit}
  `;
};

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;
export type Key = typeof keys.$inferSelect;
export type NewKey = typeof keys.$inferInsert;
export type LegalDocument = typeof legalDocuments.$inferSelect;
export type NewLegalDocument = typeof legalDocuments.$inferInsert;
export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;
export type VectorOperation = typeof vectorOperations.$inferSelect;
export type NewVectorOperation = typeof vectorOperations.$inferInsert;
export type QdrantCollection = typeof qdrantCollections.$inferSelect;
export type NewQdrantCollection = typeof qdrantCollections.$inferInsert;