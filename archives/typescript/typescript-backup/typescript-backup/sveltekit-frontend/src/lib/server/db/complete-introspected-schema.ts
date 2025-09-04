/**
 * Complete Introspected Schema for Legal AI Database
 * Generated from PostgreSQL introspection (60+ tables)
 * Includes pgvector extension and missing keys table
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
  real,
  primaryKey,
  foreignKey,
  bigint,
  date
} from 'drizzle-orm/pg-core';
import { relations, sql } from 'drizzle-orm';

// ============================================================================
// CORE USER MANAGEMENT TABLES
// ============================================================================

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  hashedPassword: varchar('hashed_password', { length: 255 }),
  username: varchar('username', { length: 100 }),
  firstName: varchar('first_name', { length: 100 }),
  lastName: varchar('last_name', { length: 100 }),
  name: varchar('name', { length: 255 }),
  role: varchar('role', { length: 50 }).default('prosecutor').notNull(),
  department: varchar('department', { length: 100 }),
  badgeNumber: varchar('badge_number', { length: 50 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  displayName: varchar('display_name', { length: 255 }),
  avatarUrl: text('avatar_url'),
  isActive: boolean('is_active').default(true).notNull(),
  emailVerified: boolean('email_verified').default(false).notNull(),
  lastLoginAt: timestamp('last_login_at', { mode: 'date' }),
  
  // Legal AI specific fields
  practiceAreas: jsonb('practice_areas').default([]),
  barNumber: varchar('bar_number', { length: 50 }),
  firmName: varchar('firm_name', { length: 200 }),
  
  // Vector embeddings for AI recommendations
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
}));

export const userProfiles = pgTable('user_profiles', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  bio: text('bio'),
  phone: varchar('phone', { length: 20 }),
  address: text('address'),
  city: varchar('city', { length: 100 }),
  state: varchar('state', { length: 50 }),
  zipCode: varchar('zip_code', { length: 10 }),
  country: varchar('country', { length: 100 }).default('USA'),
  timezone: varchar('timezone', { length: 50 }).default('America/New_York'),
  preferences: jsonb('preferences').default({}),
  socialLinks: jsonb('social_links').default({}),
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  userIdIdx: index('user_profiles_user_id_idx').on(table.userId),
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
// KEYS TABLE (MISSING FROM SCHEMA)
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
// LEGAL CASE MANAGEMENT TABLES
// ============================================================================

export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  title: varchar('title', { length: 500 }).notNull(),
  description: text('description'),
  caseNumber: varchar('case_number', { length: 100 }),
  
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
  userIdIdx: index('cases_user_id_idx').on(table.userId),
}));

// ============================================================================
// EVIDENCE MANAGEMENT TABLES
// ============================================================================

export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').notNull().references(() => cases.id, { onDelete: 'cascade' }),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  title: varchar('title', { length: 500 }).notNull(),
  description: text('description'),
  evidenceType: varchar('evidence_type', { length: 50 }).notNull(),
  subType: varchar('sub_type', { length: 100 }),
  
  // File information
  fileName: varchar('file_name', { length: 255 }),
  filePath: varchar('file_path', { length: 1000 }),
  fileSize: bigint('file_size', { mode: 'number' }),
  mimeType: varchar('mime_type', { length: 100 }),
  fileHash: varchar('file_hash', { length: 64 }),
  
  // Chain of custody
  collectedAt: timestamp('collected_at', { mode: 'date' }),
  collectedBy: varchar('collected_by', { length: 255 }),
  location: text('location'),
  chainOfCustody: jsonb('chain_of_custody').default([]),
  
  // AI analysis
  aiAnalysis: jsonb('ai_analysis').default({}),
  aiTags: jsonb('ai_tags').default([]),
  aiSummary: text('ai_summary'),
  
  // Vector embedding
  contentEmbedding: vector('content_embedding', { dimensions: 384 }),
  
  // Status and metadata
  isAdmissible: boolean('is_admissible').default(true),
  confidentialityLevel: varchar('confidentiality_level', { length: 20 }).default('restricted'),
  status: varchar('status', { length: 20 }).default('active').notNull(),
  metadata: jsonb('metadata').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  caseIdIdx: index('evidence_case_id_idx').on(table.caseId),
  evidenceTypeIdx: index('evidence_evidence_type_idx').on(table.evidenceType),
  statusIdx: index('evidence_status_idx').on(table.status),
  userIdIdx: index('evidence_user_id_idx').on(table.userId),
}));

// ============================================================================
// LEGAL DOCUMENTS TABLES
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
}));

export const documentProcessing = pgTable('document_processing', {
  id: uuid('id').primaryKey().defaultRandom(),
  documentId: uuid('document_id').references(() => legalDocuments.id, { onDelete: 'cascade' }),
  originalName: varchar('original_name', { length: 255 }),
  documentType: varchar('document_type', { length: 50 }),
  caseId: uuid('case_id').references(() => cases.id),
  textLength: integer('text_length'),
  summary: text('summary'),
  extractedText: text('extracted_text'),
  ocrText: text('ocr_text'),
  metadata: jsonb('metadata').default({}),
  processingStatus: varchar('processing_status', { length: 50 }).default('pending'),
  processingError: text('processing_error'),
  confidence: real('confidence'),
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  documentIdIdx: index('document_processing_document_id_idx').on(table.documentId),
  caseIdIdx: index('document_processing_case_id_idx').on(table.caseId),
  statusIdx: index('document_processing_status_idx').on(table.processingStatus),
}));

export const documentChunks = pgTable('document_chunks', {
  id: uuid('id').primaryKey().defaultRandom(),
  documentId: uuid('document_id').notNull().references(() => legalDocuments.id, { onDelete: 'cascade' }),
  chunkIndex: integer('chunk_index').notNull(),
  content: text('content').notNull(),
  chunkHash: varchar('chunk_hash', { length: 64 }),
  startPosition: integer('start_position'),
  endPosition: integer('end_position'),
  tokenCount: integer('token_count'),
  
  // Vector embedding
  embedding: vector('embedding', { dimensions: 384 }),
  
  // Metadata
  metadata: jsonb('metadata').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  documentIdIdx: index('document_chunks_document_id_idx').on(table.documentId),
  chunkIndexIdx: index('document_chunks_chunk_index_idx').on(table.chunkIndex),
  hashIdx: index('document_chunks_hash_idx').on(table.chunkHash),
}));

// ============================================================================
// REPORTS AND ANALYTICS TABLES
// ============================================================================

export const reports = pgTable('reports', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'set null' }),
  title: varchar('title', { length: 500 }).notNull(),
  description: text('description'),
  reportType: varchar('report_type', { length: 50 }).notNull(),
  
  // Report content
  content: text('content'),
  summary: text('summary'),
  findings: jsonb('findings').default({}),
  recommendations: jsonb('recommendations').default([]),
  
  // Status and visibility
  status: varchar('status', { length: 20 }).default('draft').notNull(),
  visibility: varchar('visibility', { length: 20 }).default('private').notNull(),
  
  // AI analysis
  aiGenerated: boolean('ai_generated').default(false),
  confidenceScore: real('confidence_score'),
  
  // Vector embedding
  contentEmbedding: vector('content_embedding', { dimensions: 384 }),
  
  // Metadata
  metadata: jsonb('metadata').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
  publishedAt: timestamp('published_at', { mode: 'date' }),
}, (table) => ({
  userIdIdx: index('reports_user_id_idx').on(table.userId),
  caseIdIdx: index('reports_case_id_idx').on(table.caseId),
  reportTypeIdx: index('reports_report_type_idx').on(table.reportType),
  statusIdx: index('reports_status_idx').on(table.status),
}));

// ============================================================================
// VECTOR OPERATIONS AND SEARCH TABLES
// ============================================================================

export const vectors = pgTable('vectors', {
  id: uuid('id').primaryKey().defaultRandom(),
  entityType: varchar('entity_type', { length: 50 }).notNull(), // 'document', 'case', 'evidence', 'user'
  entityId: uuid('entity_id').notNull(),
  vectorType: varchar('vector_type', { length: 50 }).notNull(), // 'content', 'title', 'summary', 'profile'
  
  // Vector data
  embedding: vector('embedding', { dimensions: 384 }).notNull(),
  modelName: varchar('model_name', { length: 100 }).default('nomic-embed-text'),
  dimensions: integer('dimensions').default(384),
  
  // Performance metrics
  similarity: real('similarity'),
  confidence: real('confidence'),
  
  // Metadata
  metadata: jsonb('metadata').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  entityTypeIdx: index('vectors_entity_type_idx').on(table.entityType),
  entityIdIdx: index('vectors_entity_id_idx').on(table.entityId),
  vectorTypeIdx: index('vectors_vector_type_idx').on(table.vectorType),
  modelIdx: index('vectors_model_idx').on(table.modelName),
  // Vector similarity index for fast semantic search
  embeddingIdx: index('vectors_embedding_hnsw_idx')
    .using('hnsw', table.embedding.op('vector_cosine_ops'))
    .with({ 'm': 16, 'ef_construction': 64 }),
}));

export const vectorOperations = pgTable('vector_operations', {
  id: uuid('id').primaryKey().defaultRandom(),
  operationType: varchar('operation_type', { length: 50 }).notNull(), // 'embed', 'search', 'sync'
  entityType: varchar('entity_type', { length: 50 }).notNull(), // 'document', 'case', 'evidence', 'user'
  entityId: uuid('entity_id').notNull(),
  
  // Vector operation details
  modelName: varchar('model_name', { length: 100 }).notNull().default('nomic-embed-text'),
  dimensions: integer('dimensions').notNull().default(384),
  similarity: varchar('similarity', { length: 20 }).default('cosine'),
  
  // Performance metrics
  processingTimeMs: integer('processing_time_ms'),
  similarityScore: real('similarity_score'),
  
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
}));

// ============================================================================
// AI AND CHAT TABLES
// ============================================================================

export const chatSessions = pgTable('chat_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'set null' }),
  title: varchar('title', { length: 255 }),
  sessionType: varchar('session_type', { length: 50 }).default('general'), // 'general', 'case_analysis', 'legal_research'
  
  // Session context
  context: jsonb('context').default({}),
  systemPrompt: text('system_prompt'),
  
  // Status
  status: varchar('status', { length: 20 }).default('active'), // 'active', 'archived', 'deleted'
  isArchived: boolean('is_archived').default(false),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
  lastMessageAt: timestamp('last_message_at', { mode: 'date' }),
}, (table) => ({
  userIdIdx: index('chat_sessions_user_id_idx').on(table.userId),
  caseIdIdx: index('chat_sessions_case_id_idx').on(table.caseId),
  statusIdx: index('chat_sessions_status_idx').on(table.status),
  sessionTypeIdx: index('chat_sessions_session_type_idx').on(table.sessionType),
}));

export const chatMessages = pgTable('chat_messages', {
  id: uuid('id').primaryKey().defaultRandom(),
  sessionId: uuid('session_id').notNull().references(() => chatSessions.id, { onDelete: 'cascade' }),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  
  // Message content
  content: text('content').notNull(),
  role: varchar('role', { length: 20 }).notNull(), // 'user', 'assistant', 'system'
  messageType: varchar('message_type', { length: 50 }).default('text'), // 'text', 'file', 'image', 'analysis'
  
  // AI model information
  modelName: varchar('model_name', { length: 100 }),
  temperature: real('temperature'),
  maxTokens: integer('max_tokens'),
  
  // Vector embedding for semantic search
  contentEmbedding: vector('content_embedding', { dimensions: 384 }),
  
  // Metadata
  metadata: jsonb('metadata').default({}),
  attachments: jsonb('attachments').default([]),
  
  // Performance metrics
  responseTimeMs: integer('response_time_ms'),
  tokenCount: integer('token_count'),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  sessionIdIdx: index('chat_messages_session_id_idx').on(table.sessionId),
  userIdIdx: index('chat_messages_user_id_idx').on(table.userId),
  roleIdx: index('chat_messages_role_idx').on(table.role),
  createdAtIdx: index('chat_messages_created_at_idx').on(table.createdAt),
}));

// ============================================================================
// SYSTEM TABLES
// ============================================================================

export const apiKeys = pgTable('api_keys', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  name: varchar('name', { length: 255 }).notNull(),
  keyHash: varchar('key_hash', { length: 255 }).notNull().unique(),
  keyPrefix: varchar('key_prefix', { length: 20 }).notNull(),
  
  // Permissions and scope
  permissions: jsonb('permissions').default([]),
  scopes: jsonb('scopes').default([]),
  
  // Usage tracking
  usageCount: integer('usage_count').default(0),
  lastUsedAt: timestamp('last_used_at', { mode: 'date' }),
  
  // Status
  isActive: boolean('is_active').default(true).notNull(),
  expiresAt: timestamp('expires_at', { mode: 'date' }),
  
  // Metadata
  metadata: jsonb('metadata').default({}),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  userIdIdx: index('api_keys_user_id_idx').on(table.userId),
  keyHashIdx: uniqueIndex('api_keys_key_hash_idx').on(table.keyHash),
  activeIdx: index('api_keys_active_idx').on(table.isActive),
}));

export const systemLogs = pgTable('system_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id),
  level: varchar('level', { length: 20 }).notNull(), // 'debug', 'info', 'warn', 'error', 'fatal'
  category: varchar('category', { length: 50 }).notNull(), // 'auth', 'api', 'database', 'ai', 'vector'
  
  // Log content
  message: text('message').notNull(),
  details: jsonb('details').default({}),
  
  // Context
  requestId: varchar('request_id', { length: 255 }),
  sessionId: varchar('session_id', { length: 255 }),
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  
  // Performance
  duration: integer('duration'), // milliseconds
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  levelIdx: index('system_logs_level_idx').on(table.level),
  categoryIdx: index('system_logs_category_idx').on(table.category),
  userIdIdx: index('system_logs_user_id_idx').on(table.userId),
  createdAtIdx: index('system_logs_created_at_idx').on(table.createdAt),
  requestIdIdx: index('system_logs_request_id_idx').on(table.requestId),
}));

export const auditLogs = pgTable('audit_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id),
  entityType: varchar('entity_type', { length: 50 }).notNull(), // 'case', 'evidence', 'document', 'user'
  entityId: uuid('entity_id').notNull(),
  
  // Action details
  action: varchar('action', { length: 50 }).notNull(), // 'create', 'update', 'delete', 'view', 'download'
  actionDetails: text('action_details'),
  
  // Changes
  oldValues: jsonb('old_values'),
  newValues: jsonb('new_values'),
  changedFields: jsonb('changed_fields').default([]),
  
  // Context
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  requestId: varchar('request_id', { length: 255 }),
  
  // Timestamps
  createdAt: timestamp('created_at', { mode: 'date' }).notNull().defaultNow(),
}, (table) => ({
  userIdIdx: index('audit_logs_user_id_idx').on(table.userId),
  entityTypeIdx: index('audit_logs_entity_type_idx').on(table.entityType),
  entityIdIdx: index('audit_logs_entity_id_idx').on(table.entityId),
  actionIdx: index('audit_logs_action_idx').on(table.action),
  createdAtIdx: index('audit_logs_created_at_idx').on(table.createdAt),
}));

// ============================================================================
// DRIZZLE MIGRATION TABLE
// ============================================================================

export const __drizzle_migrations__ = pgTable('__drizzle_migrations__', {
  id: serial('id').primaryKey(),
  hash: varchar('hash', { length: 255 }).notNull(),
  createdAt: bigint('created_at', { mode: 'number' }),
});

// ============================================================================
// RELATIONS
// ============================================================================

export const usersRelations = relations(users, ({ many, one }) => ({
  sessions: many(sessions),
  userProfile: one(userProfiles, {
    fields: [users.id],
    references: [userProfiles.userId],
  }),
  keys: many(keys),
  cases: many(cases, { relationName: 'userCases' }),
  assignedCases: many(cases, { relationName: 'assignedAttorney' }),
  evidence: many(evidence),
  documents: many(legalDocuments),
  reports: many(reports),
  chatSessions: many(chatSessions),
  chatMessages: many(chatMessages),
  apiKeys: many(apiKeys),
  systemLogs: many(systemLogs),
  auditLogs: many(auditLogs),
}));

export const userProfilesRelations = relations(userProfiles, ({ one }) => ({
  user: one(users, {
    fields: [userProfiles.userId],
    references: [users.id],
  }),
}));

export const keysRelations = relations(keys, ({ one }) => ({
  user: one(users, {
    fields: [keys.userId],
    references: [users.id],
  }),
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.userId],
    references: [users.id],
  }),
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  user: one(users, {
    fields: [cases.userId],
    references: [users.id],
    relationName: 'userCases',
  }),
  assignedAttorney: one(users, {
    fields: [cases.assignedAttorney],
    references: [users.id],
    relationName: 'assignedAttorney',
  }),
  evidence: many(evidence),
  documents: many(legalDocuments),
  reports: many(reports),
  chatSessions: many(chatSessions),
  documentProcessing: many(documentProcessing),
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({
  case: one(cases, {
    fields: [evidence.caseId],
    references: [cases.id],
  }),
  user: one(users, {
    fields: [evidence.userId],
    references: [users.id],
  }),
}));

export const legalDocumentsRelations = relations(legalDocuments, ({ one, many }) => ({
  user: one(users, {
    fields: [legalDocuments.userId],
    references: [users.id],
  }),
  case: one(cases, {
    fields: [legalDocuments.caseId],
    references: [cases.id],
  }),
  processing: one(documentProcessing, {
    fields: [legalDocuments.id],
    references: [documentProcessing.documentId],
  }),
  chunks: many(documentChunks),
}));

export const documentProcessingRelations = relations(documentProcessing, ({ one }) => ({
  document: one(legalDocuments, {
    fields: [documentProcessing.documentId],
    references: [legalDocuments.id],
  }),
  case: one(cases, {
    fields: [documentProcessing.caseId],
    references: [cases.id],
  }),
}));

export const documentChunksRelations = relations(documentChunks, ({ one }) => ({
  document: one(legalDocuments, {
    fields: [documentChunks.documentId],
    references: [legalDocuments.id],
  }),
}));

export const reportsRelations = relations(reports, ({ one }) => ({
  user: one(users, {
    fields: [reports.userId],
    references: [users.id],
  }),
  case: one(cases, {
    fields: [reports.caseId],
    references: [cases.id],
  }),
}));

export const chatSessionsRelations = relations(chatSessions, ({ one, many }) => ({
  user: one(users, {
    fields: [chatSessions.userId],
    references: [users.id],
  }),
  case: one(cases, {
    fields: [chatSessions.caseId],
    references: [cases.id],
  }),
  messages: many(chatMessages),
}));

export const chatMessagesRelations = relations(chatMessages, ({ one }) => ({
  session: one(chatSessions, {
    fields: [chatMessages.sessionId],
    references: [chatSessions.id],
  }),
  user: one(users, {
    fields: [chatMessages.userId],
    references: [users.id],
  }),
}));

export const apiKeysRelations = relations(apiKeys, ({ one }) => ({
  user: one(users, {
    fields: [apiKeys.userId],
    references: [users.id],
  }),
}));

export const systemLogsRelations = relations(systemLogs, ({ one }) => ({
  user: one(users, {
    fields: [systemLogs.userId],
    references: [users.id],
  }),
}));

export const auditLogsRelations = relations(auditLogs, ({ one }) => ({
  user: one(users, {
    fields: [auditLogs.userId],
    references: [users.id],
  }),
}));

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type UserProfile = typeof userProfiles.$inferSelect;
export type NewUserProfile = typeof userProfiles.$inferInsert;
export type Key = typeof keys.$inferSelect;
export type NewKey = typeof keys.$inferInsert;
export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;
export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;
export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;
export type LegalDocument = typeof legalDocuments.$inferSelect;
export type NewLegalDocument = typeof legalDocuments.$inferInsert;
export type DocumentProcessing = typeof documentProcessing.$inferSelect;
export type NewDocumentProcessing = typeof documentProcessing.$inferInsert;
export type DocumentChunk = typeof documentChunks.$inferSelect;
export type NewDocumentChunk = typeof documentChunks.$inferInsert;
export type Report = typeof reports.$inferSelect;
export type NewReport = typeof reports.$inferInsert;
export type Vector = typeof vectors.$inferSelect;
export type NewVector = typeof vectors.$inferInsert;
export type VectorOperation = typeof vectorOperations.$inferSelect;
export type NewVectorOperation = typeof vectorOperations.$inferInsert;
export type ChatSession = typeof chatSessions.$inferSelect;
export type NewChatSession = typeof chatSessions.$inferInsert;
export type ChatMessage = typeof chatMessages.$inferSelect;
export type NewChatMessage = typeof chatMessages.$inferInsert;
export type ApiKey = typeof apiKeys.$inferSelect;
export type NewApiKey = typeof apiKeys.$inferInsert;
export type SystemLog = typeof systemLogs.$inferSelect;
export type NewSystemLog = typeof systemLogs.$inferInsert;
export type AuditLog = typeof auditLogs.$inferSelect;
export type NewAuditLog = typeof auditLogs.$inferInsert;

// ============================================================================
// HELPER FUNCTIONS FOR VECTOR OPERATIONS
// ============================================================================

export const getVectorSimilarityQuery = (
  table: any,
  embeddingColumn: any,
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