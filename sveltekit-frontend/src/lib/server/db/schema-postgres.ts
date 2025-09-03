/**
 * PostgreSQL Schema with Exact Database Column Mapping
 * Aligned with actual database structure for Lucia v3 compatibility
 * Uses snake_case column names to match PostgreSQL conventions
 */

import {
  pgTable,
  text,
  uuid,
  timestamp,
  varchar,
  boolean,
  jsonb,
  index
} from 'drizzle-orm/pg-core';
import { vector } from 'pgvector/drizzle-orm';
import { relations } from 'drizzle-orm/relations';

// === USERS TABLE ===
// Maps exactly to PostgreSQL users table with snake_case columns
export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  hashed_password: varchar('hashed_password', { length: 255 }),
  username: varchar('username', { length: 100 }),
  first_name: varchar('first_name', { length: 100 }),
  last_name: varchar('last_name', { length: 100 }),
  role: varchar('role', { length: 50 }).default('user').notNull(),
  department: varchar('department', { length: 100 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  permissions: jsonb('permissions').default([]).notNull(),
  is_active: boolean('is_active').default(true).notNull(),
  email_verified: boolean('email_verified').default(false).notNull(),
  avatar_url: varchar('avatar_url', { length: 500 }),
  last_login_at: timestamp('last_login_at', { withTimezone: true, mode: 'date' }),
  practice_areas: jsonb('practice_areas').default([]),
  bar_number: varchar('bar_number', { length: 50 }),
  firm_name: varchar('firm_name', { length: 200 }),
  profile_embedding: vector('profile_embedding', { dimensions: 384 }),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  deleted_at: timestamp('deleted_at', { withTimezone: true, mode: 'date' })
}, (table: typeof users) => ({
  // Indexes matching database structure
  emailIdx: index('users_email_idx').on(table.email),
  usernameIdx: index('users_username_idx').on(table.username),
  roleIdx: index('users_role_idx').on(table.role),
  activeIdx: index('users_active_idx').on(table.is_active),
  profileEmbeddingIdx: index('users_profile_embedding_hnsw_idx').using('hnsw', table.profile_embedding.op('vector_cosine_ops'))
}));

// === SESSIONS TABLE ===
// Lucia v3 compatible sessions table with required column names
export const sessions = pgTable("sessions", {
  id: varchar("id", { length: 255 }).primaryKey(),
  user_id: uuid("user_id")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expires_at: timestamp("expires_at", {
    withTimezone: true,
    mode: "date"
  }).notNull(),
  // Additional columns for enhanced session management
  ip_address: varchar("ip_address", { length: 45 }),
  user_agent: text("user_agent"),
  session_context: jsonb("session_context").default({}),
  created_at: timestamp("created_at", {
    withTimezone: true,
    mode: "date"
  }).defaultNow().notNull()
}, (table: typeof sessions) => ({
  // Indexes matching database structure (use snake_case keys)
  expires_at_idx: index('sessions_expires_at_idx').on(table.expires_at),
  user_id_idx: index('sessions_user_id_idx').on(table.user_id)
}));

// === BASIC LEGAL TABLES ===
// Note: These are simplified versions that focus on the essential structure
// The actual database may have more tables that aren't critical for auth

export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseNumber: varchar('case_number', { length: 100 }),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  status: varchar('status', { length: 50 }).default('open').notNull(),
  priority: varchar('priority', { length: 20 }).default('medium').notNull(),
  assigned_attorney: uuid('assigned_attorney').references(() => users.id),
  createdBy: uuid('created_by').references(() => users.id),
  assignedTo: uuid('assigned_to').references(() => users.id),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_id: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  fileName: varchar('file_name', { length: 255 }),
  originalFileName: varchar('original_file_name', { length: 255 }),
  fileSize: varchar('file_size', { length: 50 }),
  fileType: varchar('file_type', { length: 100 }),
  filePath: varchar('file_path', { length: 500 }),
  evidence_type: varchar('evidence_type', { length: 100 }).notNull(),
  type: varchar('type', { length: 100 }), // separate field for general type classification
  createdBy: uuid('created_by').references(() => users.id),
  tags: jsonb('tags').default([]),
  metadata: jsonb('metadata').default({}),
  isPublic: boolean('is_public').default(false),
  ocrText: text('ocr_text'),
  contentText: text('content_text'),
  // OCR integration enhancement fields (nullable to avoid migration breakage if not yet applied)
  ocr_confidence: varchar('ocr_confidence', { length: 32 }),
  ocr_word_count: varchar('ocr_word_count', { length: 32 }),
  ocr_processing_time_ms: varchar('ocr_processing_time_ms', { length: 32 }),
  ocr_metadata: jsonb('ocr_metadata').default({}),
  embedding: vector('embedding', { dimensions: 384 }),
  uploadedAt: timestamp('uploaded_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  processedAt: timestamp('processed_at', { withTimezone: true, mode: 'date' }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table: typeof evidence) => ({
  caseIdIdx: index('evidence_case_id_idx').on(table.case_id),
  fileTypeIdx: index('evidence_file_type_idx').on(table.fileType),
  uploadedAtIdx: index('evidence_uploaded_at_idx').on(table.uploadedAt),
  embeddingIdx: index('evidence_embedding_hnsw_idx').using('hnsw', table.embedding.op('vector_cosine_ops')),
  tagsIdx: index('evidence_tags_gin_idx').using('gin', table.tags)
}));

export const legal_documents = pgTable('legal_documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  document_type: varchar('document_type', { length: 100 }).notNull(),
  content: text('content'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const documentChunks = pgTable('document_chunks', {
  id: uuid('id').primaryKey().defaultRandom(),
  document_id: uuid('document_id').notNull(),
  document_type: varchar('document_type', { length: 100 }).default('evidence').notNull(),
  chunk_index: varchar('chunk_index', { length: 50 }).notNull(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 384 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table: typeof documentChunks) => ({
  documentIdIdx: index('document_chunks_document_id_idx').on(table.document_id),
  embeddingIdx: index('document_chunks_embedding_hnsw_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

// === RELATIONS ===
export const usersRelations = relations(users, ({ many }) => ({
  sessions: many(sessions),
  cases: many(cases)
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.user_id],
    references: [users.id]
  })
}));

// === LUCIA KEYS TABLE ===
// For Lucia v3 adapter compatibility (snake_case columns)
export const keys = pgTable('keys', {
  id: varchar('id', { length: 255 }).primaryKey(),
  user_id: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  hashed_password: varchar('hashed_password', { length: 255 }),
  provider_id: varchar('provider_id', { length: 255 }),
  provider_user_id: varchar('provider_user_id', { length: 255 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const casesRelations = relations(cases, ({ one, many }) => ({
  assignedAttorney: one(users, {
    fields: [cases.assigned_attorney],
    references: [users.id]
  }),
  evidence: many(evidence)
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({
  case: one(cases, {
    fields: [evidence.case_id],
    references: [cases.id]
  })
}));

// Relations defined later to avoid circular dependencies

// Type exports for Lucia auth compatibility
// Additional missing tables that are referenced in errors
export const userProfiles = pgTable('user_profiles', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const reports = pgTable('reports', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content'),
  report_type: varchar('report_type', { length: 100 }).default('analysis').notNull(),
  status: varchar('status', { length: 50 }).default('draft').notNull(),
  case_id: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  created_by: uuid('created_by').references(() => users.id, { onDelete: 'cascade' }),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table: typeof reports) => ({
  createdByIdx: index('reports_created_by_idx').on(table.created_by),
  caseIdIdx: index('reports_case_id_idx').on(table.case_id),
  statusIdx: index('reports_status_idx').on(table.status)
}));

export const statutes = pgTable('statutes', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const legalAnalysisSessions = pgTable('legal_analysis_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  session_data: jsonb('session_data').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const userAiQueries = pgTable('user_ai_queries', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  query: text('query').notNull(),
  response: text('response'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const autoTags = pgTable('auto_tags', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 100 }).notNull(),
  category: varchar('category', { length: 50 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const caseScores = pgTable('case_scores', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_id: uuid('case_id').references(() => cases.id),
  score: text('score').notNull(),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const ragSessions = pgTable('rag_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  session_data: jsonb('session_data').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const ragMessages = pgTable('rag_messages', {
  id: uuid('id').primaryKey().defaultRandom(),
  session_id: uuid('session_id').references(() => ragSessions.id),
  message: text('message').notNull(),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

// Enhanced conversation tables for AI Assistant
export const conversations = pgTable('conversations', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  case_id: uuid('case_id').references(() => cases.id, { onDelete: 'set null' }),
  title: varchar('title', { length: 255 }).notNull(),
  context: jsonb('context').default({}),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  archived_at: timestamp('archived_at', { withTimezone: true, mode: 'date' })
}, (table: typeof conversations) => ({
  userIdIdx: index('conversations_user_id_idx').on(table.user_id),
  caseIdIdx: index('conversations_case_id_idx').on(table.case_id),
  createdAtIdx: index('conversations_created_at_idx').on(table.created_at)
}));

export const conversationMessages = pgTable('conversation_messages', {
  id: uuid('id').primaryKey().defaultRandom(),
  conversation_id: uuid('conversation_id').references(() => conversations.id, { onDelete: 'cascade' }),
  role: varchar('role', { length: 20 }).notNull(), // 'user' or 'assistant'
  content: text('content').notNull(),
  model: varchar('model', { length: 100 }),
  token_count: varchar('token_count', { length: 50 }),
  processing_time: varchar('processing_time', { length: 50 }),
  confidence: varchar('confidence', { length: 50 }),
  vector_search_results: jsonb('vector_search_results').default([]),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table: typeof conversationMessages) => ({
  conversationIdIdx: index('conversation_messages_conversation_id_idx').on(table.conversation_id),
  roleIdx: index('conversation_messages_role_idx').on(table.role),
  createdAtIdx: index('conversation_messages_created_at_idx').on(table.created_at)
}));

export const vectorMetadata = pgTable('vector_metadata', {
  id: uuid('id').primaryKey().defaultRandom(),
  document_id: uuid('document_id').notNull(),
  vector_id: varchar('vector_id', { length: 255 }),
  embedding: vector('embedding', { dimensions: 384 }),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table: typeof vectorMetadata) => ({
  documentIdIdx: index('vector_metadata_document_id_idx').on(table.document_id),
  embeddingIdx: index('vector_metadata_embedding_hnsw_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

export const criminals = pgTable('criminals', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  aliases: jsonb('aliases').default([]),
  description: text('description'),
  case_ids: jsonb('case_ids').default([]),
  risk_level: varchar('risk_level', { length: 50 }).default('medium'),
  status: varchar('status', { length: 50 }).default('active'),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const personsOfInterest = pgTable('persons_of_interest', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  aliases: jsonb('aliases').default([]),
  description: text('description'),
  case_ids: jsonb('case_ids').default([]),
  risk_level: varchar('risk_level', { length: 50 }).default('low'),
  status: varchar('status', { length: 50 }).default('active'),
  contact_info: jsonb('contact_info').default({}),
  created_by: uuid('created_by').references(() => users.id, { onDelete: 'cascade' }),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table: typeof personsOfInterest) => ({
  createdByIdx: index('persons_of_interest_created_by_idx').on(table.created_by),
  nameIdx: index('persons_of_interest_name_idx').on(table.name),
  riskLevelIdx: index('persons_of_interest_risk_level_idx').on(table.risk_level),
  statusIdx: index('persons_of_interest_status_idx').on(table.status)
}));

export const canvasStates = pgTable('canvas_states', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  case_id: uuid('case_id').references(() => cases.id),
  name: varchar('name', { length: 255 }),
  canvas_data: jsonb('canvas_data').default({}),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const embeddingCache = pgTable('embedding_cache', {
  id: uuid('id').primaryKey().defaultRandom(),
  content_hash: varchar('content_hash', { length: 255 }).notNull().unique(),
  embedding: vector('embedding', { dimensions: 384 }),
  model_name: varchar('model_name', { length: 100 }).notNull(),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  expires_at: timestamp('expires_at', { withTimezone: true, mode: 'date' })
}, (table: typeof ragMessages) => ({
  contentHashIdx: index('embedding_cache_content_hash_idx').on(table.content_hash),
  embeddingIdx: index('embedding_cache_embedding_hnsw_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

export type User = typeof users.$inferSelect;
export type Session = typeof sessions.$inferSelect;
export type DatabaseUserAttributes = Omit<User, 'id'>;
export type NewUserAiQuery = typeof userAiQueries.$inferInsert;
export type NewAutoTag = typeof autoTags.$inferInsert;
export type NewDocumentChunk = typeof documentChunks.$inferInsert;

// Compatibility type aliases (some files expect PascalCase type names)
export type Case = typeof cases.$inferSelect;
export type Evidence = typeof evidence.$inferSelect;
export type LegalDocument = typeof legal_documents.$inferSelect;
export type DocumentChunk = typeof documentChunks.$inferSelect;
export type Report = typeof reports.$inferSelect;
export type Statute = typeof statutes.$inferSelect;
export type LegalAnalysisSession = typeof legalAnalysisSessions.$inferSelect;
export type UserAiQuery = typeof userAiQueries.$inferSelect;
export type AutoTagType = typeof autoTags.$inferSelect;
export type CaseScoreType = typeof caseScores.$inferSelect;
export type RagSessionType = typeof ragSessions.$inferSelect;
export type RagMessageType = typeof ragMessages.$inferSelect;
export type Conversation = typeof conversations.$inferSelect;
export type ConversationMessage = typeof conversationMessages.$inferSelect;
export type NewConversation = typeof conversations.$inferInsert;
export type NewConversationMessage = typeof conversationMessages.$inferInsert;
export type VectorMetadataType = typeof vectorMetadata.$inferSelect;
export type Criminal = typeof criminals.$inferSelect;
export type PersonOfInterest = typeof personsOfInterest.$inferSelect;
export type CanvasState = typeof canvasStates.$inferSelect;
export type EmbeddingCache = typeof embeddingCache.$inferSelect;
export type UserProfile = typeof userProfiles.$inferSelect;

// Missing tables referenced in CRUD endpoints
export const caseDocuments = pgTable('case_documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }).notNull(),
  title: varchar('title', { length: 255 }).notNull(),
  document_type: varchar('document_type', { length: 100 }).notNull(),
  file_path: varchar('file_path', { length: 500 }),
  file_size: varchar('file_size', { length: 50 }),
  mime_type: varchar('mime_type', { length: 100 }),
  content: text('content'),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const caseActivities = pgTable('case_activities', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }).notNull(),
  type: varchar('type', { length: 100 }).notNull(),
  description: text('description').notNull(),
  userId: uuid('user_id').references(() => users.id),
  timestamp: timestamp('timestamp', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  metadata: jsonb('metadata').default({})
});

export const caseTimeline = pgTable('case_timeline', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }).notNull(),
  event: varchar('event', { length: 255 }).notNull(),
  description: text('description'),
  timestamp: timestamp('timestamp', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  type: varchar('type', { length: 50 }).default('event'),
  metadata: jsonb('metadata').default({})
});

// === FEEDBACK AND ANALYTICS TABLES ===
export const userRatings = pgTable('user_ratings', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  content_id: uuid('content_id'),
  content_type: varchar('content_type', { length: 100 }),
  rating: varchar('rating', { length: 20 }),
  feedback: text('feedback'),
  timestamp: timestamp('timestamp', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const interactionHistory = pgTable('interaction_history', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  action: varchar('action', { length: 255 }),
  context: jsonb('context').default({}),
  timestamp: timestamp('timestamp', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const trainingData = pgTable('training_data', {
  id: uuid('id').primaryKey().defaultRandom(),
  source_type: varchar('source_type', { length: 100 }),
  data: jsonb('data'),
  labels: jsonb('labels'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const userBehaviorPatterns = pgTable('user_behavior_patterns', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  pattern_type: varchar('pattern_type', { length: 100 }),
  pattern_data: jsonb('pattern_data'),
  confidence: varchar('confidence', { length: 20 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

export const feedbackMetrics = pgTable('feedback_metrics', {
  id: uuid('id').primaryKey().defaultRandom(),
  metric_name: varchar('metric_name', { length: 255 }),
  metric_value: varchar('metric_value', { length: 255 }),
  context: jsonb('context').default({}),
  timestamp: timestamp('timestamp', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

// Type exports for new tables
export type CaseDocument = typeof caseDocuments.$inferSelect;
export type CaseActivity = typeof caseActivities.$inferSelect;
export type CaseTimeline = typeof caseTimeline.$inferSelect;
export type UserRating = typeof userRatings.$inferSelect;
export type InteractionHistory = typeof interactionHistory.$inferSelect;
export type TrainingData = typeof trainingData.$inferSelect;
export type UserBehaviorPattern = typeof userBehaviorPatterns.$inferSelect;
export type FeedbackMetric = typeof feedbackMetrics.$inferSelect;

// New type exports for Drizzle inserts
export type NewUserRating = typeof userRatings.$inferInsert;
export type NewInteractionHistory = typeof interactionHistory.$inferInsert;
export type NewTrainingData = typeof trainingData.$inferInsert;
export type NewUserBehaviorPattern = typeof userBehaviorPatterns.$inferInsert;
export type NewFeedbackMetric = typeof feedbackMetrics.$inferInsert;

// Compatibility named exports / aliases for varied import patterns used in the codebase
export const legalDocuments = legal_documents;
export const document_chunks = documentChunks;
export const documents = legal_documents;
export const persons_of_interest = personsOfInterest;
export const embedding_cache = embeddingCache;

// Conversation relations (defined at end to avoid circular dependencies)
export const conversationsRelations = relations(conversations, ({ one, many }) => ({
  user: one(users, {
    fields: [conversations.user_id],
    references: [users.id]
  }),
  case: one(cases, {
    fields: [conversations.case_id],
    references: [cases.id]
  }),
  messages: many(conversationMessages)
}));

export const conversationMessagesRelations = relations(conversationMessages, ({ one }) => ({
  conversation: one(conversations, {
    fields: [conversationMessages.conversation_id],
    references: [conversations.id]
  })
}));
