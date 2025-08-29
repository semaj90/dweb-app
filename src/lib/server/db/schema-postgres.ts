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
import { relations } from 'drizzle-orm';

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
  deleted_at: timestamp('deleted_at', { withTimezone: true, mode: 'date' }),
}, (table) => ({
  // Indexes matching database structure
  emailIdx: index('users_email_idx').on(table.email),
  usernameIdx: index('users_username_idx').on(table.username),
  roleIdx: index('users_role_idx').on(table.role),
  activeIdx: index('users_active_idx').on(table.is_active),
  profileEmbeddingIdx: index('users_profile_embedding_hnsw_idx').using('hnsw', table.profile_embedding.op('vector_cosine_ops')),
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
    mode: "date",
  }).notNull(),
  // Additional columns for enhanced session management
  ip_address: varchar("ip_address", { length: 45 }),
  user_agent: text("user_agent"),
  session_context: jsonb("session_context").default({}),
  created_at: timestamp("created_at", {
    withTimezone: true,
    mode: "date",
  }).defaultNow().notNull(),
}, (table) => ({
  // Indexes matching database structure (use snake_case keys)
  expires_at_idx: index('sessions_expires_at_idx').on(table.expires_at),
  user_id_idx: index('sessions_user_id_idx').on(table.user_id),
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
  userId: uuid('created_by').references(() => users.id), // alias for createdBy
  assignedTo: uuid('assigned_to').references(() => users.id),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_id: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }), // alias for compatibility
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content'), // evidence content
  description: text('description'),
  evidence_type: varchar('evidence_type', { length: 100 }).notNull(),
  evidenceType: varchar('evidence_type', { length: 100 }), // alias for evidence_type
  type: varchar('type', { length: 100 }), // alias for evidence_type
  createdBy: uuid('created_by').references(() => users.id),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const legal_documents = pgTable('legal_documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  document_type: varchar('document_type', { length: 100 }).notNull(),
  content: text('content'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const documentChunks = pgTable('document_chunks', {
  id: uuid('id').primaryKey().defaultRandom(),
  document_id: uuid('document_id').notNull(),
  document_type: varchar('document_type', { length: 100 }).default('evidence').notNull(),
  chunk_index: varchar('chunk_index', { length: 50 }).notNull(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 384 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
}, (table) => ({
  documentIdIdx: index('document_chunks_document_id_idx').on(table.document_id),
  embeddingIdx: index('document_chunks_embedding_hnsw_idx').using('hnsw', table.embedding.op('vector_cosine_ops')),
}));

// === RELATIONS ===
export const usersRelations = relations(users, ({ many }) => ({
  sessions: many(sessions),
  cases: many(cases),
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.user_id],
    references: [users.id],
  }),
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
    references: [users.id],
  }),
  evidence: many(evidence),
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({
  case: one(cases, {
    fields: [evidence.case_id],
    references: [cases.id],
  }),
}));

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
  status: varchar('status', { length: 50 }).default('draft').notNull(),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

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

export const vectorMetadata = pgTable('vector_metadata', {
  id: uuid('id').primaryKey().defaultRandom(),
  document_id: uuid('document_id').notNull(),
  vector_id: varchar('vector_id', { length: 255 }),
  embedding: vector('embedding', { dimensions: 384 }),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
}, (table) => ({
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
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull()
});

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
}, (table) => ({
  contentHashIdx: index('embedding_cache_content_hash_idx').on(table.content_hash),
  embeddingIdx: index('embedding_cache_embedding_hnsw_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

export type User = typeof users.$inferSelect;
export type Session = typeof sessions.$inferSelect;
export type DatabaseUserAttributes = Omit<User, 'id'>;
export type NewUserAiQuery = typeof userAiQueries.$inferInsert;
export type NewAutoTag = typeof autoTags.$inferInsert;
export type NewDocumentChunk = typeof documentChunks.$inferInsert;