/**
 * PostgreSQL Schema with Exact Database Column Mapping
 * Aligned with actual database structure for Lucia v3 compatibility
 * Uses snake_case column names to match PostgreSQL conventions
 */

let pgTable: any;
let text: any;
let uuid: any;
let timestamp: any;
let varchar: any;
let boolean: any;
let jsonb: any;
let index: any;
let vector: (name: string, opts: { dimensions: number }) => any;

try {
  // Try to load the real pg-core from drizzle if available at runtime
  // (works in CommonJS environments). If using ESM-only runtime, replace
  // this with an appropriate dynamic import.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const core = require('drizzle-orm/pg-core');
  pgTable = core.pgTable;
  text = core.text;
  uuid = core.uuid;
  timestamp = core.timestamp;
  varchar = core.varchar;
  boolean = core.boolean;
  jsonb = core.jsonb;
  index = core.index;
  // pgvector helper may or may not be provided by the package/environment
  vector = core.vector ?? ((_name: string, _opts: { dimensions: number }) => ({} as any));
} catch (e) {
  // Fallback stubs to allow type-checking and editing when drizzle-orm or pgvector
  // are not installed; these are minimal no-op implementations.
  pgTable = (..._args: any[]) => ({} as any);
  text = (_name: string) => ({} as any);
  uuid = (_name: string) => ({} as any);
  timestamp = (_name: string, _opts?: any) => ({} as any);
  varchar = (_name: string, _opts?: any) => ({} as any);
  boolean = (_name: string) => ({} as any);
  jsonb = (_name: string) => ({} as any);
  index = (_name: string) => ({ on: (_: any) => ({} as any) });
  vector = (_name: string, _opts: { dimensions: number }) => ({} as any);
}

import { relations } from 'drizzle-orm';

// === USERS TABLE ===
export const users = pgTable(
  'users',
  {
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
  }
);

// === SESSIONS TABLE ===
// Lucia v3 compatible sessions table with required column names
export const sessions = pgTable('sessions', {
  id: varchar('id', { length: 255 }).primaryKey(),
  user_id: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expires_at: timestamp('expires_at', { withTimezone: true, mode: 'date' }).notNull(),
  // Additional columns for enhanced session management
  ip_address: varchar('ip_address', { length: 45 }),
  user_agent: text('user_agent'),
  session_context: jsonb('session_context').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === CASES TABLE ===
export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_number: varchar('case_number', { length: 100 }),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  status: varchar('status', { length: 50 }).default('open').notNull(),
  priority: varchar('priority', { length: 20 }).default('medium').notNull(),
  assigned_attorney: uuid('assigned_attorney').references(() => users.id),
  created_by: uuid('created_by').references(() => users.id),
  assigned_to: uuid('assigned_to').references(() => users.id),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === EVIDENCE TABLE ===
export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_id: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content'),
  description: text('description'),
  evidence_type: varchar('evidence_type', { length: 100 }).notNull(),
  created_by: uuid('created_by').references(() => users.id),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === LEGAL DOCUMENTS ===
export const legal_documents = pgTable('legal_documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  document_type: varchar('document_type', { length: 100 }).notNull(),
  content: text('content'),
  content_embedding: vector('content_embedding', { dimensions: 384 }),
  jurisdiction: text('jurisdiction'),
  analysis_results: jsonb('analysis_results'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === DOCUMENT CHUNKS ===
export const document_chunks = pgTable('document_chunks', {
  id: uuid('id').primaryKey().defaultRandom(),
  document_id: uuid('document_id').notNull(),
  document_type: varchar('document_type', { length: 100 }).default('evidence').notNull(),
  chunk_index: varchar('chunk_index', { length: 50 }).notNull(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 384 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === LUCIA KEYS TABLE ===
export const keys = pgTable('keys', {
  id: varchar('id', { length: 255 }).primaryKey(),
  user_id: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  hashed_password: varchar('hashed_password', { length: 255 }),
  provider_id: varchar('provider_id', { length: 255 }),
  provider_user_id: varchar('provider_user_id', { length: 255 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === VECTOR METADATA ===
export const vector_metadata = pgTable('vector_metadata', {
  id: uuid('id').primaryKey().defaultRandom(),
  document_id: uuid('document_id').notNull(),
  vector_id: varchar('vector_id', { length: 255 }),
  embedding: vector('embedding', { dimensions: 384 }),
  risk_level: varchar('risk_level', { length: 50 }).default('low'),
  status: varchar('status', { length: 50 }).default('active'),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === CANVAS STATES ===
export const canvas_states = pgTable('canvas_states', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  case_id: uuid('case_id').references(() => cases.id),
  name: varchar('name', { length: 255 }),
  canvas_data: jsonb('canvas_data').default({}),
  metadata: jsonb('metadata').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === EMBEDDING CACHE ===
export const embedding_cache = pgTable(
  'embedding_cache',
  {
    id: uuid('id').primaryKey().defaultRandom(),
    content_hash: varchar('content_hash', { length: 255 }).notNull().unique(),
    embedding: vector('embedding', { dimensions: 384 }),
    model_name: varchar('model_name', { length: 100 }).notNull(),
    metadata: jsonb('metadata').default({}),
    created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
    expires_at: timestamp('expires_at', { withTimezone: true, mode: 'date' }),
  },
  (table) => ({
    contentHashIdx: index('embedding_cache_content_hash_idx').on(table.content_hash),
  })
);

export const statutes = pgTable('statutes', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const legal_analysis_sessions = pgTable('legal_analysis_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  session_data: jsonb('session_data').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const user_ai_queries = pgTable('user_ai_queries', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  query: text('query').notNull(),
  response: text('response'),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const auto_tags = pgTable('auto_tags', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 100 }).notNull(),
  category: varchar('category', { length: 50 }),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const case_scores = pgTable('case_scores', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_id: uuid('case_id').references(() => cases.id),
  score: text('score').notNull(),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const rag_sessions = pgTable('rag_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  user_id: uuid('user_id').references(() => users.id),
  session_data: jsonb('session_data').default({}),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const rag_messages = pgTable('rag_messages', {
  id: uuid('id').primaryKey().defaultRandom(),
  session_id: uuid('session_id').references(() => rag_sessions.id),
  message: text('message').notNull(),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

// === RELATIONS ===
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

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.user_id],
    references: [users.id],
  }),
}));

// === TYPE EXPORTS ===
export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;

export type LegalDocument = typeof legal_documents.$inferSelect;
export type NewLegalDocument = typeof legal_documents.$inferInsert;

export type DocumentChunk = typeof document_chunks.$inferSelect;
export type NewDocumentChunk = typeof document_chunks.$inferInsert;

export type VectorMetadata = typeof vector_metadata.$inferSelect;
export type NewVectorMetadata = typeof vector_metadata.$inferInsert;

export type EmbeddingCache = typeof embedding_cache.$inferSelect;
export type NewEmbeddingCache = typeof embedding_cache.$inferInsert;

export type Statute = typeof statutes.$inferSelect;
export type NewStatute = typeof statutes.$inferInsert;

export type LegalAnalysisSession = typeof legal_analysis_sessions.$inferSelect;
export type NewLegalAnalysisSession = typeof legal_analysis_sessions.$inferInsert;

export type UserAiQuery = typeof user_ai_queries.$inferSelect;
export type NewUserAiQuery = typeof user_ai_queries.$inferInsert;

export type AutoTag = typeof auto_tags.$inferSelect;
export type NewAutoTag = typeof auto_tags.$inferInsert;

export type CaseScore = typeof case_scores.$inferSelect;
export type NewCaseScore = typeof case_scores.$inferInsert;

export type RagSession = typeof rag_sessions.$inferSelect;
export type NewRagSession = typeof rag_sessions.$inferInsert;

export type RagMessage = typeof rag_messages.$inferSelect;
export type NewRagMessage = typeof rag_messages.$inferInsert;

export type CanvasState = typeof canvas_states.$inferSelect;
export type NewCanvasState = typeof canvas_states.$inferInsert;

export type Key = typeof keys.$inferSelect;
export type NewKey = typeof keys.$inferInsert;

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;

export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;
