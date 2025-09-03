
// DEPRECATION WARNING:
// This legacy camelCase schema is retained temporarily for non-auth code paths.
// Do NOT import from '$lib/database/schema' for authentication/session logic.
// Use '$lib/server/db/schema-postgres' instead. A runtime guard below logs when
// auth-critical tables are imported from this module.
import { pgTable, text, timestamp, integer, boolean, json, uuid, varchar, vector } from "drizzle-orm/pg-core";
import { sql } from 'drizzle-orm';

// Users table with enhanced authentication fields
export const users = pgTable('users', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  email: varchar('email', { length: 255 }).notNull().unique(),
  passwordHash: text('password_hash').notNull(),
  firstName: varchar('first_name', { length: 100 }),
  lastName: varchar('last_name', { length: 100 }),
  role: varchar('role', { length: 50 }).notNull().default('user'), // admin, prosecutor, detective, user
  isActive: boolean('is_active').default(true),
  emailVerified: timestamp('email_verified'),
  emailVerificationToken: varchar('email_verification_token', { length: 255 }),
  passwordResetToken: varchar('password_reset_token', { length: 255 }),
  passwordResetExpires: timestamp('password_reset_expires'),
  lastLoginAt: timestamp('last_login_at'),
  loginAttempts: integer('login_attempts').default(0),
  lockoutUntil: timestamp('lockout_until'),
  twoFactorSecret: text('two_factor_secret'),
  twoFactorEnabled: boolean('two_factor_enabled').default(false),
  profilePicture: text('profile_picture'),
  preferences: json('preferences').default(sql`'{}'::json`),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Sessions table for Lucia v3 compatibility
export const sessions = pgTable('sessions', {
  id: text('id').primaryKey(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: timestamp('expires_at', { withTimezone: true, mode: 'date' }).notNull(),
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  createdAt: timestamp('created_at').defaultNow()
});

// User audit logs for security tracking
export const userAuditLogs = pgTable('user_audit_logs', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  userId: uuid('user_id').references(() => users.id),
  action: varchar('action', { length: 100 }).notNull(), // login, logout, password_change, profile_update, etc.
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow()
});

// Cases table
export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  status: varchar('status', { length: 50 }).default('active'), // active, closed, archived
  priority: varchar('priority', { length: 20 }).default('medium'), // low, medium, high, critical
  caseNumber: varchar('case_number', { length: 100 }).unique(),
  createdBy: uuid('created_by').references(() => users.id),
  assignedTo: uuid('assigned_to').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Documents table with vector embeddings
export const documents = pgTable('documents', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  caseId: uuid('case_id').references(() => cases.id),
  title: varchar('title', { length: 255 }).notNull(),
  filename: varchar('filename', { length: 255 }),
  fileType: varchar('file_type', { length: 50 }),
  fileSize: integer('file_size'),
  content: text('content'), // Extracted text content
  extractedText: text('extracted_text'), // Full extracted text for search
  embedding: vector('embedding', { dimensions: 1536 }), // OpenAI ada-002 or similar
  metadata: json('metadata'),
  tags: json('tags').default(sql`'[]'::json`),
  isIndexed: boolean('is_indexed').default(false),
  source: varchar('source', { length: 100 }).default('upload'), // upload, scan, email, etc.
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Evidence table
export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  caseId: uuid('case_id').references(() => cases.id),
  documentId: uuid('document_id').references(() => documents.id),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  evidenceType: varchar('evidence_type', { length: 50 }), // document, image, video, audio, physical
  hash: varchar('hash', { length: 256 }), // File integrity hash
  chainOfCustody: json('chain_of_custody').default(sql`'[]'::json`),
  isAdmissible: boolean('is_admissible'),
  admissibilityNotes: text('admissibility_notes'),
  tags: json('tags').default(sql`'[]'::json`),
  aiAnalysis: json('ai_analysis'),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// AI chat history and interactions
export const aiInteractions = pgTable('ai_interactions', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  userId: uuid('user_id').references(() => users.id),
  caseId: uuid('case_id').references(() => cases.id),
  sessionId: varchar('session_id', { length: 255 }),
  prompt: text('prompt').notNull(),
  response: text('response').notNull(),
  model: varchar('model', { length: 100 }),
  tokensUsed: integer('tokens_used'),
  responseTime: integer('response_time'), // milliseconds
  confidence: integer('confidence'), // 0-100
  feedback: json('feedback'),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow()
});

// Search index for semantic search
export const searchIndex = pgTable('search_index', {
  id: uuid('id').primaryKey().default(sql`gen_random_uuid()`),
  entityType: varchar('entity_type', { length: 50 }).notNull(), // document, case, evidence, etc.
  entityId: uuid('entity_id').notNull(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 1536 }),
  metadata: json('metadata'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Relations for type safety
export const relations = {
  users: {
    sessions: sessions,
    auditLogs: userAuditLogs,
    createdCases: cases,
    assignedCases: cases,
    createdDocuments: documents,
    createdEvidence: evidence,
    aiInteractions: aiInteractions
  },
  sessions: {
    user: users
  },
  userAuditLogs: {
    user: users
  },
  cases: {
    creator: users,
    assignee: users,
    documents: documents,
    evidence: evidence,
    aiInteractions: aiInteractions
  },
  documents: {
    case: cases,
    creator: users,
    evidence: evidence
  },
  evidence: {
    case: cases,
    document: documents,
    creator: users
  },
  aiInteractions: {
    user: users,
    case: cases
  }
};

// Runtime guard: flag unintended server-side auth imports of legacy schema
const gAny = globalThis as any;
if (!gAny.__legacy_schema_warned) {
  gAny.__legacy_schema_warned = true;
  console.log('[LEGACY-SCHEMA] Loaded legacy $lib/database/schema (avoid for auth)');
}

// Type exports for use in application
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;

export type UserAuditLog = typeof userAuditLogs.$inferSelect;
export type NewUserAuditLog = typeof userAuditLogs.$inferInsert;

export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;

export type Document = typeof documents.$inferSelect;
export type NewDocument = typeof documents.$inferInsert;

export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;

export type AIInteraction = typeof aiInteractions.$inferSelect;
export type NewAIInteraction = typeof aiInteractions.$inferInsert;

export type SearchIndex = typeof searchIndex.$inferSelect;
export type NewSearchIndex = typeof searchIndex.$inferInsert;

// Re-export missing tables from additional-tables.ts
export { embeddingCache } from '../server/db/additional-tables';

// Database connection re-export { db } from '../server/db/index';