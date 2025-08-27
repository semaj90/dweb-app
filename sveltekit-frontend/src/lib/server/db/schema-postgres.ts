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
  userId: uuid("user_id") // Maps to user_id column but named userId for Lucia
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expiresAt: timestamp("expires_at", { // Maps to expires_at column but named expiresAt for Lucia
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
  // Indexes matching database structure
  expiresAtIdx: index('sessions_expires_at_idx').on(table.expiresAt),
  userIdIdx: index('sessions_user_id_idx').on(table.userId),
}));

// === BASIC LEGAL TABLES ===
// Note: These are simplified versions that focus on the essential structure
// The actual database may have more tables that aren't critical for auth

export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  status: varchar('status', { length: 50 }).default('open').notNull(),
  assigned_attorney: uuid('assigned_attorney').references(() => users.id),
  created_at: timestamp('created_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
  updated_at: timestamp('updated_at', { withTimezone: true, mode: 'date' }).defaultNow().notNull(),
});

export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().defaultRandom(),
  case_id: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  evidence_type: varchar('evidence_type', { length: 100 }).notNull(),
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
    fields: [sessions.userId],
    references: [users.id],
  }),
}));

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