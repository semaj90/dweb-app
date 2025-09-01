/**
 * Enhanced Drizzle Schema with pgvector Support
 * PostgreSQL + pgvector integration for YoRHa Legal AI Platform
 */

import { pgTable, text, serial, timestamp, integer, vector, uuid, boolean, jsonb, numeric } from "drizzle-orm/pg-core";
import { relations } from 'drizzle-orm';
import { createSelectSchema, createUpdateSchema, createInsertSchema } from 'drizzle-zod';
import { z } from 'zod';

// Users table with enhanced authentication
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  email: text("email").notNull().unique(),
  password_hash: text("password_hash").notNull(),
  created_at: timestamp("created_at").defaultNow().notNull(),
  updated_at: timestamp("updated_at").defaultNow().notNull()
});

// Documents table with proper pgvector support (1536 dimensions for OpenAI)
export const documents = pgTable("documents", {
  id: text("id").primaryKey(), // UUID as text for flexibility
  filename: text("filename").notNull(),
  content: text("content").notNull(),
  summary: text("summary"),
  embedding: vector("embedding", { dimensions: 1536 }), // OpenAI embeddings
  user_id: integer("user_id").references(() => users.id).notNull(),
  created_at: timestamp("created_at").defaultNow().notNull(),
  updated_at: timestamp("updated_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default('{}')
});

// Legal cases table (enhanced from existing schema)
export const cases = pgTable("cases", {
  id: uuid("id").primaryKey().defaultRandom(),
  title: text("title").notNull(),
  description: text("description"),
  status: text("status").default('open'),
  priority: text("priority").default('medium'),
  user_id: integer("user_id").references(() => users.id).notNull(),
  case_embedding: vector("case_embedding", { dimensions: 384 }), // nomic-embed-text
  created_at: timestamp("created_at").defaultNow().notNull(),
  updated_at: timestamp("updated_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default('{}')
});

// Evidence table with embeddings
export const evidence = pgTable("evidence", {
  id: uuid("id").primaryKey().defaultRandom(),
  case_id: uuid("case_id").references(() => cases.id, { onDelete: 'cascade' }).notNull(),
  user_id: integer("user_id").references(() => users.id).notNull(),
  title: text("title").notNull(),
  description: text("description"),
  content_text: text("content_text"),
  file_path: text("file_path"),
  file_type: text("file_type"),
  file_size: integer("file_size"),
  embedding: vector("embedding", { dimensions: 384 }), // nomic-embed-text
  tags: text("tags").array(),
  created_at: timestamp("created_at").defaultNow().notNull(),
  updated_at: timestamp("updated_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default('{}')
});

// Document chunks for RAG (chunked documents with embeddings)
export const document_chunks = pgTable("document_chunks", {
  id: uuid("id").primaryKey().defaultRandom(),
  document_id: text("document_id").references(() => documents.id, { onDelete: 'cascade' }),
  evidence_id: uuid("evidence_id").references(() => evidence.id, { onDelete: 'cascade' }),
  chunk_index: integer("chunk_index").notNull(),
  chunk_text: text("chunk_text").notNull(),
  embedding: vector("embedding", { dimensions: 384 }).notNull(), // nomic-embed-text
  token_count: integer("token_count"),
  created_at: timestamp("created_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default('{}')
});

// Citations table (fixed schema with proper foreign keys)
export const citations = pgTable("citations", {
  id: uuid("id").primaryKey().defaultRandom(),
  case_id: uuid("case_id").references(() => cases.id, { onDelete: 'cascade' }),
  document_id: text("document_id").references(() => documents.id, { onDelete: 'cascade' }),
  citation_text: text("citation_text").notNull(),
  citation_type: text("citation_type"),
  source: text("source"),
  page_number: integer("page_number"),
  relevance_score: numeric("relevance_score", { precision: 3, scale: 2 }),
  context: text("context"),
  verified: boolean("verified").default(false),
  created_at: timestamp("created_at").defaultNow().notNull(),
  updated_at: timestamp("updated_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default('{}')
});

// Sessions table for authentication
export const sessions = pgTable("sessions", {
  id: text("id").primaryKey(),
  user_id: integer("user_id").references(() => users.id, { onDelete: 'cascade' }).notNull(),
  expires_at: timestamp("expires_at", { withTimezone: true }).notNull(),
  created_at: timestamp("created_at").defaultNow().notNull()
});

// AI History table for agent interactions
export const aiHistory = pgTable("ai_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  user_id: integer("user_id").references(() => users.id, { onDelete: 'cascade' }).notNull(),
  agent_type: text("agent_type").notNull(),
  interaction_type: text("interaction_type").notNull(),
  prompt: text("prompt").notNull(),
  response: text("response").notNull(),
  model_used: text("model_used"),
  tokens_used: integer("tokens_used"),
  created_at: timestamp("created_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default('{}')
});

// Relations for better query experience
export const usersRelations = relations(users, ({ many }) => ({
  documents: many(documents),
  cases: many(cases),
  evidence: many(evidence),
  sessions: many(sessions),
  aiHistory: many(aiHistory)
}));

export const documentsRelations = relations(documents, ({ one, many }) => ({
  user: one(users, {
    fields: [documents.user_id],
    references: [users.id]
  }),
  chunks: many(document_chunks),
  citations: many(citations)
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  user: one(users, {
    fields: [cases.user_id],
    references: [users.id]
  }),
  evidence: many(evidence),
  citations: many(citations)
}));

export const evidenceRelations = relations(evidence, ({ one, many }) => ({
  case: one(cases, {
    fields: [evidence.case_id],
    references: [cases.id]
  }),
  user: one(users, {
    fields: [evidence.user_id],
    references: [users.id]
  }),
  chunks: many(document_chunks)
}));

export const documentChunksRelations = relations(document_chunks, ({ one }) => ({
  document: one(documents, {
    fields: [document_chunks.document_id],
    references: [documents.id]
  }),
  evidence: one(evidence, {
    fields: [document_chunks.evidence_id],
    references: [evidence.id]
  })
}));

export const citationsRelations = relations(citations, ({ one }) => ({
  case: one(cases, {
    fields: [citations.case_id],
    references: [cases.id]
  }),
  document: one(documents, {
    fields: [citations.document_id],
    references: [documents.id]
  })
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.user_id],
    references: [users.id]
  })
}));

export const aiHistoryRelations = relations(aiHistory, ({ one }) => ({
  user: one(users, {
    fields: [aiHistory.user_id],
    references: [users.id]
  })
}));

// Type exports for TypeScript
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

export type Document = typeof documents.$inferSelect;
export type NewDocument = typeof documents.$inferInsert;

export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;

export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;

export type DocumentChunk = typeof document_chunks.$inferSelect;
export type NewDocumentChunk = typeof document_chunks.$inferInsert;

export type Citation = typeof citations.$inferSelect;
export type NewCitation = typeof citations.$inferInsert;

export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;

export type AiHistory = typeof aiHistory.$inferSelect;
export type NewAiHistory = typeof aiHistory.$inferInsert;

// Profile table for user profiles
export const profileTable = pgTable('profile', {
  id: uuid('id').primaryKey(),
  firstName: text('first_name').notNull(),
  lastName: text('last_name').notNull()
});

// Profile relations
export const profileRelations = relations(profileTable, ({ one }) => ({
  user: one(users, {
    fields: [profileTable.id],
    references: [users.id]
  })
}));

// Profile types
export type Profile = typeof profileTable.$inferSelect;
export type NewProfile = typeof profileTable.$inferInsert;

// Drizzle-zod schemas for SuperForms compatibility
export const profileTableSelectSchema = createSelectSchema(profileTable);
export const profileTableUpdateSchema = createUpdateSchema(profileTable);
export const profileTableInsertSchema = createInsertSchema(profileTable);
;
// Additional schemas for other tables (optional - add as needed)
export const usersSelectSchema = createSelectSchema(users);
export const usersUpdateSchema = createUpdateSchema(users);
export const usersInsertSchema = createInsertSchema(users);
;
export const casesSelectSchema = createSelectSchema(cases);
export const casesUpdateSchema = createUpdateSchema(cases);
export const casesInsertSchema = createInsertSchema(cases);
;
// Helper function to extract Zod schema from drizzle-zod BuildSchema for SuperForms compatibility
export function extractZodSchema<T extends { _def: { schema: any } }>(drizzleZodSchema: T) {
  return drizzleZodSchema._def.schema;
}

// Pre-extracted schemas for common use with SuperForms
export const profileUpdateZodSchema = extractZodSchema(profileTableUpdateSchema);
export const profileInsertZodSchema = extractZodSchema(profileTableInsertSchema);
export const profileSelectZodSchema = extractZodSchema(profileTableSelectSchema);
;
export const usersUpdateZodSchema = extractZodSchema(usersUpdateSchema);
export const usersInsertZodSchema = extractZodSchema(usersInsertSchema);
export const usersSelectZodSchema = extractZodSchema(usersSelectSchema);
;
export const casesUpdateZodSchema = extractZodSchema(casesUpdateSchema);
export const casesInsertZodSchema = extractZodSchema(casesInsertSchema);
export const casesSelectZodSchema = extractZodSchema(casesSelectSchema);
;