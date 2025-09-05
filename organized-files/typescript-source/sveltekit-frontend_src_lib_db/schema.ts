// @ts-nocheck
// Main Drizzle Schema - Legal AI Case Management System
// This is the central schema file that Drizzle expects

import {
  pgTable,
  text,
  timestamp,
  uuid,
  integer,
  jsonb,
  boolean,
  decimal,
  varchar,
} from "drizzle-orm/pg-core";
import { vector } from "pgvector/drizzle-orm"; // pgvector support
import { relations } from "drizzle-orm";

// Users table
export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  email: text("email").notNull().unique(),
  name: text("name").notNull(),
  role: text("role").notNull().default("user"), // admin, prosecutor, detective, user
  passwordHash: text("password_hash").notNull(),
  lastLogin: timestamp("last_login"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Cases table
export const cases = pgTable("cases", {
  id: uuid("id").primaryKey().defaultRandom(),
  title: text("title").notNull(),
  description: text("description"),
  status: text("status").notNull().default("active"), // active, closed, archived
  priority: text("priority").default("medium"), // low, medium, high, urgent
  assignedTo: uuid("assigned_to").references(() => users.id),
  createdBy: uuid("created_by")
    .references(() => users.id)
    .notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
  metadata: jsonb("metadata"),
});

// Evidence table
export const evidence = pgTable("evidence", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id")
    .references(() => cases.id)
    .notNull(),
  title: text("title").notNull(),
  description: text("description"),
  type: text("type").notNull(), // document, image, video, audio, physical, digital
  content: text("content"), // Extracted text content
  filePath: text("file_path"), // Path to uploaded file
  fileSize: integer("file_size"), // File size in bytes
  mimeType: text("mime_type"), // MIME type of file
  hash: text("hash"), // File hash for integrity
  tags: jsonb("tags"), // AI-generated tags
  summary: text("summary"), // AI-generated summary
  embedding: text("embedding"), // Vector embeddings as text
  createdBy: uuid("created_by")
    .references(() => users.id)
    .notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
  metadata: jsonb("metadata"), // Additional metadata (tags, analysis results, etc.)
});

// Documents table (for AI-powered document analysis)
export const documents = pgTable("documents", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id),
  evidenceId: uuid("evidence_id").references(() => evidence.id),
  filename: text("filename").notNull(),
  filePath: text("file_path").notNull(),
  extractedText: text("extracted_text"),
  embeddings: text("embeddings"), // Vector embeddings stored as text for similarity search
  analysis: jsonb("analysis"), // AI analysis results
  createdBy: uuid("created_by")
    .references(() => users.id)
    .notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Notes table
export const notes = pgTable("notes", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id")
    .references(() => cases.id)
    .notNull(),
  evidenceId: uuid("evidence_id").references(() => evidence.id),
  content: text("content").notNull(),
  isPrivate: boolean("is_private").default(false),
  createdBy: uuid("created_by")
    .references(() => users.id)
    .notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// AI History table (for tracking AI interactions)
export const aiHistory = pgTable("ai_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id),
  userId: uuid("user_id")
    .references(() => users.id)
    .notNull(),
  prompt: text("prompt").notNull(),
  response: text("response").notNull(),
  model: text("model").notNull(),
  tokensUsed: integer("tokens_used"),
  cost: integer("cost"), // Cost in cents
  createdAt: timestamp("created_at").defaultNow().notNull(),
  metadata: jsonb("metadata"),
});

// Collaboration sessions (for real-time collaboration)
export const collaborationSessions = pgTable("collaboration_sessions", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id")
    .references(() => cases.id)
    .notNull(),
  userId: uuid("user_id")
    .references(() => users.id)
    .notNull(),
  sessionId: text("session_id").notNull(),
  isActive: boolean("is_active").default(true),
  lastActivity: timestamp("last_activity").defaultNow().notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Legal documents (for legal AI features)
export const legalDocuments = pgTable("legal_documents", {
  id: uuid("id").primaryKey().defaultRandom(),
  title: varchar("title", { length: 500 }).notNull(),
  documentType: varchar("document_type", { length: 50 }).notNull(), // statute, regulation, case_law, precedent, contract
  jurisdiction: varchar("jurisdiction", { length: 100 }),
  court: varchar("court", { length: 200 }),
  citation: varchar("citation", { length: 300 }),
  fullCitation: text("full_citation"),
  fullText: text("full_text"),
  content: text("content"), // Main document content for search/display
  summary: text("summary"),
  // Aligned to 384 to match nomic-embed-text dimensions
  embedding: vector("embedding", { dimensions: 384 }),
  keywords: jsonb("keywords").default([]),
  topics: jsonb("topics").default([]),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  evidenceId: uuid("evidence_id").references(() => evidence.id, { onDelete: "cascade" }),
  isActive: boolean("is_active").default(true),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Document sections for RAG chunking
export const documentSections = pgTable("document_sections", {
  id: uuid("id").primaryKey().defaultRandom(),
  documentId: uuid("document_id").notNull().references(() => legalDocuments.id, { onDelete: "cascade" }),
  sectionNumber: integer("section_number").notNull(),
  title: varchar("title", { length: 500 }),
  content: text("content").notNull(),
  metadata: jsonb("metadata").default({}),
  // Aligned to 384 dimensions to match nomic-embed-text model
  embedding: vector("embedding", { dimensions: 384 }),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Citations for legal research
export const citations = pgTable("citations", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  documentId: uuid("document_id").references(() => legalDocuments.id, { onDelete: "cascade" }),
  citationType: varchar("citation_type", { length: 50 }).notNull(), // primary_authority, secondary_authority, supporting
  relevanceScore: decimal("relevance_score", { precision: 3, scale: 2 }), // 0.00 - 1.00
  pageNumber: integer("page_number"),
  quotedText: text("quoted_text"),
  annotation: text("annotation"),
  formattedCitation: text("formatted_citation"),
  isKeyAuthority: boolean("is_key_authority").default(false),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Reports for case management
export const reports = pgTable("reports", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  title: varchar("title", { length: 255 }).notNull(),
  content: text("content"),
  reportType: varchar("report_type", { length: 50 }).default("case_summary"),
  status: varchar("status", { length: 20 }).default("draft"),
  isPublic: boolean("is_public").default(false),
  tags: jsonb("tags").default([]),
  metadata: jsonb("metadata").default({}),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// RAG (Retrieval Augmented Generation) sessions
export const ragSessions = pgTable("rag_sessions", {
  id: uuid("id").primaryKey().defaultRandom(),
  sessionId: varchar("session_id", { length: 255 }).notNull().unique(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  title: varchar("title", { length: 255 }),
  model: varchar("model", { length: 100 }),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// RAG messages
export const ragMessages = pgTable("rag_messages", {
  id: uuid("id").primaryKey().defaultRandom(),
  sessionId: varchar("session_id", { length: 255 }).notNull(),
  messageIndex: integer("message_index").notNull(),
  role: varchar("role", { length: 20 }).notNull(), // 'user', 'assistant'
  content: text("content").notNull(),
  retrievedSources: jsonb("retrieved_sources").default([]),
  sourceCount: integer("source_count").default(0),
  retrievalScore: varchar("retrieval_score", { length: 10 }),
  processingTime: integer("processing_time"),
  model: varchar("model", { length: 100 }),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Relations
export const usersRelations = relations(users, ({ many }) => ({
  createdCases: many(cases, { relationName: "case_creator" }),
  assignedCases: many(cases, { relationName: "case_assignee" }),
  evidence: many(evidence),
  notes: many(notes),
  aiHistory: many(aiHistory),
  documents: many(documents),
  collaborationSessions: many(collaborationSessions),
  legalDocuments: many(legalDocuments),
  citations: many(citations),
  reports: many(reports),
  ragSessions: many(ragSessions),
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  creator: one(users, {
    fields: [cases.createdBy],
    references: [users.id],
    relationName: "case_creator",
  }),
  assignee: one(users, {
    fields: [cases.assignedTo],
    references: [users.id],
    relationName: "case_assignee",
  }),
  evidence: many(evidence),
  notes: many(notes),
  documents: many(documents),
  collaborationSessions: many(collaborationSessions),
  legalDocuments: many(legalDocuments),
  citations: many(citations),
  reports: many(reports),
}));

export const evidenceRelations = relations(evidence, ({ one, many }) => ({
  case: one(cases, {
    fields: [evidence.caseId],
    references: [cases.id],
  }),
  creator: one(users, {
    fields: [evidence.createdBy],
    references: [users.id],
  }),
  documents: many(documents),
  notes: many(notes),
}));

export const documentsRelations = relations(documents, ({ one }) => ({
  case: one(cases, {
    fields: [documents.caseId],
    references: [cases.id],
  }),
  evidence: one(evidence, {
    fields: [documents.evidenceId],
    references: [evidence.id],
  }),
  creator: one(users, {
    fields: [documents.createdBy],
    references: [users.id],
  }),
}));

export const notesRelations = relations(notes, ({ one }) => ({
  case: one(cases, {
    fields: [notes.caseId],
    references: [cases.id],
  }),
  evidence: one(evidence, {
    fields: [notes.evidenceId],
    references: [evidence.id],
  }),
  creator: one(users, {
    fields: [notes.createdBy],
    references: [users.id],
  }),
}));

export const aiHistoryRelations = relations(aiHistory, ({ one }) => ({
  case: one(cases, {
    fields: [aiHistory.caseId],
    references: [cases.id],
  }),
  user: one(users, {
    fields: [aiHistory.userId],
    references: [users.id],
  }),
}));

export const collaborationSessionsRelations = relations(
  collaborationSessions,
  ({ one }) => ({
    case: one(cases, {
      fields: [collaborationSessions.caseId],
      references: [cases.id],
    }),
    user: one(users, {
      fields: [collaborationSessions.userId],
      references: [users.id],
    }),
  }),
);

// Legal documents relations
export const legalDocumentsRelations = relations(legalDocuments, ({ one, many }) => ({
  case: one(cases, {
    fields: [legalDocuments.caseId],
    references: [cases.id],
  }),
  evidence: one(evidence, {
    fields: [legalDocuments.evidenceId],
    references: [evidence.id],
  }),
  createdBy: one(users, {
    fields: [legalDocuments.createdBy],
    references: [users.id],
  }),
  documentSections: many(documentSections),
  citations: many(citations),
}));

// Document sections relations
export const documentSectionsRelations = relations(documentSections, ({ one }) => ({
  document: one(legalDocuments, {
    fields: [documentSections.documentId],
    references: [legalDocuments.id],
  }),
}));

// Citations relations
export const citationsRelations = relations(citations, ({ one }) => ({
  case: one(cases, {
    fields: [citations.caseId],
    references: [cases.id],
  }),
  document: one(legalDocuments, {
    fields: [citations.documentId],
    references: [legalDocuments.id],
  }),
  createdBy: one(users, {
    fields: [citations.createdBy],
    references: [users.id],
  }),
}));

// Reports relations
export const reportsRelations = relations(reports, ({ one }) => ({
  case: one(cases, {
    fields: [reports.caseId],
    references: [cases.id],
  }),
  createdBy: one(users, {
    fields: [reports.createdBy],
    references: [users.id],
  }),
}));

// RAG sessions relations
export const ragSessionsRelations = relations(ragSessions, ({ one, many }) => ({
  user: one(users, {
    fields: [ragSessions.userId],
    references: [users.id],
  }),
  messages: many(ragMessages),
}));

// RAG messages relations
export const ragMessagesRelations = relations(ragMessages, ({ one }) => ({
  session: one(ragSessions, {
    fields: [ragMessages.sessionId],
    references: [ragSessions.sessionId],
  }),
}));

// Chat sessions for Enhanced AI Chat
export const chatSessions = pgTable('chat_sessions', {
  id: uuid('id').primaryKey(),
  model: text('model').notNull().default('gemma3-legal'),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  messageCount: integer('message_count').default(0).notNull(),
  isActive: boolean('is_active').default(true).notNull()
});

// Chat messages for Enhanced AI Chat
export const chatMessages = pgTable('chat_messages', {
  id: uuid('id').primaryKey(),
  sessionId: uuid('session_id').references(() => chatSessions.id).notNull(),
  content: text('content').notNull(),
  role: text('role').notNull(), // 'user' | 'assistant' | 'system'
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  embedding: text('embedding'), // JSON string of embedding vector for pgvector
  metadata: jsonb('metadata').default({}),
  model: text('model'),
  confidence: decimal('confidence', { precision: 5, scale: 4 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull()
});

// Export all vector tables and types
export * from "./schema/vectors";

// Export types for TypeScript
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;
export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;
export type Document = typeof documents.$inferSelect;
export type NewDocument = typeof documents.$inferInsert;
export type Note = typeof notes.$inferSelect;
export type NewNote = typeof notes.$inferInsert;
export type AIHistory = typeof aiHistory.$inferSelect;
export type NewAIHistory = typeof aiHistory.$inferInsert;
export type CollaborationSession = typeof collaborationSessions.$inferSelect;
export type NewCollaborationSession = typeof collaborationSessions.$inferInsert;
export type ChatSession = typeof chatSessions.$inferSelect;
export type NewChatSession = typeof chatSessions.$inferInsert;
export type ChatMessage = typeof chatMessages.$inferSelect;
export type NewChatMessage = typeof chatMessages.$inferInsert;
