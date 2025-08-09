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
} from "drizzle-orm/pg-core";
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

// Relations
export const usersRelations = relations(users, ({ many }) => ({
  createdCases: many(cases, { relationName: "case_creator" }),
  assignedCases: many(cases, { relationName: "case_assignee" }),
  evidence: many(evidence),
  notes: many(notes),
  aiHistory: many(aiHistory),
  documents: many(documents),
  collaborationSessions: many(collaborationSessions),
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
