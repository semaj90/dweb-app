// PostgreSQL schema for production
import { relations } from "drizzle-orm";
import {
  boolean,
  decimal,
  integer,
  jsonb,
  pgTable,
  serial,
  text,
  timestamp,
  uuid,
  varchar,
} from "drizzle-orm/pg-core";

// === AUTHENTICATION & USER MANAGEMENT ===

export const users = pgTable("users", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  emailVerified: timestamp("email_verified", { mode: "date" }),
  hashedPassword: text("hashed_password"),
  name: text("name"),
  firstName: varchar("first_name", { length: 100 }),
  lastName: varchar("last_name", { length: 100 }),
  avatarUrl: text("avatar_url"),
  role: varchar("role", { length: 50 }).default("prosecutor").notNull(),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === LUCIA v3 AUTHENTICATION TABLES ===

export const sessions = pgTable("sessions", {
  id: text("id").primaryKey(),
  userId: uuid("user_id")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expiresAt: timestamp("expires_at", {
    withTimezone: true,
    mode: "date",
  }).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

export const emailVerificationCodes = pgTable("email_verification_codes", {
  id: serial("id").primaryKey(),
  userId: uuid("user_id")
    .notNull()
    .unique()
    .references(() => users.id, { onDelete: "cascade" }),
  email: varchar("email", { length: 255 }).notNull(),
  code: varchar("code", { length: 8 }).notNull(),
  expiresAt: timestamp("expires_at", {
    withTimezone: true,
    mode: "date",
  }).notNull(),
});

export const passwordResetTokens = pgTable("password_reset_tokens", {
  tokenHash: varchar("token_hash", { length: 63 }).primaryKey(),
  userId: uuid("user_id")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expiresAt: timestamp("expires_at", {
    withTimezone: true,
    mode: "date",
  }).notNull(),
});

// === CASE MANAGEMENT ===

export const cases = pgTable("cases", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseNumber: varchar("case_number", { length: 50 }).notNull().unique(),
  title: varchar("title", { length: 255 }).notNull(),
  name: varchar("name", { length: 255 }),
  description: text("description"),
  incidentDate: timestamp("incident_date", { mode: "date" }),
  location: text("location"),
  priority: varchar("priority", { length: 20 }).default("medium").notNull(),
  status: varchar("status", { length: 20 }).default("open").notNull(),
  category: varchar("category", { length: 50 }),
  dangerScore: integer("danger_score").default(0).notNull(),
  estimatedValue: decimal("estimated_value", { precision: 12, scale: 2 }),
  jurisdiction: varchar("jurisdiction", { length: 100 }),
  leadProsecutor: uuid("lead_prosecutor"),
  assignedTeam: jsonb("assigned_team").default([]).notNull(),
  tags: jsonb("tags").default([]).notNull(),
  aiSummary: text("ai_summary"),
  aiTags: jsonb("ai_tags").default([]).notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdBy: uuid("created_by"),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
  closedAt: timestamp("closed_at", { mode: "date" }),
});

// === CRIMINAL RECORDS ===

export const criminals = pgTable("criminals", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  firstName: varchar("first_name", { length: 100 }).notNull(),
  lastName: varchar("last_name", { length: 100 }).notNull(),
  middleName: varchar("middle_name", { length: 100 }),
  aliases: jsonb("aliases").default([]).notNull(),
  dateOfBirth: timestamp("date_of_birth", { mode: "date" }),
  placeOfBirth: varchar("place_of_birth", { length: 200 }),
  address: text("address"),
  phone: varchar("phone", { length: 20 }),
  email: varchar("email", { length: 255 }),
  socialSecurityNumber: varchar("ssn", { length: 11 }),
  driversLicense: varchar("drivers_license", { length: 50 }),
  height: integer("height"),
  weight: integer("weight"),
  eyeColor: varchar("eye_color", { length: 20 }),
  hairColor: varchar("hair_color", { length: 20 }),
  distinguishingMarks: text("distinguishing_marks"),
  photoUrl: text("photo_url"),
  fingerprints: jsonb("fingerprints").default({}),
  threatLevel: varchar("threat_level", { length: 20 }).default("low").notNull(),
  status: varchar("status", { length: 20 }).default("active").notNull(),
  notes: text("notes"),
  aiSummary: text("ai_summary"),
  aiTags: jsonb("ai_tags").default([]).notNull(),
  createdBy: uuid("created_by"),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === EVIDENCE MANAGEMENT ===

export const evidence = pgTable("evidence", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id"),
  criminalId: uuid("criminal_id"),
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  evidenceType: varchar("evidence_type", { length: 50 }).notNull(),
  fileType: varchar("file_type", { length: 50 }),
  subType: varchar("sub_type", { length: 50 }),
  fileUrl: text("file_url"),
  fileName: varchar("file_name", { length: 255 }),
  fileSize: integer("file_size"),
  mimeType: varchar("mime_type", { length: 100 }),
  hash: varchar("hash", { length: 128 }),
  tags: jsonb("tags").default([]).notNull(),
  chainOfCustody: jsonb("chain_of_custody").default([]).notNull(),
  collectedAt: timestamp("collected_at", { mode: "date" }),
  collectedBy: varchar("collected_by", { length: 255 }),
  location: text("location"),
  labAnalysis: jsonb("lab_analysis").default({}).notNull(),
  aiAnalysis: jsonb("ai_analysis").default({}).notNull(),
  aiTags: jsonb("ai_tags").default([]).notNull(),
  aiSummary: text("ai_summary"),
  summary: text("summary"),
  isAdmissible: boolean("is_admissible").default(true).notNull(),
  confidentialityLevel: varchar("confidentiality_level", { length: 20 })
    .default("standard")
    .notNull(),
  canvasPosition: jsonb("canvas_position").default({}).notNull(),
  uploadedBy: uuid("uploaded_by"),
  uploadedAt: timestamp("uploaded_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === CASE ACTIVITIES & TIMELINE ===

export const caseActivities = pgTable("case_activities", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id").notNull(),
  activityType: varchar("activity_type", { length: 50 }).notNull(),
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  scheduledFor: timestamp("scheduled_for", { mode: "date" }),
  completedAt: timestamp("completed_at", { mode: "date" }),
  status: varchar("status", { length: 20 }).default("pending").notNull(),
  priority: varchar("priority", { length: 20 }).default("medium").notNull(),
  assignedTo: uuid("assigned_to"),
  relatedEvidence: jsonb("related_evidence").default([]).notNull(),
  relatedCriminals: jsonb("related_criminals").default([]).notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdBy: uuid("created_by"),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === ATTACHMENT VERIFICATIONS ===

export const attachmentVerifications = pgTable("attachment_verifications", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  attachmentId: uuid("attachment_id").notNull(),
  verifiedBy: uuid("verified_by")
    .notNull()
    .references(() => users.id),
  verificationStatus: varchar("verification_status", { length: 50 })
    .default("pending")
    .notNull(),
  verificationNotes: text("verification_notes"),
  verifiedAt: timestamp("verified_at", { mode: "date" }).defaultNow().notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === THEMES & UI CUSTOMIZATION ===

export const themes = pgTable("themes", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  name: varchar("name", { length: 100 }).notNull(),
  description: text("description"),
  cssVariables: jsonb("css_variables").notNull(),
  colorPalette: jsonb("color_palette").notNull(),
  isSystem: boolean("is_system").default(false).notNull(),
  isPublic: boolean("is_public").default(false).notNull(),
  createdBy: uuid("created_by").references(() => users.id, {
    onDelete: "cascade",
  }),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === VECTOR EMBEDDINGS FOR AI SEARCH ===

export const contentEmbeddings = pgTable("content_embeddings", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  contentId: uuid("content_id").notNull(),
  contentType: varchar("content_type", { length: 50 }).notNull(), // 'case', 'evidence', 'criminal'
  textContent: text("text_content").notNull(),
  embedding: text("embedding"), // JSON array of numbers for vector
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === RAG (Retrieval Augmented Generation) SESSIONS ===

export const ragSessions = pgTable("rag_sessions", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  sessionId: varchar("session_id", { length: 255 }).notNull().unique(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  title: varchar("title", { length: 255 }),
  model: varchar("model", { length: 100 }),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const ragMessages = pgTable("rag_messages", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  sessionId: varchar("session_id", { length: 255 }).notNull(),
  messageIndex: integer("message_index").notNull(),
  role: varchar("role", { length: 20 }).notNull(), // 'user', 'assistant'
  content: text("content").notNull(),
  retrievedSources: jsonb("retrieved_sources").default([]).notNull(),
  sourceCount: integer("source_count").default(0).notNull(),
  retrievalScore: varchar("retrieval_score", { length: 10 }),
  processingTime: integer("processing_time"),
  model: varchar("model", { length: 100 }),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

// === REPORTS & CITATIONS ===

export const reports = pgTable("reports", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  title: varchar("title", { length: 255 }).notNull(),
  content: text("content"),
  reportType: varchar("report_type", { length: 50 }).default("case_summary"),
  status: varchar("status", { length: 20 }).default("draft"),
  isPublic: boolean("is_public").default(false),
  tags: jsonb("tags").default([]).notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const canvasStates = pgTable("canvas_states", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  name: varchar("name", { length: 255 }).notNull(),
  canvasData: jsonb("canvas_data").notNull(),
  version: integer("version").default(1),
  isDefault: boolean("is_default").default(false),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const personsOfInterest = pgTable("persons_of_interest", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  name: varchar("name", { length: 255 }).notNull(),
  aliases: jsonb("aliases").default([]).notNull(),
  relationship: varchar("relationship", { length: 100 }),
  threatLevel: varchar("threat_level", { length: 20 }).default("low"),
  status: varchar("status", { length: 20 }).default("active"),
  profileData: jsonb("profile_data").default({}).notNull(),
  tags: jsonb("tags").default([]).notNull(),
  position: jsonb("position").default({}).notNull(),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const hashVerifications = pgTable("hash_verifications", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  evidenceId: uuid("evidence_id").references(() => evidence.id, {
    onDelete: "cascade",
  }),
  verifiedHash: varchar("verified_hash", { length: 64 }).notNull(),
  storedHash: varchar("stored_hash", { length: 64 }),
  result: boolean("result").notNull(),
  verificationMethod: varchar("verification_method", { length: 50 }).default(
    "manual",
  ),
  verifiedBy: uuid("verified_by").references(() => users.id),
  verifiedAt: timestamp("verified_at", { mode: "date" }).defaultNow(),
  notes: text("notes"),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

// === AI REPORTS & ANALYSIS ===

export const aiReports = pgTable("ai_reports", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  reportType: varchar("report_type", { length: 50 }).notNull(), // case_overview, evidence_analysis, timeline_summary, prosecution_strategy
  title: varchar("title", { length: 255 }).notNull(),
  content: text("content").notNull(),
  richTextContent: jsonb("rich_text_content"), // Tiptap JSON content
  metadata: jsonb("metadata").default({}).notNull(), // confidence, keyPoints, recommendations, etc.
  canvasElements: jsonb("canvas_elements").default([]).notNull(),
  generatedBy: varchar("generated_by", { length: 100 }).default("gemma3-legal"),
  confidence: decimal("confidence", { precision: 3, scale: 2 }).default("0.85"),
  isActive: boolean("is_active").default(true),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const statutes = pgTable("statutes", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  title: varchar("title", { length: 255 }).notNull(),
  code: varchar("code", { length: 100 }).notNull(),
  description: text("description"),
  category: varchar("category", { length: 100 }),
  jurisdiction: varchar("jurisdiction", { length: 100 }),
  isActive: boolean("is_active").default(true),
  penalties: jsonb("penalties").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === CONTEXT7 LEGAL AI ENHANCEMENT TABLES ===

export const legalDocuments = pgTable("legal_documents", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom(),
  content: text("content").notNull(),
  embedding: text("embedding"), // 1536-dimensional vector for legal embeddings
  legalCategory: varchar("legal_category", { length: 50 }),
  caseReference: varchar("case_reference", { length: 100 }),
  jurisdiction: varchar("jurisdiction", { length: 50 }),
  documentType: varchar("document_type", { length: 50 }), // statute, precedent, brief, motion
  confidence: decimal("confidence", { precision: 3, scale: 2 }).default("0.85"),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  evidenceId: uuid("evidence_id").references(() => evidence.id, { onDelete: "cascade" }),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const legalPrecedents = pgTable("legal_precedents", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom(),
  caseTitle: varchar("case_title", { length: 255 }).notNull(),
  citation: varchar("citation", { length: 255 }).notNull(),
  court: varchar("court", { length: 100 }),
  year: integer("year"),
  jurisdiction: varchar("jurisdiction", { length: 50 }),
  summary: text("summary"),
  fullText: text("full_text"),
  embedding: text("embedding"), // Legal document vector
  relevanceScore: decimal("relevance_score", { precision: 3, scale: 2 }),
  legalPrinciples: jsonb("legal_principles").default([]).notNull(),
  linkedCases: jsonb("linked_cases").default([]).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

export const legalAnalysisSessions = pgTable("legal_analysis_sessions", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  sessionType: varchar("session_type", { length: 50 }).default("case_analysis"),
  analysisPrompt: text("analysis_prompt"),
  analysisResult: text("analysis_result"),
  confidenceLevel: decimal("confidence_level", { precision: 3, scale: 2 }),
  sourcesUsed: jsonb("sources_used").default([]).notNull(),
  model: varchar("model", { length: 100 }).default("gemma3-legal"),
  processingTime: integer("processing_time"), // milliseconds
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull(),
});

// === VECTOR SEARCH TABLES ===

export const userEmbeddings = pgTable("user_embeddings", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  content: text("content").notNull(),
  embedding: text("embedding").notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

export const chatEmbeddings = pgTable("chat_embeddings", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  conversationId: uuid("conversation_id").notNull(),
  messageId: uuid("message_id").notNull(),
  content: text("content").notNull(),
  embedding: text("embedding").notNull(),
  role: varchar("role", { length: 20 }).notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

export const evidenceVectors = pgTable("evidence_vectors", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  evidenceId: uuid("evidence_id").references(() => evidence.id, {
    onDelete: "cascade",
  }),
  content: text("content").notNull(),
  embedding: text("embedding").notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

export const caseEmbeddings = pgTable("case_embeddings", {
  id: uuid("id")
    .primaryKey()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom()
    .defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  content: text("content").notNull(),
  embedding: text("embedding").notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
});

// === ADDITIONAL RELATIONS ===

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

export const canvasStatesRelations = relations(canvasStates, ({ one }) => ({
  case: one(cases, {
    fields: [canvasStates.caseId],
    references: [cases.id],
  }),
  createdBy: one(users, {
    fields: [canvasStates.createdBy],
    references: [users.id],
  }),
}));

export const aiReportsRelations = relations(aiReports, ({ one }) => ({
  case: one(cases, {
    fields: [aiReports.caseId],
    references: [cases.id],
  }),
  createdBy: one(users, {
    fields: [aiReports.createdBy],
    references: [users.id],
  }),
}));

export const personsOfInterestRelations = relations(
  personsOfInterest,
  ({ one }) => ({
    case: one(cases, {
      fields: [personsOfInterest.caseId],
      references: [cases.id],
    }),
    createdBy: one(users, {
      fields: [personsOfInterest.createdBy],
      references: [users.id],
    }),
  }),
);

export const hashVerificationsRelations = relations(
  hashVerifications,
  ({ one }) => ({
    evidence: one(evidence, {
      fields: [hashVerifications.evidenceId],
      references: [evidence.id],
    }),
    verifiedBy: one(users, {
      fields: [hashVerifications.verifiedBy],
      references: [users.id],
    }),
  }),
);

export const userEmbeddingsRelations = relations(userEmbeddings, ({ one }) => ({
  user: one(users, {
    fields: [userEmbeddings.userId],
    references: [users.id],
  }),
}));

export const chatEmbeddingsRelations = relations(chatEmbeddings, ({ one }) => ({
  // No direct relations as conversationId may not have a table
}));

export const evidenceVectorsRelations = relations(
  evidenceVectors,
  ({ one }) => ({
    evidence: one(evidence, {
      fields: [evidenceVectors.evidenceId],
      references: [evidence.id],
    }),
  }),
);

export const caseEmbeddingsRelations = relations(caseEmbeddings, ({ one }) => ({
  case: one(cases, {
    fields: [caseEmbeddings.caseId],
    references: [cases.id],
  }),
}));

// === CONTEXT7 LEGAL AI RELATIONS ===

export const legalDocumentsRelations = relations(legalDocuments, ({ one }) => ({
  case: one(cases, {
    fields: [legalDocuments.caseId],
    references: [cases.id],
  }),
  evidence: one(evidence, {
    fields: [legalDocuments.evidenceId],
    references: [evidence.id],
  }),
}));

export const legalPrecedentsRelations = relations(legalPrecedents, ({ many }) => ({
  // No direct foreign key relations but used via vector similarity
}));

export const legalAnalysisSessionsRelations = relations(legalAnalysisSessions, ({ one }) => ({
  case: one(cases, {
    fields: [legalAnalysisSessions.caseId],
    references: [cases.id],
  }),
  user: one(users, {
    fields: [legalAnalysisSessions.userId],
    references: [users.id],
  }),
}));

// === RELATIONSHIPS ===

export const usersRelations = relations(users, ({ many }) => ({
  casesAsLead: many(cases, { relationName: "leadProsecutor" }),
  casesCreated: many(cases, { relationName: "createdBy" }),
  evidenceUploaded: many(evidence),
  activitiesAssigned: many(caseActivities, { relationName: "assignedTo" }),
  activitiesCreated: many(caseActivities, { relationName: "createdBy" }),
  criminalsCreated: many(criminals),
  sessions: many(sessions),
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.userId],
    references: [users.id],
  }),
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  leadProsecutor: one(users, {
    fields: [cases.leadProsecutor],
    references: [users.id],
    relationName: "leadProsecutor",
  }),
  createdBy: one(users, {
    fields: [cases.createdBy],
    references: [users.id],
    relationName: "createdBy",
  }),
  evidence: many(evidence),
  activities: many(caseActivities),
}));

export const criminalsRelations = relations(criminals, ({ one, many }) => ({
  createdBy: one(users, {
    fields: [criminals.createdBy],
    references: [users.id],
  }),
  evidence: many(evidence),
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({
  uploadedBy: one(users, {
    fields: [evidence.uploadedBy],
    references: [users.id],
  }),
  case: one(cases, {
    fields: [evidence.caseId],
    references: [cases.id],
  }),
}));

export const caseActivitiesRelations = relations(caseActivities, ({ one }) => ({
  case: one(cases, {
    fields: [caseActivities.caseId],
    references: [cases.id],
  }),
  assignedTo: one(users, {
    fields: [caseActivities.assignedTo],
    references: [users.id],
    relationName: "assignedTo",
  }),
  createdBy: one(users, {
    fields: [caseActivities.createdBy],
    references: [users.id],
    relationName: "createdBy",
  }),
}));

export const attachmentVerificationsRelations = relations(
  attachmentVerifications,
  ({ one }) => ({
    verifiedBy: one(users, {
      fields: [attachmentVerifications.verifiedBy],
      references: [users.id],
    }),
  }),
);

export const themesRelations = relations(themes, ({ one }) => ({
  createdBy: one(users, {
    fields: [themes.createdBy],
    references: [users.id],
  }),
}));

export const contentEmbeddingsRelations = relations(
  contentEmbeddings,
  ({ one }) => ({
    content: one(cases, {
      fields: [contentEmbeddings.contentId],
      references: [cases.id],
    }),
  }),
);

export const ragSessionsRelations = relations(ragSessions, ({ one }) => ({
  user: one(users, {
    fields: [ragSessions.userId],
    references: [users.id],
  }),
}));

export const ragMessagesRelations = relations(ragMessages, ({ one }) => ({
  session: one(ragSessions, {
    fields: [ragMessages.sessionId],
    references: [ragSessions.sessionId],
  }),
}));
