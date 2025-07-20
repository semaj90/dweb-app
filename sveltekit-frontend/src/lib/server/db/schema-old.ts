// This file re-exports everything from the unified schema for backward compatibility
export * from "$lib/server/db/schema-postgres";

import { cases } from "$lib/server/db/schema-postgres";
import {
  boolean,
  foreignKey,
  integer,
  jsonb,
  numeric,
  pgTable,
  primaryKey,
  text,
  timestamp,
  unique,
  uuid,
  varchar,
} from "drizzle-orm/pg-core";

export const caseLawLinks = pgTable(
  "case_law_links",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    caseId: uuid("case_id").notNull(),
    statuteId: uuid("statute_id"),
    lawParagraphId: uuid("law_paragraph_id"),
    linkType: varchar("link_type", { length: 50 }).notNull(),
    description: text(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "case_law_links_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.statuteId],
      foreignColumns: [statutes.id],
      name: "case_law_links_statute_id_statutes_id_fk",
    }),
    foreignKey({
      columns: [table.lawParagraphId],
      foreignColumns: [lawParagraphs.id],
      name: "case_law_links_law_paragraph_id_law_paragraphs_id_fk",
    }),
  ],
);

export const lawParagraphs = pgTable(
  "law_paragraphs",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    statuteId: uuid("statute_id").notNull(),
    paragraphNumber: varchar("paragraph_number", { length: 50 }).notNull(),
    content: text().notNull(),
    aiSummary: text("ai_summary"),
    tags: jsonb().default([]).notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    paragraphText: text("paragraph_text"),
    anchorId: varchar("anchor_id", { length: 100 }),
    linkedCaseIds: jsonb("linked_case_ids").default([]).notNull(),
    crimeSuggestions: jsonb("crime_suggestions").default([]).notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.statuteId],
      foreignColumns: [statutes.id],
      name: "law_paragraphs_statute_id_statutes_id_fk",
    }).onDelete("cascade"),
  ],
);

export const crimes = pgTable(
  "crimes",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    caseId: uuid("case_id"),
    criminalId: uuid("criminal_id"),
    statuteId: uuid("statute_id"),
    name: varchar({ length: 255 }).notNull(),
    description: text(),
    chargeLevel: varchar("charge_level", { length: 50 }),
    status: varchar({ length: 50 }).default("pending").notNull(),
    incidentDate: timestamp("incident_date", { mode: "string" }),
    arrestDate: timestamp("arrest_date", { mode: "string" }),
    filingDate: timestamp("filing_date", { mode: "string" }),
    notes: text(),
    aiSummary: text("ai_summary"),
    metadata: jsonb().default({}).notNull(),
    createdBy: uuid("created_by"),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "crimes_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.criminalId],
      foreignColumns: [criminals.id],
      name: "crimes_criminal_id_criminals_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.statuteId],
      foreignColumns: [statutes.id],
      name: "crimes_statute_id_statutes_id_fk",
    }),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "crimes_created_by_users_id_fk",
    }),
  ],
);

export const contentEmbeddings = pgTable("content_embeddings", {
  id: uuid().defaultRandom().primaryKey().notNull(),
  entityType: varchar("entity_type", { length: 50 }).notNull(),
  entityId: uuid("entity_id").notNull(),
  contentType: varchar("content_type", { length: 50 }).notNull(),
  embedding: jsonb().notNull(),
  text: text().notNull(),
  createdAt: timestamp("created_at", { mode: "string" }).defaultNow().notNull(),
});

export const criminals = pgTable(
  "criminals",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    firstName: varchar("first_name", { length: 100 }).notNull(),
    lastName: varchar("last_name", { length: 100 }).notNull(),
    middleName: varchar("middle_name", { length: 100 }),
    aliases: jsonb().default([]).notNull(),
    dateOfBirth: timestamp("date_of_birth", { mode: "string" }),
    address: text(),
    phone: varchar({ length: 20 }),
    email: varchar({ length: 255 }),
    height: integer(),
    weight: integer(),
    eyeColor: varchar("eye_color", { length: 20 }),
    hairColor: varchar("hair_color", { length: 20 }),
    distinguishingMarks: text("distinguishing_marks"),
    photoUrl: text("photo_url"),
    threatLevel: varchar("threat_level", { length: 20 })
      .default("low")
      .notNull(),
    status: varchar({ length: 20 }).default("active").notNull(),
    priors: jsonb().default([]).notNull(),
    convictions: jsonb().default([]).notNull(),
    notes: text(),
    aiSummary: text("ai_summary"),
    aiTags: jsonb("ai_tags").default([]).notNull(),
    aiAnalysis: jsonb("ai_analysis").default({}).notNull(),
    createdBy: uuid("created_by"),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    name: varchar({ length: 255 }),
    convictionStatus: varchar("conviction_status", { length: 50 }),
    sentenceLength: varchar("sentence_length", { length: 100 }),
    convictionDate: timestamp("conviction_date", { mode: "string" }),
    escapeAttempts: integer("escape_attempts").default(0),
    gangAffiliations: text("gang_affiliations"),
  },
  (table) => [
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "criminals_created_by_users_id_fk",
    }),
  ],
);

export const evidence = pgTable(
  "evidence",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    caseId: uuid("case_id"),
    criminalId: uuid("criminal_id"),
    title: varchar({ length: 255 }).notNull(),
    description: text(),
    fileUrl: text("file_url"),
    fileType: varchar("file_type", { length: 100 }),
    fileSize: integer("file_size"),
    tags: jsonb().default([]).notNull(),
    uploadedBy: uuid("uploaded_by"),
    uploadedAt: timestamp("uploaded_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    fileName: varchar("file_name", { length: 255 }),
    hash: varchar("hash", { length: 64 }), // SHA256 hash for file integrity
    summary: text(),
    aiSummary: text("ai_summary"),
  },
  (table) => [
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "evidence_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.criminalId],
      foreignColumns: [criminals.id],
      name: "evidence_criminal_id_criminals_id_fk",
    }),
    foreignKey({
      columns: [table.uploadedBy],
      foreignColumns: [users.id],
      name: "evidence_uploaded_by_users_id_fk",
    }),
  ],
);

export const sessions = pgTable(
  "sessions",
  {
    id: text().primaryKey().notNull(),
    userId: uuid("user_id").notNull(),
    expiresAt: timestamp("expires_at", { mode: "string" }).notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.userId],
      foreignColumns: [users.id],
      name: "sessions_user_id_users_id_fk",
    }).onDelete("cascade"),
  ],
);

export const statutes = pgTable(
  "statutes",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    code: varchar({ length: 50 }).notNull(),
    title: varchar({ length: 255 }).notNull(),
    description: text(),
    fullText: text("full_text"),
    category: varchar({ length: 100 }),
    severity: varchar({ length: 20 }),
    minPenalty: varchar("min_penalty", { length: 255 }),
    maxPenalty: varchar("max_penalty", { length: 255 }),
    jurisdiction: varchar({ length: 100 }),
    effectiveDate: timestamp("effective_date", { mode: "string" }),
    aiSummary: text("ai_summary"),
    tags: jsonb().default([]).notNull(),
    relatedStatutes: jsonb("related_statutes").default([]).notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    name: varchar({ length: 255 }),
    sectionNumber: varchar("section_number", { length: 50 }),
  },
  (table) => [unique("statutes_code_unique").on(table.code)],
);

export const users = pgTable(
  "users",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    email: varchar({ length: 255 }).notNull(),
    emailVerified: timestamp("email_verified", { mode: "string" }),
    hashedPassword: text("hashed_password"),
    role: varchar({ length: 50 }).default("prosecutor").notNull(),
    isActive: boolean("is_active").default(true).notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    firstName: varchar("first_name", { length: 100 }),
    lastName: varchar("last_name", { length: 100 }),
    name: text(),
    title: varchar({ length: 100 }),
    department: varchar({ length: 200 }),
    phone: varchar({ length: 20 }),
    officeAddress: text("office_address"),
    avatar: text(),
    bio: text(),
    specializations: jsonb().default([]).notNull(),
    username: varchar({ length: 100 }),
    image: text(),
    avatarUrl: text("avatar_url"),
    provider: varchar({ length: 50 }).default("credentials"),
    profile: jsonb().default({}),
  },
  (table) => [unique("users_email_unique").on(table.email)],
);

export const caseActivities = pgTable(
  "case_activities",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    caseId: uuid("case_id").notNull(),
    activityType: varchar("activity_type", { length: 50 }).notNull(),
    title: varchar({ length: 255 }).notNull(),
    description: text(),
    scheduledFor: timestamp("scheduled_for", { mode: "string" }),
    completedAt: timestamp("completed_at", { mode: "string" }),
    status: varchar({ length: 20 }).default("pending").notNull(),
    priority: varchar({ length: 20 }).default("medium").notNull(),
    assignedTo: uuid("assigned_to"),
    relatedEvidence: jsonb("related_evidence").default([]).notNull(),
    relatedCriminals: jsonb("related_criminals").default([]).notNull(),
    metadata: jsonb().default({}).notNull(),
    createdBy: uuid("created_by"),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "case_activities_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.assignedTo],
      foreignColumns: [users.id],
      name: "case_activities_assigned_to_users_id_fk",
    }),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "case_activities_created_by_users_id_fk",
    }),
  ],
);

export const verificationToken = pgTable(
  "verificationToken",
  {
    identifier: text().notNull(),
    token: text().notNull(),
    expires: timestamp({ mode: "string" }).notNull(),
  },
  (table) => [
    primaryKey({
      columns: [table.identifier, table.token],
      name: "verificationToken_identifier_token_pk",
    }),
  ],
);

export const account = pgTable(
  "account",
  {
    userId: uuid().notNull(),
    type: text().notNull(),
    provider: text().notNull(),
    providerAccountId: text().notNull(),
    refreshToken: text("refresh_token"),
    accessToken: text("access_token"),
    expiresAt: integer("expires_at"),
    tokenType: text("token_type"),
    scope: text(),
    idToken: text("id_token"),
    sessionState: text("session_state"),
  },
  (table) => [
    foreignKey({
      columns: [table.userId],
      foreignColumns: [users.id],
      name: "account_userId_users_id_fk",
    }).onDelete("cascade"),
    primaryKey({
      columns: [table.provider, table.providerAccountId],
      name: "account_provider_providerAccountId_pk",
    }),
  ],
);

// === REPORT BUILDER TABLES ===

// Reports table for the Case Books and Report Builder feature
export const reports = pgTable(
  "reports",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    title: varchar({ length: 255 }).notNull(),
    content: text(), // HTML content from contenteditable editor
    summary: text(),
    caseId: uuid("case_id").notNull(),
    reportType: varchar("report_type", { length: 50 })
      .default("prosecution_memo")
      .notNull(), // 'prosecution_memo', 'case_brief', 'evidence_summary', 'timeline', 'custom'
    status: varchar({ length: 20 }).default("draft").notNull(), // 'draft', 'in_review', 'final', 'archived'
    confidentialityLevel: varchar("confidentiality_level", { length: 20 })
      .default("restricted")
      .notNull(),
    jurisdiction: varchar({ length: 100 }),
    tags: jsonb().default([]).notNull(),
    metadata: jsonb().default({}).notNull(), // Store additional properties like priority, category, etc.
    sections: jsonb().default([]).notNull(), // Array of ReportSection objects
    aiSummary: text("ai_summary"),
    aiTags: jsonb("ai_tags").default([]).notNull(),
    wordCount: integer("word_count").default(0).notNull(),
    estimatedReadTime: integer("estimated_read_time").default(0).notNull(), // in minutes
    templateId: uuid("template_id"), // Future: link to report templates
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    lastEditedBy: uuid("last_edited_by"),
    publishedAt: timestamp("published_at", { mode: "string" }),
    archivedAt: timestamp("archived_at", { mode: "string" }),
  },
  (table) => [
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "reports_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "reports_created_by_users_id_fk",
    }),
    foreignKey({
      columns: [table.lastEditedBy],
      foreignColumns: [users.id],
      name: "reports_last_edited_by_users_id_fk",
    }),
  ],
);

// Citation Points table for managing citations and references
export const citationPoints = pgTable(
  "citation_points",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    text: text().notNull(), // The actual citation text
    source: varchar({ length: 500 }).notNull(), // Source reference (statute code, case name, etc.)
    page: integer(), // Page number if applicable
    context: text(), // Surrounding context or quote
    type: varchar({ length: 50 }).default("statute").notNull(), // 'statute', 'case_law', 'evidence', 'expert_opinion', 'testimony'
    jurisdiction: varchar({ length: 100 }),
    tags: jsonb().default([]).notNull(),
    caseId: uuid("case_id"),
    reportId: uuid("report_id"),
    evidenceId: uuid("evidence_id"), // Link to evidence if applicable
    statuteId: uuid("statute_id"), // Link to statute if applicable
    aiSummary: text("ai_summary"),
    relevanceScore: numeric("relevance_score", {
      precision: 4,
      scale: 3,
    }).default("0.0"), // AI-computed relevance 0.0-1.0
    metadata: jsonb().default({}).notNull(), // Additional properties, AI analysis, etc.
    isBookmarked: boolean("is_bookmarked").default(false).notNull(),
    usageCount: integer("usage_count").default(0).notNull(), // How many times cited
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "citation_points_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.reportId],
      foreignColumns: [reports.id],
      name: "citation_points_report_id_reports_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.evidenceId],
      foreignColumns: [evidence.id],
      name: "citation_points_evidence_id_evidence_id_fk",
    }).onDelete("set null"),
    foreignKey({
      columns: [table.statuteId],
      foreignColumns: [statutes.id],
      name: "citation_points_statute_id_statutes_id_fk",
    }).onDelete("set null"),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "citation_points_created_by_users_id_fk",
    }),
  ],
);

// Canvas States table for storing Fabric.js canvas data
export const canvasStates = pgTable(
  "canvas_states",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    title: varchar({ length: 255 }),
    reportId: uuid("report_id").notNull(),
    canvasData: jsonb("canvas_data").notNull(), // Serialized Fabric.js canvas state
    thumbnailUrl: text("thumbnail_url"), // Generated thumbnail for quick preview
    dimensions: jsonb().default({ width: 800, height: 600 }).notNull(),
    backgroundColor: varchar("background_color", { length: 20 }).default(
      "#ffffff",
    ),
    metadata: jsonb().default({}).notNull(), // Evidence IDs, citation IDs, annotations, etc.
    version: integer().default(1).notNull(), // For version control
    isTemplate: boolean("is_template").default(false).notNull(),
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.reportId],
      foreignColumns: [reports.id],
      name: "canvas_states_report_id_reports_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "canvas_states_created_by_users_id_fk",
    }),
  ],
);

// AI Analysis table for storing AI-generated insights
export const aiAnalyses = pgTable(
  "ai_analyses",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    reportId: uuid("report_id"),
    citationId: uuid("citation_id"),
    caseId: uuid("case_id"),
    analysisType: varchar("analysis_type", { length: 50 }).notNull(), // 'summary', 'keyword_extraction', 'sentiment', 'citation_suggestion', 'legal_precedent'
    inputData: jsonb("input_data").notNull(), // The data that was analyzed
    result: jsonb().notNull(), // AI analysis result with confidence, content, metadata
    model: varchar({ length: 100 }), // AI model used (e.g., 'gpt-4', 'claude-3', 'local-llm')
    tokens: integer().default(0), // Token count for cost tracking
    processingTime: integer("processing_time").default(0), // milliseconds
    confidence: numeric({ precision: 4, scale: 3 }).default("0.0"), // 0.0-1.0
    status: varchar({ length: 20 }).default("completed").notNull(), // 'pending', 'completed', 'failed'
    errorMessage: text("error_message"),
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.reportId],
      foreignColumns: [reports.id],
      name: "ai_analyses_report_id_reports_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.citationId],
      foreignColumns: [citationPoints.id],
      name: "ai_analyses_citation_id_citation_points_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.caseId],
      foreignColumns: [cases.id],
      name: "ai_analyses_case_id_cases_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "ai_analyses_created_by_users_id_fk",
    }),
  ],
);

// Report Templates table for standardized report formats
export const reportTemplates = pgTable(
  "report_templates",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    name: varchar({ length: 255 }).notNull(),
    description: text(),
    templateContent: text("template_content").notNull(), // HTML template with placeholders
    reportType: varchar("report_type", { length: 50 }).notNull(),
    jurisdiction: varchar({ length: 100 }),
    isDefault: boolean("is_default").default(false).notNull(),
    isActive: boolean("is_active").default(true).notNull(),
    metadata: jsonb().default({}).notNull(), // Template-specific configuration
    usageCount: integer("usage_count").default(0).notNull(),
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
    updatedAt: timestamp("updated_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "report_templates_created_by_users_id_fk",
    }),
  ],
);

// Export History table for tracking PDF/document exports
export const exportHistory = pgTable(
  "export_history",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    reportId: uuid("report_id").notNull(),
    exportFormat: varchar("export_format", { length: 20 }).notNull(), // 'pdf', 'docx', 'html', 'json'
    fileName: varchar("file_name", { length: 500 }).notNull(),
    fileSize: integer("file_size"), // in bytes
    downloadUrl: text("download_url"),
    status: varchar({ length: 20 }).default("pending").notNull(), // 'pending', 'completed', 'failed', 'expired'
    options: jsonb().default({}).notNull(), // Export options used
    errorMessage: text("error_message"),
    expiresAt: timestamp("expires_at", { mode: "string" }), // When download link expires
    downloadCount: integer("download_count").default(0).notNull(),
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.reportId],
      foreignColumns: [reports.id],
      name: "export_history_report_id_reports_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.createdBy],
      foreignColumns: [users.id],
      name: "export_history_created_by_users_id_fk",
    }),
  ],
);

// Hash Verifications table for tracking file integrity verification history
export const hashVerifications = pgTable(
  "hash_verifications",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    evidenceId: uuid("evidence_id").notNull(),
    verifiedHash: varchar("verified_hash", { length: 64 }).notNull(), // Hash provided for verification
    storedHash: varchar("stored_hash", { length: 64 }), // Hash stored in database
    result: boolean().notNull(), // True if hashes match, false otherwise
    verificationMethod: varchar("verification_method", { length: 50 })
      .default("manual")
      .notNull(), // 'manual', 'api', 'bulk', 'automatic'
    verifiedBy: uuid("verified_by").notNull(),
    notes: text(),
    verifiedAt: timestamp("verified_at", { mode: "string" })
      .defaultNow()
      .notNull(),
  },
  (table) => [
    foreignKey({
      columns: [table.evidenceId],
      foreignColumns: [evidence.id],
      name: "hash_verifications_evidence_id_evidence_id_fk",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.verifiedBy],
      foreignColumns: [users.id],
      name: "hash_verifications_verified_by_users_id_fk",
    }),
  ],
);
