// Unified Schema (lightweight aggregator)
// Bridges canonical snake_case auth + core tables (schema-postgres) with supplemental
// evidence/analysis domain tables defined in ../schema.ts, applying safe aliasing
// to avoid identifier collisions. This file is the single Drizzle CLI entry point.

// Export core (snake_case) tables used by auth/session logic (authoritative definitions live in schema-postgres)
export * from './schema-postgres';

// Re-export evidence domain tables (camelCase variants) with aliases where names overlap
export {
  evidenceProcessTable,
  evidenceOcrTable,
  evidenceEmbeddingsTable,
  evidenceVectorsTable,
  evidenceAnalysisTable,
  evidenceTable as evidence_v2,
  casesTable as cases_v2,
  reportsTable as reports_v2,
  systemHealthTable,
  queueStatsTable,
  legalDocuments as legal_documents_v2,
  contentEmbeddings as content_embeddings_v2
} from '../schema';

// NOTE:
//  - The *_v2 aliases represent legacy/alternate modeling that co-exists during migration.
//  - Migrations generated from this file will not create duplicate base tables already present
//    (e.g. 'evidence') because we only export its v2 alias (evidence_v2) not the underlying name.
//  - Downstream code should gradually consolidate to the snake_case tables from schema-postgres.
import { pgTable, uuid, varchar, text, timestamp, integer, decimal, boolean, jsonb, serial, index } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users, statutes, evidence, criminals, cases } from './schema-postgres';
import { createId } from '@paralleldrive/cuid2';
// Core tables imported explicitly for foreign key references; their definitions live in schema-postgres.

// Type definitions for complex JSON fields
export interface DocumentMetadataExt {
  keywords: string[];
  customFields: Record<string, unknown>;
  confidentialityLevel: 'public' | 'restricted' | 'confidential' | 'top_secret';
}

export interface Citation {
  id: string;
  text: string;
  source: string;
  page?: number;
  url?: string;
}

export interface AutoSaveData {
  content: string;
  citations: Citation[];
  autoSavedAt: string;
  isDirty: boolean;
}

export interface Collaborator {
  userId: string;
  role: 'viewer' | 'editor' | 'owner';
  addedAt: string;
}

// === CORE TABLES REMOVED HERE ===
// NOTE: users, sessions, cases, evidence, legal_documents, notes, reports, etc. are
// defined authoritatively in schema-postgres.ts and re-exported above. They were
// previously duplicated here with alternate/camelCase column names which caused
// migration drift & duplicate identifier issues. If extended versions are needed
// during migration, introduce them with *_v2 variable names AND distinct physical
// table names (e.g. 'cases_v2') to avoid conflicts.

// === CRIMINALS ===

// (Removed criminals shadow table – canonical criminals lives in schema-postgres)

// === THEMES ===

// (Removed themes & userThemes shadow tables – rely on canonical or add *_v2 if divergence required)

export type MinimalSchemaNames = 'users' | 'sessions' | 'cases' | 'evidence';

// === CASE MANAGEMENT ===

// (cases definition removed – rely on canonical schema-postgres cases)

// === CASE-CRIMINAL RELATIONSHIPS ===

// (Removed caseCriminals duplicate – rely on canonical or reintroduce with distinct name if missing)

// === PERSONS OF INTEREST ===

// (Removed personsOfInterest duplicate – canonical version already exported)

// === EVIDENCE MANAGEMENT ===

// (evidence definition removed – rely on canonical schema-postgres evidence)

// === STATUTES & LEGAL REFERENCES ===

// (Removed statutes duplicate – canonical table present)

// === CASE ACTIVITIES & TIMELINE ===

// (Removed caseActivities duplicate – canonical version retained)

// === AI & SEARCH METADATA ===

// (Removed aiAnalyses duplicate – canonical or future analytics table should be single source)

// (Removed searchTags duplicate – canonical table lives in schema-postgres if required)

// === EXPORT & REPORTING ===

// === LEGAL DOCUMENTS ===

// (legalDocuments alternative model removed – canonical legal_documents from schema-postgres retained)

// === NOTES ===

// (notes duplicate removed – rely on canonical notes table if present in schema-postgres)

// === SAVED CITATIONS (domain-specific extension retained) ===

export const savedCitations = pgTable(
  "saved_citations",
  {
    id: text("id")
      .primaryKey()
      .$defaultFn(() => createId()), // Custom CUID2 ID
    userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
    citationPointId: uuid("citation_point_id").references(
      () => citationPoints.id,
      { onDelete: "cascade" },
    ),
    title: varchar("title", { length: 255 }),
    description: text("description"),

    // Type-safe JSON fields
    citationData: jsonb("citation_data").notNull(),
    tags: jsonb("tags").default(sql`'[]'::jsonb`).notNull(),
    metadata: jsonb("metadata")
      .default(sql`'{}'::jsonb`)
      .notNull(),

    // Organization
    category: varchar("category", { length: 50 }).default("general").notNull(),
    isFavorite: boolean("is_favorite").default(false).notNull(),
    isArchived: boolean("is_archived").default(false).notNull(),

    // Usage tracking
    usageCount: integer("usage_count").default(0).notNull(),
    lastUsedAt: timestamp("last_used_at", { mode: "date" }),

    // Timestamps
    createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
  },
  (table) => ({
    // Indexes for efficient queries
    userIdIdx: index("saved_citations_user_id_idx").on(table.userId),
    categoryIdx: index("saved_citations_category_idx").on(table.category),
    isFavoriteIdx: index("saved_citations_favorite_idx").on(table.isFavorite),
    usageCountIdx: index("saved_citations_usage_idx").on(table.usageCount)
  }),
);

// === CANVAS STATES ===

export const canvasStates = pgTable("canvas_states", {
  id: serial("id").primaryKey(),
  title: varchar("title", { length: 255 }),
  // reportId: uuid("report_id").references(() => reports.id, {
  //   onDelete: "cascade"
  // }), // TODO: Define reports table or remove this reference
  reportId: uuid("report_id"), // Simplified - no foreign key reference
  caseId: uuid("case_id")
    .notNull()
    .references(() => cases.id, { onDelete: "cascade" }),
  canvasData: text("canvas_data").notNull(),
  thumbnailUrl: text("thumbnail_url"),
  dimensions: jsonb("dimensions")
    .default({ width: 800, height: 600 })
    .notNull(),
  backgroundColor: varchar("background_color", { length: 20 }).default(
    "#ffffff",
  ),
  version: integer("version").default(1).notNull(),
  isTemplate: boolean("is_template").default(false).notNull(),
  imagePreview: text("image_preview"),
  metadata: text("metadata"),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === CITATION POINTS ===

export const citationPoints = pgTable("citation_points", {
  id: uuid("id").primaryKey().defaultRandom(),
  text: text("text").notNull(), // The actual citation text
  source: varchar("source", { length: 500 }).notNull(), // Source reference (statute code, case name, etc.)
  page: integer("page"), // Page number if applicable
  context: text("context"), // Surrounding context or quote
  type: varchar("type", { length: 50 }).default("statute").notNull(), // 'statute', 'case_law', 'evidence', 'expert_opinion', 'testimony'
  jurisdiction: varchar("jurisdiction", { length: 100 }),
  tags: jsonb("tags").default([]).notNull(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  // reportId: uuid("report_id").references(() => reports.id, {
  //   onDelete: "cascade"
  // }), // TODO: Define reports table or remove this reference  
  reportId: uuid("report_id"), // Simplified - no foreign key reference
  evidenceId: uuid("evidence_id").references(() => evidence.id, {
    onDelete: "set null"
  }),
  statuteId: uuid("statute_id").references(() => statutes.id, {
    onDelete: "set null"
  }),
  aiSummary: text("ai_summary"),
  relevanceScore: decimal("relevance_score", {
    precision: 4,
    scale: 3
  }).default("0.0"),
  metadata: jsonb("metadata").default({}).notNull(),
  isBookmarked: boolean("is_bookmarked").default(false).notNull(),
  usageCount: integer("usage_count").default(0).notNull(),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === HASH VERIFICATIONS ===

export const hashVerifications = pgTable("hash_verifications", {
  id: uuid("id").primaryKey().defaultRandom(),
  evidenceId: uuid("evidence_id")
    .notNull()
    .references(() => evidence.id, { onDelete: "cascade" }),
  verifiedHash: varchar("verified_hash", { length: 64 }).notNull(),
  storedHash: varchar("stored_hash", { length: 64 }),
  result: boolean("result").notNull(),
  verificationMethod: varchar("verification_method", { length: 50 })
    .default("manual")
    .notNull(),
  verifiedBy: uuid("verified_by")
    .notNull()
    .references(() => users.id),
  notes: text("notes"),
  verifiedAt: timestamp("verified_at", { mode: "date" }).defaultNow().notNull()
});

// === ATTACHMENT VERIFICATIONS ===

export const attachmentVerifications = pgTable("attachment_verifications", {
  id: uuid("id").primaryKey().defaultRandom(),
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
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === CRIMES (legacy compatibility) ===

export const crimes = pgTable("crimes", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
  criminalId: uuid("criminal_id").references(() => criminals.id, {
    onDelete: "cascade"
  }),
  statuteId: uuid("statute_id").references(() => statutes.id),
  name: varchar("name", { length: 255 }).notNull(),
  description: text("description"),
  chargeLevel: varchar("charge_level", { length: 50 }),
  status: varchar("status", { length: 50 }).default("pending").notNull(),
  incidentDate: timestamp("incident_date", { mode: "date" }),
  arrestDate: timestamp("arrest_date", { mode: "date" }),
  filingDate: timestamp("filing_date", { mode: "date" }),
  notes: text("notes"),
  aiSummary: text("ai_summary"),
  metadata: jsonb("metadata").default({}).notNull(),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === RELATIONSHIPS ===

// (All relations for core tables removed from this aggregator to avoid duplicate declarations.
//  Use relations defined in schema-postgres.ts exclusively.)
