// Unified Schema (lightweight aggregator)
// Bridges canonical snake_case auth + core tables (schema-postgres) with supplemental
// evidence/analysis domain tables defined in ../schema.ts, applying safe aliasing
// to avoid identifier collisions. This file is the single Drizzle CLI entry point.

// Export core (snake_case) tables used by auth/session logic (authoritative definitions live in schema-postgres)
export * from './schema-postgres';
import { users, sessions, cases, evidence, reports, legalDocuments, notes } from './schema-postgres';

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
import { pgTable, uuid, varchar, text, timestamp, integer, decimal, boolean, jsonb, index } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { relations } from 'drizzle-orm/relations';
import { createId } from '@paralleldrive/cuid2';

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

export const criminals = pgTable('criminals', {
  id: uuid('id').primaryKey().defaultRandom(),
  firstName: varchar('first_name', { length: 100 }).notNull(),
  lastName: varchar('last_name', { length: 100 }).notNull(),
  aliases: jsonb('aliases').default([]).notNull(),
  dateOfBirth: timestamp('date_of_birth', { mode: 'date' }),
  placeOfBirth: varchar('place_of_birth', { length: 255 }),
  gender: varchar('gender', { length: 20 }),
  height: varchar('height', { length: 20 }),
  weight: varchar('weight', { length: 20 }),
  eyeColor: varchar('eye_color', { length: 20 }),
  hairColor: varchar('hair_color', { length: 20 }),
  distinguishingMarks: text('distinguishing_marks'),
  lastKnownAddress: text('last_known_address'),
  phoneNumbers: jsonb('phone_numbers').default([]).notNull(),
  email: varchar('email', { length: 255 }),
  socialSecurityNumber: varchar('social_security_number', { length: 20 }),
  driverLicenseNumber: varchar('driver_license_number', { length: 50 }),
  fingerprints: text('fingerprints'),
  dnaProfile: text('dna_profile'),
  mugshots: jsonb('mugshots').default([]).notNull(),
  knownAssociates: jsonb('known_associates').default([]).notNull(),
  criminalHistory: jsonb('criminal_history').default([]).notNull(),
  riskLevel: varchar('risk_level', { length: 20 }).default('low').notNull(),
  status: varchar('status', { length: 50 }).default('active').notNull(),
  notes: text('notes'),
  aiAnalysis: jsonb('ai_analysis').default({}).notNull(),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull()
});

// === THEMES ===

export const themes = pgTable('themes', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 100 }).notNull().unique(),
  colors: jsonb('colors').notNull(),
  fonts: jsonb('fonts').default({}).notNull(),
  isDefault: boolean('is_default').default(false).notNull(),
  isPublic: boolean('is_public').default(true).notNull(),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull()
});

export const userThemes = pgTable('user_themes', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  themeId: uuid('theme_id').notNull().references(() => themes.id, { onDelete: 'cascade' }),
  isActive: boolean('is_active').default(false).notNull(),
  customizations: jsonb('customizations').default({}).notNull(),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull()
});

export type MinimalSchemaNames = 'users' | 'sessions' | 'cases' | 'criminals' | 'evidence' | 'themes' | 'userThemes';

// === CASE MANAGEMENT ===

// (cases definition removed – rely on canonical schema-postgres cases)

// === CASE-CRIMINAL RELATIONSHIPS ===

export const caseCriminals = pgTable("case_criminals", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id")
    .notNull()
    .references(() => cases.id, { onDelete: "cascade" }),
  criminalId: uuid("criminal_id")
    .notNull()
    .references(() => criminals.id, { onDelete: "cascade" }),
  role: varchar("role", { length: 50 }).default("suspect").notNull(), // suspect, defendant, witness, victim
  charges: jsonb("charges").default([]).notNull(), // array of charge objects
  conviction: boolean("conviction").default(false).notNull(),
  sentencing: jsonb("sentencing").default({}).notNull(),
  notes: text("notes"),
  addedBy: uuid("added_by").references(() => users.id),
  addedAt: timestamp("added_at", { mode: "date" }).defaultNow().notNull()
});

// === PERSONS OF INTEREST ===

export const personsOfInterest = pgTable("persons_of_interest", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id")
    .notNull()
    .references(() => cases.id, { onDelete: "cascade" }),

  // Basic information
  name: varchar("name", { length: 256 }).notNull(),
  aliases: jsonb("aliases").default([]).notNull(),
  profileImageUrl: text("profile_image_url"),

  // The structured "Who, What, Why, How" profile data
  profileData: jsonb("profile_data")
    .default({
      who: "",
      what: "",
      why: "",
      how: ""
    })
    .notNull(),

  // Position on canvas
  posX: decimal("pos_x", { precision: 10, scale: 2 }).default("100").notNull(),
  posY: decimal("pos_y", { precision: 10, scale: 2 }).default("100").notNull(),

  // POI metadata
  relationship: varchar("relationship", { length: 100 }), // e.g., 'suspect', 'witness', 'victim', 'co-conspirator'
  threatLevel: varchar("threat_level", { length: 20 }).default("low").notNull(),
  status: varchar("status", { length: 20 }).default("active").notNull(),
  tags: jsonb("tags").default([]).notNull(),

  // Timestamps and ownership
  createdBy: uuid("created_by")
    .notNull()
    .references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === EVIDENCE MANAGEMENT ===

// (evidence definition removed – rely on canonical schema-postgres evidence)

// === STATUTES & LEGAL REFERENCES ===

export const statutes = pgTable("statutes", {
  id: uuid("id").primaryKey().defaultRandom(),
  code: varchar("code", { length: 50 }).notNull().unique(),
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  fullText: text("full_text"),
  category: varchar("category", { length: 100 }),
  severity: varchar("severity", { length: 20 }), // misdemeanor, felony, infraction
  minPenalty: varchar("min_penalty", { length: 255 }),
  maxPenalty: varchar("max_penalty", { length: 255 }),
  jurisdiction: varchar("jurisdiction", { length: 100 }),
  effectiveDate: timestamp("effective_date", { mode: "date" }),
  aiSummary: text("ai_summary"),
  tags: jsonb("tags").default([]).notNull(),
  relatedStatutes: jsonb("related_statutes").default([]).notNull(),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === CASE ACTIVITIES & TIMELINE ===

export const caseActivities = pgTable("case_activities", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id")
    .notNull()
    .references(() => cases.id, { onDelete: "cascade" }),
  activityType: varchar("activity_type", { length: 50 }).notNull(), // court_date, evidence_added, witness_interview, etc.
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  scheduledFor: timestamp("scheduled_for", { mode: "date" }),
  completedAt: timestamp("completed_at", { mode: "date" }),
  status: varchar("status", { length: 20 }).default("pending").notNull(),
  priority: varchar("priority", { length: 20 }).default("medium").notNull(),
  assignedTo: uuid("assigned_to").references(() => users.id),
  relatedEvidence: jsonb("related_evidence").default([]).notNull(), // evidence IDs
  relatedCriminals: jsonb("related_criminals").default([]).notNull(), // criminal IDs
  metadata: jsonb("metadata").default({}).notNull(),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { mode: "date" }).defaultNow().notNull()
});

// === AI & SEARCH METADATA ===

export const aiAnalyses = pgTable("ai_analyses", {
  id: uuid("id").primaryKey().defaultRandom(),
  entityType: varchar("entity_type", { length: 20 }).notNull(), // case, criminal, evidence
  entityId: uuid("entity_id").notNull(),
  analysisType: varchar("analysis_type", { length: 50 }).notNull(), // summary, sentiment, classification, etc.
  prompt: text("prompt"),
  response: jsonb("response").notNull(),
  confidence: decimal("confidence", { precision: 5, scale: 4 }), // 0.0000 to 1.0000
  model: varchar("model", { length: 100 }),
  version: varchar("version", { length: 20 }),
  processingTime: integer("processing_time"), // milliseconds
  tokens: integer("tokens"),
  cost: decimal("cost", { precision: 8, scale: 6 }), // API cost
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull()
});

export const searchTags = pgTable("search_tags", {
  id: uuid("id").primaryKey().defaultRandom(),
  entityType: varchar("entity_type", { length: 20 }).notNull(),
  entityId: uuid("entity_id").notNull(),
  tag: varchar("tag", { length: 100 }).notNull(),
  category: varchar("category", { length: 50 }), // auto, manual, ai
  confidence: decimal("confidence", { precision: 5, scale: 4 }),
  source: varchar("source", { length: 50 }), // ai_model, user, ocr, etc.
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at", { mode: "date" }).defaultNow().notNull()
});

// === EXPORT & REPORTING ===

// (reports definition removed – rely on canonical schema-postgres or add distinct reports_v2 later)

// === LEGAL DOCUMENTS ===

// (legalDocuments alternative model removed – canonical legal_documents from schema-postgres retained)

// === NOTES ===

// (notes duplicate removed – rely on canonical notes table if present in schema-postgres)
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
  reportId: uuid("report_id").references(() => reports.id, {
    onDelete: "cascade"
  }),
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

// === SAVED CITATIONS ===

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
  })
);
// (citationPoints moved above savedCitations to prevent TDZ runtime errors)

// === CITATION POINTS ===
// (duplicate definition removed – original citationPoints table is defined earlier above savedCitations)

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

export const usersRelations = relations(users, ({ one, many }) => ({
  sessions: many(sessions),
  casesAsLead: many(cases, { relationName: "leadProsecutor" }),
  casesCreated: many(cases, { relationName: "createdBy" }),
  evidenceUploaded: many(evidence),
  activitiesAssigned: many(caseActivities, { relationName: "assignedTo" }),
  activitiesCreated: many(caseActivities, { relationName: "createdBy" }),
  criminalsCreated: many(criminals),
  aiAnalyses: many(aiAnalyses),
  searchTags: many(searchTags),
  reports: many(reports),
  legalDocuments: many(legalDocuments),
  notes: many(notes),
  savedCitations: many(savedCitations)
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, {
    fields: [sessions.userId],
    references: [users.id]
  })
}));

export const casesRelations = relations(cases, ({ one, many }) => ({
  leadProsecutor: one(users, {
    fields: [cases.leadProsecutor],
    references: [users.id],
    relationName: "leadProsecutor"
  }),
  createdBy: one(users, {
    fields: [cases.createdBy],
    references: [users.id],
    relationName: "createdBy"
  }),
  criminals: many(caseCriminals),
  evidence: many(evidence),
  activities: many(caseActivities),
  personsOfInterest: many(personsOfInterest),
  legalDocuments: many(legalDocuments),
  notes: many(notes)
}));

export const criminalsRelations = relations(criminals, ({ one, many }) => ({
  createdBy: one(users, {
    fields: [criminals.createdBy],
    references: [users.id]
  }),
  cases: many(caseCriminals),
  evidence: many(evidence)
}));

export const caseCriminalsRelations = relations(caseCriminals, ({ one }) => ({
  case: one(cases, {
    fields: [caseCriminals.caseId],
    references: [cases.id]
  }),
  criminal: one(criminals, {
    fields: [caseCriminals.criminalId],
    references: [criminals.id]
  }),
  addedBy: one(users, {
    fields: [caseCriminals.addedBy],
    references: [users.id]
  })
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({
  uploadedBy: one(users, {
    fields: [evidence.uploadedBy],
    references: [users.id]
  }),
  case: one(cases, {
    fields: [evidence.caseId],
    references: [cases.id]
  })
}));

export const themesRelations = relations(themes, ({ one }) => ({
  createdBy: one(users, {
    fields: [themes.createdBy],
    references: [users.id]
  })
}));

export const reportsRelations = relations(reports, ({ one }) => ({
  createdBy: one(users, {
    fields: [reports.createdBy],
    references: [users.id]
  }),
  lastEditedBy: one(users, {
    fields: [reports.lastEditedBy],
    references: [users.id]
  }),
  generatedBy: one(users, {
    fields: [reports.generatedBy],
    references: [users.id]
  })
}));

export const citationPointsRelations = relations(citationPoints, ({ one }) => ({
  createdBy: one(users, {
    fields: [citationPoints.createdBy],
    references: [users.id]
  })
}));

export const attachmentVerificationsRelations = relations(
  attachmentVerifications,
  ({ one }) => ({
    verifiedBy: one(users, {
      fields: [attachmentVerifications.verifiedBy],
      references: [users.id]
    })
  }),
);

export const personsOfInterestRelations = relations(
  personsOfInterest,
  ({ one }) => ({
    case: one(cases, {
      fields: [personsOfInterest.caseId],
      references: [cases.id]
    }),
    createdBy: one(users, {
      fields: [personsOfInterest.createdBy],
      references: [users.id]
    })
  }),
);

export const legalDocumentsRelations = relations(legalDocuments, ({ one }) => ({
  case: one(cases, {
    fields: [legalDocuments.caseId],
    references: [cases.id]
  }),
  user: one(users, {
    fields: [legalDocuments.userId],
    references: [users.id]
  })
}));

export const notesRelations = relations(notes, ({ one }) => ({
  case: one(cases, {
    fields: [notes.caseId],
    references: [cases.id]
  }),
  user: one(users, {
    fields: [notes.userId],
    references: [users.id]
  })
}));
