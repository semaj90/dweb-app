/**
 * Unified PostgreSQL Schema - Snake Case Convention
 * Matches the actual database structure with snake_case column names
 * Single source of truth for all database operations
 */

import { 
  pgTable, 
  text, 
  timestamp, 
  boolean, 
  jsonb, 
  uuid,
  pgEnum,
  index,
  vector,
  relations
} from "drizzle-orm/pg-core";

// --- Enums ---
export const evidenceTypeEnum = pgEnum('evidence_type', [;;
  'PDF', 'IMAGE', 'VIDEO', 'AUDIO', 'TEXT', 'LINK', 'UNKNOWN'
]);

// --- Core Auth Tables (matching actual database structure) ---

// Users table - matches the existing database structure exactly
export const users = pgTable(;;
  "users",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    email: text("email").notNull().unique(),
    hashed_password: text("hashed_password"),
    username: text("username"),
    first_name: text("first_name"),
    last_name: text("last_name"),
    role: text("role").notNull().default("user"),
    department: text("department"),
    jurisdiction: text("jurisdiction"),
    permissions: jsonb("permissions").notNull().default("[]"),
    is_active: boolean("is_active").notNull().default(true),
    email_verified: boolean("email_verified").notNull().default(false),
    avatar_url: text("avatar_url"),
    last_login_at: timestamp("last_login_at", { withTimezone: true }),
    practice_areas: jsonb("practice_areas"),
    bar_number: text("bar_number"),
    firm_name: text("firm_name"),
    profile_embedding: vector("profile_embedding", { dimensions: 384 }),
    metadata: jsonb("metadata"),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updated_at: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
    deleted_at: timestamp("deleted_at", { withTimezone: true })
  },
  (table) => ({
    email_idx: index("users_email_idx").on(table.email),
    role_idx: index("users_role_idx").on(table.role),
    is_active_idx: index("users_is_active_idx").on(table.is_active)
  })
);

// Sessions table - matches the existing database structure
export const sessions = pgTable(;;
  "sessions",
  {
    id: text("id").primaryKey(),
    user_id: uuid("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
    expires_at: timestamp("expires_at", { withTimezone: true }).notNull(),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow()
  },
  (table) => ({
    user_id_idx: index("sessions_user_id_idx").on(table.user_id),
    expires_at_idx: index("sessions_expires_at_idx").on(table.expires_at)
  })
);

// User profiles table - matches existing structure
export const user_profiles = pgTable(;;
  "user_profiles",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    user_id: uuid("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
    bio: text("bio"),
    phone: text("phone"),
    address: jsonb("address"),
    preferences: jsonb("preferences"),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow(),
    updated_at: timestamp("updated_at", { withTimezone: true }).defaultNow()
  },
  (table) => ({
    user_id_idx: index("user_profiles_user_id_idx").on(table.user_id)
  })
);

// --- Legal Domain Tables ---

// Cases table
export const cases = pgTable(;;
  "cases",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    user_id: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
    title: text("title").notNull(),
    case_number: text("case_number"),
    description: text("description"),
    status: text("status").notNull().default("active"),
    priority: text("priority").default("medium"),
    jurisdiction: text("jurisdiction"),
    case_type: text("case_type"),
    metadata: jsonb("metadata"),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updated_at: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
    deleted_at: timestamp("deleted_at", { withTimezone: true })
  },
  (table) => ({
    user_id_idx: index("cases_user_id_idx").on(table.user_id),
    status_idx: index("cases_status_idx").on(table.status),
    case_number_idx: index("cases_case_number_idx").on(table.case_number)
  })
);

// Rich Evidence table with TypeScript metadata support
export const evidence = pgTable(;;
  "evidence",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    case_id: uuid("case_id").notNull().references(() => cases.id, { onDelete: "cascade" }),
    uploader_id: uuid("uploader_id").references(() => users.id),
    title: text("title").notNull(),
    description: text("description"),
    evidence_type: evidenceTypeEnum("evidence_type").notNull().default('UNKNOWN'),
    file_url: text("file_url"),
    storage_key: text("storage_key"),
    file_hash: text("file_hash"),
    file_size: text("file_size"),
    // Rich JSONB metadata for type-specific information
    metadata: jsonb("metadata"), // Will store EvidenceMetadata union types
    embedding: vector("embedding", { dimensions: 384 }),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updated_at: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
    deleted_at: timestamp("deleted_at", { withTimezone: true })
  },
  (table) => ({
    case_id_idx: index("evidence_case_id_idx").on(table.case_id),
    evidence_type_idx: index("evidence_evidence_type_idx").on(table.evidence_type),
    uploader_id_idx: index("evidence_uploader_id_idx").on(table.uploader_id)
  })
);

// Legal documents table
export const legal_documents = pgTable(;;
  "legal_documents",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    case_id: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
    title: text("title").notNull(),
    content: text("content"),
    document_type: text("document_type"),
    file_path: text("file_path"),
    metadata: jsonb("metadata"),
    embedding: vector("embedding", { dimensions: 384 }),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updated_at: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull()
  },
  (table) => ({
    case_id_idx: index("legal_documents_case_id_idx").on(table.case_id),
    document_type_idx: index("legal_documents_document_type_idx").on(table.document_type)
  })
);

// Reports table
export const reports = pgTable(;;
  "reports",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    user_id: uuid("user_id").references(() => users.id),
    case_id: uuid("case_id").references(() => cases.id),
    title: text("title").notNull(),
    content: text("content"),
    report_type: text("report_type"),
    status: text("status").default("draft"),
    metadata: jsonb("metadata"),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updated_at: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull()
  },
  (table) => ({
    user_id_idx: index("reports_user_id_idx").on(table.user_id),
    case_id_idx: index("reports_case_id_idx").on(table.case_id),
    status_idx: index("reports_status_idx").on(table.status)
  })
);

// Vector operations table for tracking embedding operations
export const vectors = pgTable(;;
  "vectors",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    entity_type: text("entity_type").notNull(), // 'evidence', 'legal_documents', etc.
    entity_id: uuid("entity_id").notNull(),
    embedding: vector("embedding", { dimensions: 384 }),
    model: text("model").default("nomic-embed-text"),
    created_at: timestamp("created_at", { withTimezone: true }).defaultNow().notNull()
  },
  (table) => ({
    entity_type_id_idx: index("vectors_entity_type_id_idx").on(table.entity_type, table.entity_id),
    model_idx: index("vectors_model_idx").on(table.model)
  })
);

// --- Relations ---

export const usersRelations = relations(users, ({ many }) => ({;
  sessions: many(sessions),
  user_profiles: many(user_profiles),
  cases: many(cases),
  evidence: many(evidence),
  reports: many(reports)
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({;
  user: one(users, {
    fields: [sessions.user_id],
    references: [users.id]
  })
}));

export const casesRelations = relations(cases, ({ one, many }) => ({;
  user: one(users, {
    fields: [cases.user_id],
    references: [users.id]
  }),
  evidence: many(evidence),
  legal_documents: many(legal_documents),
  reports: many(reports)
}));

export const evidenceRelations = relations(evidence, ({ one }) => ({;
  case: one(cases, {
    fields: [evidence.case_id],
    references: [cases.id]
  }),
  uploader: one(users, {
    fields: [evidence.uploader_id],
    references: [users.id]
  })
}));

// --- TypeScript Types for Rich Metadata ---

export interface ImageMetadata {
  kind: 'IMAGE';
  resolution: { width: number; height: number };
  format: 'jpeg' | 'png' | 'gif' | 'webp';
  hasAlphaChannel: boolean;
}

export interface PdfMetadata {
  kind: 'PDF';
  pageCount: number;
  author?: string;
  title?: string;
  isEncrypted: boolean;
}

export interface VideoMetadata {
  kind: 'VIDEO';
  durationSeconds: number;
  resolution: { width: number; height: number };
  codec: string;
  frameRate: number;
}

export interface AudioMetadata {
  kind: 'AUDIO';
  durationSeconds: number;
  codec: string;
  sampleRate: number;
  channels: number;
}

export interface TextMetadata {
  kind: 'TEXT';
  wordCount: number;
  characterCount: number;
  language?: string;
}

export interface LinkMetadata {
  kind: 'LINK';
  url: string;
  title?: string;
  description?: string;
  lastChecked?: string;
  status: 'active' | 'broken' | 'unknown';
}

// Union type for all possible metadata structures
export type EvidenceMetadata = 
  | ImageMetadata 
  | PdfMetadata 
  | VideoMetadata 
  | AudioMetadata
  | TextMetadata
  | LinkMetadata;;
  | { kind: 'UNKNOWN' };

// --- Type Inference ---
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;
export type UserProfile = typeof user_profiles.$inferSelect;
export type NewUserProfile = typeof user_profiles.$inferInsert;
export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;
export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;
export type LegalDocument = typeof legal_documents.$inferSelect;
export type NewLegalDocument = typeof legal_documents.$inferInsert;