// @ts-nocheck
// src/lib/server/db/schema-postgres-enhanced.ts
import {
  pgTable,
  text,
  timestamp,
  uuid,
  boolean,
  integer,
  real,
  jsonb,
  index,
} from "drizzle-orm/pg-core";
import { vector } from "pgvector/drizzle-orm";
import { relations } from "drizzle-orm";

// Enable vector extension
// CREATE EXTENSION IF NOT EXISTS vector;

// Core tables with vector support
export const cases = pgTable(
  "cases",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: uuid("user_id").notNull(),
    title: text("title").notNull(),
    caseNumber: text("case_number").notNull().unique(),
    description: text("description").notNull(),
    incidentDate: timestamp("incident_date"),
    location: text("location"),
    priority: text("priority").notNull().default("medium"), // low, medium, high, urgent
    status: text("status").notNull().default("open"), // open, closed, pending, archived, under_review
    category: text("category").notNull(),
    dangerScore: integer("danger_score").default(0), // 0-10
    estimatedValue: real("estimated_value"),
    jurisdiction: text("jurisdiction"),
    leadProsecutor: text("lead_prosecutor"),
    assignedTeam: jsonb("assigned_team").default([]),
    tags: jsonb("tags").default([]),
    aiSummary: text("ai_summary"),
    metadata: jsonb("metadata").default({}),

    // Vector embeddings
    titleEmbedding: vector("title_embedding", { dimensions: 768 }),
    contentEmbedding: vector("content_embedding", { dimensions: 768 }),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    userIdIdx: index("cases_user_id_idx").on(table.userId),
    statusIdx: index("cases_status_idx").on(table.status),
    priorityIdx: index("cases_priority_idx").on(table.priority),
    categoryIdx: index("cases_category_idx").on(table.category),
    titleEmbeddingIdx: index("cases_title_embedding_idx").on(table.titleEmbedding),
    contentEmbeddingIdx: index("cases_content_embedding_idx").on(table.contentEmbedding),
  }),
);

export const evidence = pgTable(
  "evidence",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
    userId: uuid("user_id").notNull(),
    title: text("title").notNull(),
    description: text("description"),
    evidenceType: text("evidence_type").notNull(), // document, image, video, audio, physical, digital, testimony
    subType: text("sub_type"),

    // File information
    fileName: text("file_name"),
    fileSize: integer("file_size"),
    mimeType: text("mime_type"),
    hash: text("hash"),

    // Chain of custody
    collectedAt: timestamp("collected_at"),
    collectedBy: text("collected_by"),
    location: text("location"),
    chainOfCustody: jsonb("chain_of_custody").default([]),

    // Classification
    tags: jsonb("tags").default([]),
    isAdmissible: boolean("is_admissible").default(true),
    confidentialityLevel: text("confidentiality_level").default("standard"), // public, standard, confidential, classified

    // Analysis
    aiAnalysis: jsonb("ai_analysis").default({}),
    aiTags: jsonb("ai_tags").default([]),
    aiSummary: text("ai_summary"),
    summary: text("summary"),

    // Vector embeddings
    titleEmbedding: vector("title_embedding", { dimensions: 768 }),
    contentEmbedding: vector("content_embedding", { dimensions: 768 }),

    // Evidence board positioning
    boardPosition: jsonb("board_position").default({ x: 0, y: 0 }),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    caseIdIdx: index("evidence_case_id_idx").on(table.caseId),
    typeIdx: index("evidence_type_idx").on(table.evidenceType),
    admissibleIdx: index("evidence_admissible_idx").on(table.isAdmissible),
    titleEmbeddingIdx: index("evidence_title_embedding_idx").on(table.titleEmbedding),
    contentEmbeddingIdx: index("evidence_content_embedding_idx").on(table.contentEmbedding),
  }),
);

export const criminals = pgTable(
  "criminals",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: uuid("user_id").notNull(),

    // Personal information
    firstName: text("first_name").notNull(),
    lastName: text("last_name").notNull(),
    middleName: text("middle_name"),
    aliases: jsonb("aliases").default([]),
    dateOfBirth: timestamp("date_of_birth"),
    placeOfBirth: text("place_of_birth"),

    // Contact information
    address: text("address"),
    phone: text("phone"),
    email: text("email"),

    // Identification
    socialSecurityNumber: text("social_security_number"),
    driversLicense: text("drivers_license"),

    // Physical description
    height: integer("height"), // inches
    weight: integer("weight"), // pounds
    eyeColor: text("eye_color"),
    hairColor: text("hair_color"),
    distinguishingMarks: text("distinguishing_marks"),

    // Status and classification
    status: text("status").notNull(), // suspect, person_of_interest, witness, victim, defendant
    dangerLevel: text("danger_level").default("low"), // low, medium, high, extreme
    currentLocation: text("current_location"),
    knownAssociates: jsonb("known_associates").default([]),
    criminalHistory: jsonb("criminal_history").default([]),

    // Case association
    associatedCases: jsonb("associated_cases").default([]),
    notes: text("notes"),
    metadata: jsonb("metadata").default({}),

    // Vector embeddings
    profileEmbedding: vector("profile_embedding", { dimensions: 768 }),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    nameIdx: index("criminals_name_idx").on(table.firstName, table.lastName),
    statusIdx: index("criminals_status_idx").on(table.status),
    dangerLevelIdx: index("criminals_danger_level_idx").on(table.dangerLevel),
    profileEmbeddingIdx: index("criminals_profile_embedding_idx").on(table.profileEmbedding),
  }),
);

// Enhanced evidence connections for the board
export const evidenceConnections = pgTable(
  "evidence_connections",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    caseId: uuid("case_id").references(() => cases.id, { onDelete: "cascade" }),
    fromEvidenceId: uuid("from_evidence_id").references(() => evidence.id, {
      onDelete: "cascade",
    }),
    toEvidenceId: uuid("to_evidence_id").references(() => evidence.id, {
      onDelete: "cascade",
    }),
    connectionType: text("connection_type").notNull().default("related"), // related, contradicts, supports, timeline
    strength: real("strength").default(0.5), // 0-1 confidence
    description: text("description"),
    metadata: jsonb("metadata").default({}),
    createdBy: uuid("created_by").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => ({
    caseIdIdx: index("evidence_connections_case_id_idx").on(table.caseId),
    fromEvidenceIdx: index("evidence_connections_from_idx").on(
      table.fromEvidenceId,
    ),
    toEvidenceIdx: index("evidence_connections_to_idx").on(table.toEvidenceId),
  }),
);

// Vector search metadata
export const vectorMetadata = pgTable(
  "vector_metadata",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    documentId: uuid("document_id").notNull().unique(),
    documentType: text("document_type").notNull(), // case, evidence, criminal
    collectionName: text("collection_name").notNull(),
    metadata: jsonb("metadata").default({}),
    contentHash: text("content_hash").notNull(),
    embeddingModel: text("embedding_model").default("nomic-embed-text"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    documentIdIdx: index("vector_metadata_document_id_idx").on(
      table.documentId,
    ),
    documentTypeIdx: index("vector_metadata_document_type_idx").on(
      table.documentType,
    ),
    collectionIdx: index("vector_metadata_collection_idx").on(
      table.collectionName,
    ),
  }),
);

// Embedding cache for performance
export const embeddingCache = pgTable(
  "embedding_cache",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    textHash: text("text_hash").notNull().unique(),
    embedding: vector("embedding", { dimensions: 768 }).notNull(),
    model: text("model").notNull().default("nomic-embed-text"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => ({
    textHashIdx: index("embedding_cache_text_hash_idx").on(table.textHash),
    embeddingIdx: index("embedding_cache_embedding_idx").on(table.embedding),
  }),
);

// Conversation and message tables for AI chat
export const conversations = pgTable(
  "conversations",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: uuid("user_id").notNull(),
    title: text("title").notNull(),
    type: text("type").notNull().default("general"), // case_analysis, evidence_review, legal_research, general
    metadata: jsonb("metadata").default({}),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    userIdIdx: index("conversations_user_id_idx").on(table.userId),
    typeIdx: index("conversations_type_idx").on(table.type),
  }),
);

export const messages = pgTable(
  "messages",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    conversationId: uuid("conversation_id").references(() => conversations.id, {
      onDelete: "cascade",
    }),
    role: text("role").notNull(), // user, assistant, system
    content: text("content").notNull(),
    metadata: jsonb("metadata").default({}),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    conversationIdIdx: index("messages_conversation_id_idx").on(
      table.conversationId,
    ),
    roleIdx: index("messages_role_idx").on(table.role),
  }),
);

// User activity and audit logging
export const userActivity = pgTable(
  "user_activity",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: uuid("user_id").notNull(),
    action: text("action").notNull(), // create, update, delete, view, search
    resourceType: text("resource_type").notNull(), // case, evidence, criminal
    resourceId: uuid("resource_id"),
    details: jsonb("details").default({}),
    ipAddress: text("ip_address"),
    userAgent: text("user_agent"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => ({
    userIdIdx: index("user_activity_user_id_idx").on(table.userId),
    actionIdx: index("user_activity_action_idx").on(table.action),
    resourceTypeIdx: index("user_activity_resource_type_idx").on(
      table.resourceType,
    ),
    createdAtIdx: index("user_activity_created_at_idx").on(table.createdAt),
  }),
);

// System configuration for AI models and settings
export const systemConfig = pgTable(
  "system_config",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    key: text("key").notNull().unique(),
    value: jsonb("value").notNull(),
    description: text("description"),
    category: text("category").default("general"),
    isEncrypted: boolean("is_encrypted").default(false),
    updatedBy: uuid("updated_by").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => ({
    keyIdx: index("system_config_key_idx").on(table.key),
    categoryIdx: index("system_config_category_idx").on(table.category),
  }),
);

// Relations
export const casesRelations = relations(cases, ({ many }) => ({
  evidence: many(evidence),
  connections: many(evidenceConnections),
}));

export const evidenceRelations = relations(evidence, ({ one, many }) => ({
  case: one(cases, {
    fields: [evidence.caseId],
    references: [cases.id],
  }),
  connectionsFrom: many(evidenceConnections, { relationName: "from" }),
  connectionsTo: many(evidenceConnections, { relationName: "to" }),
}));

export const evidenceConnectionsRelations = relations(
  evidenceConnections,
  ({ one }) => ({
    case: one(cases, {
      fields: [evidenceConnections.caseId],
      references: [cases.id],
    }),
    fromEvidence: one(evidence, {
      fields: [evidenceConnections.fromEvidenceId],
      references: [evidence.id],
      relationName: "from",
    }),
    toEvidence: one(evidence, {
      fields: [evidenceConnections.toEvidenceId],
      references: [evidence.id],
      relationName: "to",
    }),
  }),
);

export const conversationsRelations = relations(conversations, ({ many }) => ({
  messages: many(messages),
}));

export const messagesRelations = relations(messages, ({ one }) => ({
  conversation: one(conversations, {
    fields: [messages.conversationId],
    references: [conversations.id],
  }),
}));

// Export all table types for TypeScript
export type Case = typeof cases.$inferSelect;
export type Evidence = typeof evidence.$inferSelect;
export type Criminal = typeof criminals.$inferSelect;
export type EvidenceConnection = typeof evidenceConnections.$inferSelect;
export type VectorMetadata = typeof vectorMetadata.$inferSelect;
export type EmbeddingCache = typeof embeddingCache.$inferSelect;
export type Conversation = typeof conversations.$inferSelect;
export type Message = typeof messages.$inferSelect;
export type UserActivity = typeof userActivity.$inferSelect;
export type SystemConfig = typeof systemConfig.$inferSelect;

// Insert types
export type NewCase = typeof cases.$inferInsert;
export type NewEvidence = typeof evidence.$inferInsert;
export type NewCriminal = typeof criminals.$inferInsert;
export type NewEvidenceConnection = typeof evidenceConnections.$inferInsert;
export type NewVectorMetadata = typeof vectorMetadata.$inferInsert;
export type NewEmbeddingCache = typeof embeddingCache.$inferInsert;
export type NewConversation = typeof conversations.$inferInsert;
export type NewMessage = typeof messages.$inferInsert;
export type NewUserActivity = typeof userActivity.$inferInsert;
export type NewSystemConfig = typeof systemConfig.$inferInsert;
