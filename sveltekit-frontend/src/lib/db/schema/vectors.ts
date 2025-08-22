
import {
  pgTable,
  text,
  timestamp,
  uuid,
  integer,
  jsonb,
  real,
  index,
  customType
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

// Custom pgvector type for Drizzle
const vector = customType<{ data: number[]; driverData: string }>({
  dataType() {
    return 'vector(384)'; // nomic-embed-text produces 384-dimensional vectors
  },
  toDriver(value: number[]): string {
    return JSON.stringify(value);
  },
  fromDriver(value: string): number[] {
    return typeof value === 'string' ? JSON.parse(value) : value;
  },
});

// Document vectors table for semantic search
export const documentVectors = pgTable("document_vectors", {
  id: uuid("id").primaryKey().defaultRandom(),
  documentId: uuid("document_id").references(() => documents.id).notNull(),
  chunkIndex: integer("chunk_index").notNull(),
  content: text("content").notNull(),
  embedding: vector("embedding").notNull(),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  embeddingIdx: index("document_vectors_embedding_idx").using("ivfflat", table.embedding),
  documentIdx: index("document_vectors_document_idx").on(table.documentId),
}));

// Case summary vectors for case-level search
export const caseSummaryVectors = pgTable("case_summary_vectors", {
  id: uuid("id").primaryKey().defaultRandom(),
  caseId: uuid("case_id").references(() => cases.id).notNull().unique(),
  summary: text("summary").notNull(),
  embedding: vector("embedding").notNull(),
  confidence: real("confidence").default(1.0),
  lastUpdated: timestamp("last_updated").defaultNow().notNull(),
}, (table) => ({
  embeddingIdx: index("case_summary_vectors_embedding_idx").using("ivfflat", table.embedding),
}));

// Evidence vectors for evidence search
export const evidenceVectors = pgTable("evidence_vectors", {
  id: uuid("id").primaryKey().defaultRandom(),
  evidenceId: uuid("evidence_id").references(() => evidence.id).notNull(),
  chunkIndex: integer("chunk_index").notNull(),
  content: text("content").notNull(),
  embedding: vector("embedding").notNull(),
  analysisType: text("analysis_type"), // summary, entities, sentiment, classification
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  embeddingIdx: index("evidence_vectors_embedding_idx").using("ivfflat", table.embedding),
  evidenceIdx: index("evidence_vectors_evidence_idx").on(table.evidenceId),
}));

// User query vectors for search history and recommendations
export const queryVectors = pgTable("query_vectors", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id).notNull(),
  query: text("query").notNull(),
  embedding: vector("embedding").notNull(),
  resultCount: integer("result_count").default(0),
  clickedResults: jsonb("clicked_results"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  embeddingIdx: index("query_vectors_embedding_idx").using("ivfflat", table.embedding),
  userIdx: index("query_vectors_user_idx").on(table.userId),
}));

// Knowledge graph nodes for recommendation system
export const knowledgeNodes = pgTable("knowledge_nodes", {
  id: uuid("id").primaryKey().defaultRandom(),
  nodeType: text("node_type").notNull(), // case, evidence, document, entity
  nodeId: uuid("node_id").notNull(),
  label: text("label").notNull(),
  embedding: vector("embedding").notNull(),
  properties: jsonb("properties"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
}, (table) => ({
  embeddingIdx: index("knowledge_nodes_embedding_idx").using("ivfflat", table.embedding),
  nodeTypeIdx: index("knowledge_nodes_type_idx").on(table.nodeType),
}));

// Knowledge graph edges for relationships
export const knowledgeEdges = pgTable("knowledge_edges", {
  id: uuid("id").primaryKey().defaultRandom(),
  sourceId: uuid("source_id").references(() => knowledgeNodes.id).notNull(),
  targetId: uuid("target_id").references(() => knowledgeNodes.id).notNull(),
  relationship: text("relationship").notNull(), // similar_to, relates_to, references, etc.
  weight: real("weight").default(1.0),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  sourceIdx: index("knowledge_edges_source_idx").on(table.sourceId),
  targetIdx: index("knowledge_edges_target_idx").on(table.targetId),
  relationshipIdx: index("knowledge_edges_relationship_idx").on(table.relationship),
}));

// Cached recommendations for users
export const recommendationCache = pgTable("recommendation_cache", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id).notNull(),
  recommendationType: text("recommendation_type").notNull(), // case, evidence, document
  recommendations: jsonb("recommendations").notNull(), // Array of recommended items
  score: real("score").default(0),
  expiresAt: timestamp("expires_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  userTypeIdx: index("recommendation_cache_user_type_idx").on(table.userId, table.recommendationType),
}));

// Import required types from main schema
import { documents, cases, evidence, users } from "../schema";

// Helper functions for vector operations
export const vectorOperations = {
  // Calculate cosine similarity
  cosineSimilarity: (embedding1: typeof sql.placeholder, embedding2: typeof sql.placeholder) => 
    sql`1 - (${embedding1} <=> ${embedding2})`,
  
  // Find nearest neighbors
  nearestNeighbors: (table: any, embedding: typeof sql.placeholder, limit: number = 10) =>
    sql`SELECT * FROM ${table} ORDER BY embedding <=> ${embedding} LIMIT ${limit}`,
  
  // Hybrid search combining vector and keyword search
  hybridSearch: (vectorScore: typeof sql.placeholder, textScore: typeof sql.placeholder, vectorWeight: number = 0.7) =>
    sql`(${vectorScore} * ${vectorWeight} + ${textScore} * ${1 - vectorWeight})`,
};

// Export types
export type DocumentVector = typeof documentVectors.$inferSelect;
export type NewDocumentVector = typeof documentVectors.$inferInsert;
export type CaseSummaryVector = typeof caseSummaryVectors.$inferSelect;
export type NewCaseSummaryVector = typeof caseSummaryVectors.$inferInsert;
export type EvidenceVector = typeof evidenceVectors.$inferSelect;
export type NewEvidenceVector = typeof evidenceVectors.$inferInsert;
export type QueryVector = typeof queryVectors.$inferSelect;
export type NewQueryVector = typeof queryVectors.$inferInsert;
export type KnowledgeNode = typeof knowledgeNodes.$inferSelect;
export type NewKnowledgeNode = typeof knowledgeNodes.$inferInsert;
export type KnowledgeEdge = typeof knowledgeEdges.$inferSelect;
export type NewKnowledgeEdge = typeof knowledgeEdges.$inferInsert;
export type RecommendationCache = typeof recommendationCache.$inferSelect;
export type NewRecommendationCache = typeof recommendationCache.$inferInsert;