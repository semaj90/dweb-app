// @ts-nocheck
// Extension to unified-schema.ts for vector search capabilities
import {
  pgTable,
  index,
  text,
  uuid,
  timestamp,
  jsonb,
  integer,
  real,
} from "drizzle-orm/pg-core";
import { vector } from "pgvector/drizzle-orm";
import { relations } from "drizzle-orm";

// Document embeddings for semantic search
export const documentEmbeddings = pgTable(
  "document_embeddings",
  {
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
    documentId: text("document_id").notNull(),
    documentType: text("document_type").notNull(), // 'case', 'evidence', 'note', 'report'
    chunkIndex: integer("chunk_index").notNull().default(0),
    chunkText: text("chunk_text").notNull(),
    embedding: vector("embedding", { dimensions: 768 }), // For nomic-embed-text
    metadata: jsonb("metadata")
      .$type<{
        pageNumber?: number;
        section?: string;
        highlighted?: boolean;
        confidence?: number;
        [key: string]: any;
      }>()
      .default({}),
    modelUsed: text("model_used").default("nomic-embed-text"),
    createdAt: timestamp("created_at").defaultNow(),
    updatedAt: timestamp("updated_at").defaultNow(),
  },
  (table) => ({
    // Vector similarity search index
    embeddingIdx: index("idx_embedding_ivfflat").on(table.embedding),
    // Regular indexes
    documentIdx: index("idx_document_lookup").on(
      table.documentId,
      table.documentType,
    ),
    createdAtIdx: index("idx_created_at").on(table.createdAt),
  }),
);

// Search queries and their embeddings for caching
export const searchQueries = pgTable(
  "search_queries",
  {
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
    userId: text("user_id").notNull(),
    queryText: text("query_text").notNull(),
    queryEmbedding: vector("query_embedding", { dimensions: 768 }),
    searchType: text("search_type").notNull().default("semantic"), // 'semantic', 'keyword', 'hybrid'
    resultsCount: integer("results_count").default(0),
    results: jsonb("results")
      .$type<{
        items: Array<{
          documentId: string;
          documentType: string;
          similarity: number;
          snippet: string;
          metadata?: any;
        }>;
        totalFound: number;
        searchTime: number;
      }>()
      .default({ items: [], totalFound: 0, searchTime: 0 }),
    createdAt: timestamp("created_at").defaultNow(),
  },
  (table) => ({
    userIdx: index("idx_search_user").on(table.userId),
    queryEmbeddingIdx: index("idx_query_embedding").on(table.queryEmbedding),
    createdAtIdx: index("idx_search_created").on(table.createdAt),
  }),
);

// AI model configurations
export const aiModels = pgTable(
  "ai_models",
  {
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
    name: text("name").notNull().unique(),
    provider: text("provider").notNull(), // 'ollama', 'openai', 'anthropic'
    modelType: text("model_type").notNull(), // 'embedding', 'chat', 'completion'
    embeddingDimensions: integer("embedding_dimensions"),
    contextLength: integer("context_length"),
    config: jsonb("config")
      .$type<{
        baseUrl?: string;
        apiKey?: string;
        temperature?: number;
        maxTokens?: number;
        [key: string]: any;
      }>()
      .default({}),
    isActive: integer("is_active").notNull().default(1),
    createdAt: timestamp("created_at").defaultNow(),
    updatedAt: timestamp("updated_at").defaultNow(),
  },
  (table) => ({
    nameIdx: index("idx_model_name").on(table.name),
    providerTypeIdx: index("idx_provider_type").on(
      table.provider,
      table.modelType,
    ),
  }),
);

// Relations for vector tables
export const documentEmbeddingsRelations = relations(
  documentEmbeddings,
  ({ one }) => ({
    // Relations to main tables can be added here based on documentId and documentType
  }),
);

export const searchQueriesRelations = relations(searchQueries, ({ one }) => ({
  // User relation can be added here
}));

// Export types
export type DocumentEmbedding = typeof documentEmbeddings.$inferSelect;
export type NewDocumentEmbedding = typeof documentEmbeddings.$inferInsert;
export type SearchQuery = typeof searchQueries.$inferSelect;
export type NewSearchQuery = typeof searchQueries.$inferInsert;
export type AIModel = typeof aiModels.$inferSelect;
export type NewAIModel = typeof aiModels.$inferInsert;
