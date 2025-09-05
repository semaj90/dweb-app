import { pgTable, text, uuid, timestamp, integer, jsonb, vector, index } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';

// Document vectors table for RAG integration
export const documentVectors = pgTable('document_vectors', {
  id: uuid('id').primaryKey().defaultRandom(),
  documentId: uuid('document_id').notNull(),
  chunkIndex: integer('chunk_index').notNull(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 384 }).notNull(),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow()
}, (table) => ({
  embeddingIdx: index('document_vectors_embedding_idx').using('hnsw', table.embedding).with({ 'm': 16, 'ef_construction': 64 }),
  documentIdx: index('document_vectors_document_id_idx').on(table.documentId),
  createdAtIdx: index('document_vectors_created_at_idx').on(table.createdAt)
}));

// Query vectors for semantic search
export const queryVectors = pgTable('query_vectors', {
  id: uuid('id').primaryKey().defaultRandom(),
  query: text('query').notNull(),
  embedding: vector('embedding', { dimensions: 384 }).notNull(),
  sessionId: uuid('session_id'),
  userId: uuid('user_id'),
  createdAt: timestamp('created_at').notNull().defaultNow()
}, (table) => ({
  embeddingIdx: index('query_vectors_embedding_idx').using('hnsw', table.embedding).with({ 'm': 16, 'ef_construction': 64 }),
  sessionIdx: index('query_vectors_session_id_idx').on(table.sessionId),
  userIdx: index('query_vectors_user_id_idx').on(table.userId),
  createdAtIdx: index('query_vectors_created_at_idx').on(table.createdAt)
}));

// Schemas
export const insertDocumentVectorSchema = createInsertSchema(documentVectors);
export const selectDocumentVectorSchema = createSelectSchema(documentVectors);

export const insertQueryVectorSchema = createInsertSchema(queryVectors);
export const selectQueryVectorSchema = createSelectSchema(queryVectors);

// Types
export type DocumentVector = typeof documentVectors.$inferSelect;
export type NewDocumentVector = typeof documentVectors.$inferInsert;

export type QueryVector = typeof queryVectors.$inferSelect;
export type NewQueryVector = typeof queryVectors.$inferInsert;