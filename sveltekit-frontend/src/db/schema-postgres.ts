/**
 * Database Schema for PostgreSQL with pgvector
 */

import { pgTable, text, uuid, timestamp, jsonb, vector, integer, boolean, decimal, varchar } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

// Content embeddings table with pgvector
export const contentEmbeddings = pgTable('content_embeddings', {
  id: uuid('id').defaultRandom().primaryKey(),
  contentId: uuid('content_id').notNull(),
  contentType: text('content_type').notNull(),
  textContent: text('text_content').notNull(),
  embedding: vector('embedding', { dimensions: 768 }),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Search queries table
export const searchQueries = pgTable('search_queries', {
  id: uuid('id').defaultRandom().primaryKey(),
  query: text('query').notNull(),
  queryEmbedding: vector('query_embedding', { dimensions: 768 }),
  results: jsonb('results'),
  searchType: text('search_type'),
  resultCount: integer('result_count'),
  createdAt: timestamp('created_at').defaultNow()
});

// Documents table
export const documents = pgTable('documents', {
  id: uuid('id').defaultRandom().primaryKey(),
  title: text('title').notNull(),
  documentType: text('document_type'),
  content: text('content'),
  embedding: vector('embedding', { dimensions: 768 }),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Embeddings table (for general embeddings)
export const embeddings = pgTable('embeddings', {
  id: uuid('id').defaultRandom().primaryKey(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 768 }),
  model: text('model'),
  metadata: jsonb('metadata'),
  documentId: uuid('document_id'),
  createdAt: timestamp('created_at').defaultNow()
});

// Legal cases table
export const legalCases = pgTable('legal_cases', {
  id: uuid('id').defaultRandom().primaryKey(),
  title: text('title').notNull(),
  documentType: text('document_type'),
  jurisdiction: text('jurisdiction'),
  court: text('court'),
  citation: text('citation'),
  fullCitation: text('full_citation'),
  docketNumber: text('docket_number'),
  dateDecided: timestamp('date_decided'),
  datePublished: timestamp('date_published'),
  summary: text('summary'),
  content: text('content'),
  embedding: vector('embedding', { dimensions: 768 }),
  metadata: jsonb('metadata'),
  tags: jsonb('tags'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

// Chat messages table
export const chatMessages = pgTable('chat_messages', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: uuid('user_id').notNull(),
  sessionId: uuid('session_id').notNull(),
  role: text('role').notNull(), // 'user' | 'assistant' | 'system'
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 768 }),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at').defaultNow()
});

// GPU processing jobs table
export const gpuJobs = pgTable('gpu_jobs', {
  id: uuid('id').defaultRandom().primaryKey(),
  jobType: text('job_type').notNull(),
  status: text('status').notNull(), // 'pending' | 'processing' | 'completed' | 'failed'
  input: jsonb('input'),
  output: jsonb('output'),
  error: text('error'),
  gpuId: integer('gpu_id'),
  processingTime: decimal('processing_time'),
  createdAt: timestamp('created_at').defaultNow(),
  completedAt: timestamp('completed_at')
});

// Export types
export type ContentEmbedding = typeof contentEmbeddings.$inferSelect;
export type NewContentEmbedding = typeof contentEmbeddings.$inferInsert;
export type SearchQuery = typeof searchQueries.$inferSelect;
export type Document = typeof documents.$inferSelect;
export type Embedding = typeof embeddings.$inferSelect;
export type LegalCase = typeof legalCases.$inferSelect;
export type ChatMessage = typeof chatMessages.$inferSelect;
export type GpuJob = typeof gpuJobs.$inferSelect;
