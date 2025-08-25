import { pgTable, uuid, text, timestamp, jsonb, boolean, integer, decimal } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// AI History Schema for Legal AI Platform
export const aiSessions = pgTable('ai_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull(),
  sessionType: text('session_type').notNull(), // 'chat', 'analysis', 'search'
  metadata: jsonb('metadata').default('{}'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const aiInteractions = pgTable('ai_interactions', {
  id: uuid('id').primaryKey().defaultRandom(),
  sessionId: uuid('session_id').notNull(),
  userMessage: text('user_message'),
  aiResponse: text('ai_response'),
  responseTime: integer('response_time'), // milliseconds
  confidence: decimal('confidence', { precision: 5, scale: 4 }),
  metadata: jsonb('metadata').default('{}'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const aiAnalysisHistory = pgTable('ai_analysis_history', {
  id: uuid('id').primaryKey().defaultRandom(),
  documentId: uuid('document_id'),
  analysisType: text('analysis_type').notNull(),
  analysisResult: jsonb('analysis_result').notNull(),
  confidence: decimal('confidence', { precision: 5, scale: 4 }),
  processingTime: integer('processing_time'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// Relations
export const aiSessionsRelations = relations(aiSessions, ({ many }) => ({
  interactions: many(aiInteractions),
}));

export const aiInteractionsRelations = relations(aiInteractions, ({ one }) => ({
  session: one(aiSessions, {
    fields: [aiInteractions.sessionId],
    references: [aiSessions.id],
  }),
}));