/**
 * AI Enrichment Jobs Schema
 * Extends the existing schema with AI processing job tracking
 */

import {
  pgTable,
  text,
  uuid,
  timestamp,
  jsonb,
  pgEnum
} from 'drizzle-orm/pg-core';

// Enums for job types and priorities
export const enrichmentJobTypeEnum = pgEnum('enrichment_job_type', [
  'case_analysis',
  'evidence_processing', 
  'document_embedding',
  'legal_research',
  'precedent_matching',
  'risk_assessment'
]);

export const enrichmentPriorityEnum = pgEnum('enrichment_priority', [
  'low',
  'medium', 
  'high',
  'urgent'
]);

export const enrichmentStatusEnum = pgEnum('enrichment_status', [
  'queued',
  'processing',
  'completed',
  'failed',
  'cancelled'
]);

export const enrichmentEntityTypeEnum = pgEnum('enrichment_entity_type', [
  'case',
  'evidence',
  'document',
  'user',
  'precedent'
]);

// AI Enrichment Jobs table
export const aiEnrichmentJobs = pgTable('ai_enrichment_jobs', {
  id: uuid('id').defaultRandom().primaryKey(),
  type: enrichmentJobTypeEnum('type').notNull(),
  entityId: uuid('entity_id').notNull(),
  entityType: enrichmentEntityTypeEnum('entity_type').notNull(),
  priority: enrichmentPriorityEnum('priority').default('medium').notNull(),
  status: enrichmentStatusEnum('status').default('queued').notNull(),
  
  // Job metadata and configuration
  metadata: jsonb('metadata').default({}).notNull(),
  config: jsonb('config').default({}).notNull(),
  
  // Processing details
  processingStartedAt: timestamp('processing_started_at', { withTimezone: true }),
  processedAt: timestamp('processed_at', { withTimezone: true }),
  
  // Results and errors
  results: jsonb('results'),
  error: text('error'),
  errorDetails: jsonb('error_details'),
  
  // Tracking
  attempts: text('attempts').default('0'),
  maxAttempts: text('max_attempts').default('3'),
  
  // Agent information
  processedBy: text('processed_by'), // Claude agent identifier
  processingNode: text('processing_node'), // Which server/worker processed it
  
  // Timestamps
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull()
});

// AI Processing Sessions - Track longer running AI workflows
export const aiProcessingSessions = pgTable('ai_processing_sessions', {
  id: uuid('id').defaultRandom().primaryKey(),
  name: text('name').notNull(),
  description: text('description'),
  
  // Session configuration
  workflow: text('workflow').notNull(), // 'case_full_analysis', 'evidence_batch_processing', etc.
  config: jsonb('config').default({}).notNull(),
  
  // Progress tracking
  totalJobs: text('total_jobs').default('0'),
  completedJobs: text('completed_jobs').default('0'),
  failedJobs: text('failed_jobs').default('0'),
  
  // Status and results
  status: enrichmentStatusEnum('status').default('queued').notNull(),
  results: jsonb('results'),
  summary: text('summary'),
  
  // User context
  userId: uuid('user_id'),
  
  // Timestamps
  startedAt: timestamp('started_at', { withTimezone: true }),
  completedAt: timestamp('completed_at', { withTimezone: true }),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull()
});

// AI Agent Metrics - Track performance and usage
export const aiAgentMetrics = pgTable('ai_agent_metrics', {
  id: uuid('id').defaultRandom().primaryKey(),
  agentId: text('agent_id').notNull(),
  agentType: text('agent_type').notNull(), // 'claude', 'ollama', 'custom'
  
  // Performance metrics
  jobType: enrichmentJobTypeEnum('job_type').notNull(),
  processingTimeMs: text('processing_time_ms'),
  tokensUsed: text('tokens_used'),
  cost: text('cost'), // In whatever unit/currency
  
  // Quality metrics
  successRate: text('success_rate'),
  qualityScore: text('quality_score'), // 0-1 based on user feedback
  
  // Context
  entityType: enrichmentEntityTypeEnum('entity_type'),
  metadata: jsonb('metadata').default({}).notNull(),
  
  // Timestamps
  processedAt: timestamp('processed_at', { withTimezone: true }).defaultNow().notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull()
});

// Export types for TypeScript
export type AIEnrichmentJob = typeof aiEnrichmentJobs.$inferSelect;
export type NewAIEnrichmentJob = typeof aiEnrichmentJobs.$inferInsert;
export type AIProcessingSession = typeof aiProcessingSessions.$inferSelect;
export type NewAIProcessingSession = typeof aiProcessingSessions.$inferInsert;
export type AIAgentMetric = typeof aiAgentMetrics.$inferSelect;
export type NewAIAgentMetric = typeof aiAgentMetrics.$inferInsert;