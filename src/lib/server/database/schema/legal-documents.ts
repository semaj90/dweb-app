// Legal Documents Database Schema
// Drizzle ORM schema for legal document management

import { pgTable, text, timestamp, jsonb, serial, boolean, integer, vector } from 'drizzle-orm/pg-core';

// Legal documents table
export const legalDocuments = pgTable('legal_documents', {
  id: serial('id').primaryKey(),
  title: text('title').notNull(),
  content: text('content').notNull(),
  documentType: text('document_type').notNull(),
  caseId: text('case_id'),
  clientId: text('client_id'),
  status: text('status').default('active'),
  tags: jsonb('tags').$type<string[]>().default([]),
  metadata: jsonb('metadata').$type<Record<string, any>>().default({}),
  embedding: vector('embedding', { dimensions: 384 }), // For semantic search
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  createdBy: text('created_by'),
  isDeleted: boolean('is_deleted').default(false),
  version: integer('version').default(1),
});

// Legal cases table
export const legalCases = pgTable('legal_cases', {
  id: serial('id').primaryKey(),
  title: text('title').notNull(),
  description: text('description'),
  clientName: text('client_name'),
  caseType: text('case_type').notNull(),
  status: text('status').default('active'),
  priority: text('priority').default('medium'),
  tags: jsonb('tags').$type<string[]>().default([]),
  metadata: jsonb('metadata').$type<Record<string, any>>().default({}),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  createdBy: text('created_by'),
  isDeleted: boolean('is_deleted').default(false),
});

// Document versions table for tracking changes
export const documentVersions = pgTable('document_versions', {
  id: serial('id').primaryKey(),
  documentId: integer('document_id').references(() => legalDocuments.id),
  version: integer('version').notNull(),
  content: text('content').notNull(),
  changes: jsonb('changes').$type<Record<string, any>>().default({}),
  createdAt: timestamp('created_at').defaultNow(),
  createdBy: text('created_by'),
});

// Document relationships table
export const documentRelationships = pgTable('document_relationships', {
  id: serial('id').primaryKey(),
  sourceDocumentId: integer('source_document_id').references(() => legalDocuments.id),
  targetDocumentId: integer('target_document_id').references(() => legalDocuments.id),
  relationshipType: text('relationship_type').notNull(), // 'reference', 'citation', 'amendment', etc.
  metadata: jsonb('metadata').$type<Record<string, any>>().default({}),
  createdAt: timestamp('created_at').defaultNow(),
});

// Additional tables for case document relationships
export const caseDocuments = pgTable('case_documents', {
  id: serial('id').primaryKey(),
  caseId: integer('case_id').references(() => legalCases.id),
  documentId: integer('document_id').references(() => legalDocuments.id),
  role: text('role').notNull(), // 'primary', 'evidence', 'reference', etc.
  createdAt: timestamp('created_at').defaultNow(),
});

// Legal entities (clients, lawyers, judges, etc.)
export const legalEntities = pgTable('legal_entities', {
  id: serial('id').primaryKey(),
  name: text('name').notNull(),
  type: text('type').notNull(), // 'client', 'lawyer', 'judge', 'firm', etc.
  contactInfo: jsonb('contact_info').$type<Record<string, any>>().default({}),
  metadata: jsonb('metadata').$type<Record<string, any>>().default({}),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
  isDeleted: boolean('is_deleted').default(false),
});

// Agent analysis cache for storing AI analysis results
export const agentAnalysisCache = pgTable('agent_analysis_cache', {
  id: serial('id').primaryKey(),
  documentId: integer('document_id').references(() => legalDocuments.id),
  agentType: text('agent_type').notNull(),
  analysis: jsonb('analysis').$type<Record<string, any>>().default({}),
  confidence: integer('confidence').default(0),
  createdAt: timestamp('created_at').defaultNow(),
  expiresAt: timestamp('expires_at'),
});

// Drizzle schema validation exports
export const insertLegalDocumentSchema = {
  title: 'string',
  content: 'string',
  documentType: 'string',
  caseId: 'string?',
  clientId: 'string?',
  status: 'string?',
  tags: 'array?',
  metadata: 'object?',
  createdBy: 'string?',
};

export const selectLegalDocumentSchema = {
  id: 'number',
  title: 'string',
  content: 'string',
  documentType: 'string',
  caseId: 'string?',
  clientId: 'string?',
  status: 'string',
  tags: 'array',
  metadata: 'object',
  embedding: 'object?',
  createdAt: 'date',
  updatedAt: 'date',
  createdBy: 'string?',
  isDeleted: 'boolean',
  version: 'number',
};

export const insertLegalCaseSchema = {
  title: 'string',
  description: 'string?',
  clientName: 'string?',
  caseType: 'string',
  status: 'string?',
  priority: 'string?',
  tags: 'array?',
  metadata: 'object?',
  createdBy: 'string?',
};

export const selectLegalCaseSchema = {
  id: 'number',
  title: 'string',
  description: 'string?',
  clientName: 'string?',
  caseType: 'string',
  status: 'string',
  priority: 'string',
  tags: 'array',
  metadata: 'object',
  createdAt: 'date',
  updatedAt: 'date',
  createdBy: 'string?',
  isDeleted: 'boolean',
};

export const insertLegalEntitySchema = {
  name: 'string',
  type: 'string',
  contactInfo: 'object?',
  metadata: 'object?',
};

export const selectLegalEntitySchema = {
  id: 'number',
  name: 'string',
  type: 'string',
  contactInfo: 'object',
  metadata: 'object',
  createdAt: 'date',
  updatedAt: 'date',
  isDeleted: 'boolean',
};

// Export types for use in other parts of the application
export type LegalDocument = typeof legalDocuments.$inferSelect;
export type NewLegalDocument = typeof legalDocuments.$inferInsert;
export type LegalCase = typeof legalCases.$inferSelect;
export type NewLegalCase = typeof legalCases.$inferInsert;
export type DocumentVersion = typeof documentVersions.$inferSelect;
export type NewDocumentVersion = typeof documentVersions.$inferInsert;
export type DocumentRelationship = typeof documentRelationships.$inferSelect;
export type NewDocumentRelationship = typeof documentRelationships.$inferInsert;
export type CaseDocument = typeof caseDocuments.$inferSelect;
export type NewCaseDocument = typeof caseDocuments.$inferInsert;
export type LegalEntity = typeof legalEntities.$inferSelect;
export type NewLegalEntity = typeof legalEntities.$inferInsert;
export type AgentAnalysisCache = typeof agentAnalysisCache.$inferSelect;
export type NewAgentAnalysisCache = typeof agentAnalysisCache.$inferInsert;