// Minimal Drizzle schema recreated to align with current SQL migrations.
// Focus: legal_documents, document_sections, citations (vector & FK integrity for RAG).
// Extend incrementally with additional tables as needed.

import { pgTable, uuid, varchar, text, jsonb, timestamp, integer, vector, numeric } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';

// legal_documents (subset of columns required by current app routes & migrations)
export const legalDocuments = pgTable('legal_documents', {
	id: uuid().defaultRandom().primaryKey().notNull(),
	title: varchar({ length: 500 }).notNull(),
	documentType: varchar('document_type', { length: 50 }).notNull(),
	jurisdiction: varchar({ length: 100 }),
	court: varchar({ length: 200 }),
	citation: varchar({ length: 300 }),
	fullCitation: text('full_citation'),
	docketNumber: varchar('docket_number', { length: 100 }),
	dateDecided: timestamp('date_decided', { withTimezone: true, mode: 'string' }),
	datePublished: timestamp('date_published', { withTimezone: true, mode: 'string' }),
	fullText: text('full_text'),
	content: text(),
	summary: text(),
	headnotes: text(),
	keywords: jsonb().default([]),
	topics: jsonb().default([]),
	parties: jsonb().default({}),
	judges: jsonb().default([]),
	attorneys: jsonb().default({}),
	outcome: varchar({ length: 100 }),
	precedentialValue: varchar('precedential_value', { length: 50 }),
	url: text(),
	pdfUrl: text('pdf_url'),
	embedding: vector({ dimensions: 384 }), // added via 20250903_add_pgvector_extension_and_evidence_index.sql
	createdAt: timestamp('created_at', { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp('updated_at', { withTimezone: true, mode: 'string' }).defaultNow().notNull()
});

// document_sections (chunking for RAG)
export const documentSections = pgTable('document_sections', {
	id: uuid().defaultRandom().primaryKey().notNull(),
	documentId: uuid('document_id').notNull(),
	sectionNumber: integer('section_number').notNull(),
	title: varchar({ length: 500 }),
	content: text().notNull(),
	metadata: jsonb().default({}),
	embedding: vector({ dimensions: 384 }),
	createdAt: timestamp('created_at', { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp('updated_at', { withTimezone: true, mode: 'string' }).defaultNow().notNull()
});

// citations (with optional document linkage)
export const citations = pgTable('citations', {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid('case_id'),
	documentId: uuid('document_id'),
	citationType: varchar('citation_type', { length: 50 }).notNull(),
	relevanceScore: numeric('relevance_score', { precision: 3, scale: 2 }),
	pageNumber: integer('page_number'),
	pinpointCitation: varchar('pinpoint_citation', { length: 100 }),
	quotedText: text('quoted_text'),
	contextBefore: text('context_before'),
	contextAfter: text('context_after'),
	annotation: text(),
	legalPrinciple: text('legal_principle'),
	citationFormat: varchar('citation_format', { length: 20 }).default('bluebook'),
	formattedCitation: text('formatted_citation'),
	shepardsTreatment: varchar('shepards_treatment', { length: 50 }),
	isKeyAuthority: varchar('is_key_authority', { length: 5 }), // kept simple; could refactor to boolean
	createdBy: uuid('created_by'),
	createdAt: timestamp('created_at', { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp('updated_at', { withTimezone: true, mode: 'string' }).defaultNow().notNull()
});

// NOTE:
// - Relations & additional tables (users, cases, evidence, etc.) can be reintroduced as needed.
// - Keep this minimal to unblock vector search & RAG endpoints relying on these core entities.
// - When expanding, prefer matching existing column names from base migration to avoid drift.

export const schemaInfo = {
	version: 'minimal-rag-sync-1',
	createdAt: sql`now()`
};
