// === MINIMAL COMPATIBLE SCHEMA ===
// This schema only includes tables that already exist in the database

import { 
  pgTable, 
  text, 
  varchar, 
  integer, 
  timestamp, 
  jsonb, 
  boolean,
  decimal,
  uuid,
  primaryKey
} from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// === EXISTING TABLES ===

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  emailVerified: timestamp('email_verified', { mode: 'date' }),
  hashedPassword: text('hashed_password'),
  role: varchar('role', { length: 50 }).default('prosecutor').notNull(),
  isActive: boolean('is_active').default(true).notNull(),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
  firstName: varchar('first_name', { length: 100 }),
  lastName: varchar('last_name', { length: 100 }),
  name: text('name'),
  username: varchar('username', { length: 100 }),
  title: varchar('title', { length: 100 }),
  department: varchar('department', { length: 200 }),
  phone: varchar('phone', { length: 20 }),
  officeAddress: text('office_address'),
  avatar: text('avatar'),
  image: text('image'),
  bio: text('bio'),
  specializations: jsonb('specializations').default([]).notNull(),
});

export const sessions = pgTable('sessions', {
  id: text('id').primaryKey(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: timestamp('expires_at', { mode: 'date' }).notNull(),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
});

export const criminals = pgTable('criminals', {
  id: uuid('id').primaryKey().defaultRandom(),
  firstName: varchar('first_name', { length: 100 }).notNull(),
  lastName: varchar('last_name', { length: 100 }).notNull(),
  middleName: varchar('middle_name', { length: 100 }),
  aliases: jsonb('aliases').default([]).notNull(),
  dateOfBirth: timestamp('date_of_birth', { mode: 'date' }),
  address: text('address'),
  phone: varchar('phone', { length: 20 }),
  email: varchar('email', { length: 255 }),
  height: integer('height'),
  weight: integer('weight'),
  eyeColor: varchar('eye_color', { length: 20 }),
  hairColor: varchar('hair_color', { length: 20 }),
  distinguishingMarks: text('distinguishing_marks'),
  photoUrl: text('photo_url'),
  threatLevel: varchar('threat_level', { length: 20 }).default('low').notNull(),
  status: varchar('status', { length: 20 }).default('active').notNull(),
  priors: jsonb('priors').default([]).notNull(),
  convictions: jsonb('convictions').default([]).notNull(),
  notes: text('notes'),
  aiSummary: text('ai_summary'),
  aiTags: jsonb('ai_tags').default([]).notNull(),
  convictionStatus: varchar('conviction_status', { length: 50 }),
  sentenceLength: varchar('sentence_length', { length: 100 }),
  convictionDate: timestamp('conviction_date', { mode: 'date' }),
  escapeAttempts: integer('escape_attempts').default(0),
  gangAffiliations: text('gang_affiliations'),
  aiAnalysis: jsonb('ai_analysis').default({}).notNull(),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
});

export const cases = pgTable('cases', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseNumber: varchar('case_number', { length: 50 }),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  incidentDate: timestamp('incident_date', { mode: 'date' }),
  location: text('location'),
  priority: varchar('priority', { length: 20 }).default('medium').notNull(),
  status: varchar('status', { length: 20 }).default('open').notNull(),
  category: varchar('category', { length: 50 }),
  dangerScore: integer('danger_score').default(0).notNull(),
  estimatedValue: decimal('estimated_value', { precision: 12, scale: 2 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  leadProsecutor: uuid('lead_prosecutor').references(() => users.id),
  assignedTeam: jsonb('assigned_team').default([]).notNull(),
  aiSummary: text('ai_summary'),
  aiTags: jsonb('ai_tags').default([]).notNull(),
  metadata: jsonb('metadata').default({}).notNull(),
  data: jsonb('data').default({}).notNull(),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
  closedAt: timestamp('closed_at', { mode: 'date' }),
  tags: jsonb('tags').default([]),
  embedding: jsonb('embedding'),
  name: varchar('name', { length: 255 }),
  summary: text('summary'),
  dateOpened: timestamp('date_opened', { mode: 'date' }),
  verdict: varchar('verdict', { length: 100 }),
  courtDates: text('court_dates'),
  linkedCriminals: text('linked_criminals'),
  linkedCrimes: text('linked_crimes'),
  notes: text('notes'),
});

export const evidence = pgTable('evidence', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  criminalId: uuid('criminal_id').references(() => criminals.id),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  evidenceType: varchar('evidence_type', { length: 50 }).notNull(),
  subType: varchar('sub_type', { length: 50 }),
  fileUrl: text('file_url'),
  fileName: varchar('file_name', { length: 255 }),
  filename: varchar('filename', { length: 255 }),
  fileType: varchar('file_type', { length: 100 }),
  fileSize: integer('file_size'),
  filePath: text('file_path'),
  mimeType: varchar('mime_type', { length: 100 }),
  hash: varchar('hash', { length: 128 }),
  chainOfCustody: jsonb('chain_of_custody').default([]).notNull(),
  collectedAt: timestamp('collected_at', { mode: 'date' }),
  collectedBy: varchar('collected_by', { length: 255 }),
  location: text('location'),
  labAnalysis: jsonb('lab_analysis').default({}).notNull(),
  aiAnalysis: jsonb('ai_analysis').default({}).notNull(),
  aiSummary: text('ai_summary'),
  aiTags: jsonb('ai_tags').default([]).notNull(),
  tags: jsonb('tags').default([]).notNull(),
  embedding: jsonb('embedding'),
  originalContent: text('original_content'),
  summary: text('summary'),
  poiId: uuid('poi_id'),
  isAdmissible: boolean('is_admissible').default(true).notNull(),
  confidentialityLevel: varchar('confidentiality_level', { length: 20 }).default('standard').notNull(),
  canvasPosition: jsonb('canvas_position').default({}).notNull(),
  uploadedBy: uuid('uploaded_by').references(() => users.id),
  uploadedAt: timestamp('uploaded_at', { mode: 'date' }).defaultNow().notNull(),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
});

export const statutes = pgTable('statutes', {
  id: uuid('id').primaryKey().defaultRandom(),
  code: varchar('code', { length: 50 }).notNull().unique(),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  fullText: text('full_text'),
  category: varchar('category', { length: 100 }),
  severity: varchar('severity', { length: 20 }),
  minPenalty: varchar('min_penalty', { length: 255 }),
  maxPenalty: varchar('max_penalty', { length: 255 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  effectiveDate: timestamp('effective_date', { mode: 'date' }),
  aiSummary: text('ai_summary'),
  tags: jsonb('tags').default([]).notNull(),
  relatedStatutes: jsonb('related_statutes').default([]).notNull(),
  name: varchar('name', { length: 255 }),
  sectionNumber: varchar('section_number', { length: 50 }),
  content: text('content'),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
});

export const crimes = pgTable('crimes', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade', onUpdate: 'cascade' }),
  criminalId: uuid('criminal_id').references(() => criminals.id, { onDelete: 'cascade', onUpdate: 'cascade' }),
  statuteId: uuid('statute_id').references(() => statutes.id, { onDelete: 'set null', onUpdate: 'cascade' }),
  name: text('name').notNull(),
  title: text('title'),
  description: text('description'),
  chargeLevel: varchar('charge_level', { length: 50 }),
  status: varchar('status', { length: 50 }).default('pending').notNull(),
  incidentDate: timestamp('incident_date', { mode: 'date' }),
  arrestDate: timestamp('arrest_date', { mode: 'date' }),
  filingDate: timestamp('filing_date', { mode: 'date' }),
  notes: text('notes'),
  aiSummary: text('ai_summary'),
  metadata: jsonb('metadata').default({}).notNull(),
  category: varchar('category', { length: 100 }),
  jurisdiction: varchar('jurisdiction', { length: 100 }),
  minPenalty: varchar('min_penalty', { length: 255 }),
  maxPenalty: varchar('max_penalty', { length: 255 }),
  severity: varchar('severity', { length: 50 }),
  relatedStatutes: jsonb('related_statutes').default([]).notNull(),
  createdBy: uuid('created_by').references(() => users.id, { onDelete: 'set null', onUpdate: 'cascade' }),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
});

export const caseActivities = pgTable('case_activities', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').notNull().references(() => cases.id, { onDelete: 'cascade' }),
  activityType: varchar('activity_type', { length: 50 }).notNull(),
  title: varchar('title', { length: 255 }).notNull(),
  description: text('description'),
  scheduledFor: timestamp('scheduled_for', { mode: 'date' }),
  completedAt: timestamp('completed_at', { mode: 'date' }),
  status: varchar('status', { length: 20 }).default('pending').notNull(),
  priority: varchar('priority', { length: 20 }).default('medium').notNull(),
  assignedTo: uuid('assigned_to').references(() => users.id),
  relatedEvidence: jsonb('related_evidence').default([]).notNull(),
  relatedCriminals: jsonb('related_criminals').default([]).notNull(),
  metadata: jsonb('metadata').default({}).notNull(),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
});

export const lawParagraphs = pgTable('law_paragraphs', {
  id: uuid('id').primaryKey().defaultRandom(),
  statuteId: uuid('statute_id').notNull().references(() => statutes.id, { onDelete: 'cascade', onUpdate: 'cascade' }),
  paragraphNumber: varchar('paragraph_number', { length: 50 }).notNull(),
  content: text('content').notNull(),
  aiSummary: text('ai_summary'),
  tags: jsonb('tags').default([]).notNull(),
  linkedCaseIds: jsonb('linked_case_ids').default([]).notNull(),
  crimeSuggestions: jsonb('crime_suggestions').default([]).notNull(),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { mode: 'date' }).defaultNow().notNull(),
  paragraphText: text('paragraph_text'),
  anchorId: uuid('anchor_id'),
});

export const contentEmbeddings = pgTable('content_embeddings', {
  id: uuid('id').primaryKey().defaultRandom(),
  entityId: uuid('entity_id').notNull(),
  entityType: varchar('entity_type', { length: 100 }).notNull(),
  contentType: varchar('content_type', { length: 100 }).notNull(),
  embedding: jsonb('embedding').notNull(),
  text: text('text').notNull(),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
});

export const caseLawLinks = pgTable('case_law_links', {
  id: uuid('id').primaryKey().defaultRandom(),
  caseId: uuid('case_id').references(() => cases.id, { onDelete: 'cascade' }),
  statuteId: uuid('statute_id').references(() => statutes.id, { onDelete: 'cascade' }),
  linkType: varchar('link_type', { length: 50 }).notNull(),
  confidence: decimal('confidence', { precision: 3, scale: 2 }),
  createdAt: timestamp('created_at', { mode: 'date' }).defaultNow().notNull(),
});

// Legacy auth tables
export const account = pgTable('account', {
  userId: uuid('userId').notNull().references(() => users.id, { onDelete: 'cascade' }),
  type: text('type').notNull(),
  provider: text('provider').notNull(),
  providerAccountId: text('providerAccountId').notNull(),
  refresh_token: text('refresh_token'),
  access_token: text('access_token'),
  expires_at: integer('expires_at'),
  token_type: text('token_type'),
  scope: text('scope'),
  id_token: text('id_token'),
  session_state: text('session_state'),
}, (table) => ({
  compoundKey: primaryKey({
    columns: [table.provider, table.providerAccountId],
  }),
}));

export const verificationToken = pgTable('verificationToken', {
  identifier: text('identifier').notNull(),
  token: text('token').notNull(),
  expires: timestamp('expires', { mode: 'date' }).notNull(),
}, (table) => ({
  compoundKey: primaryKey({ columns: [table.identifier, table.token] }),
}));

// === TYPE EXPORTS ===
import type { InferSelectModel, InferInsertModel } from 'drizzle-orm';

export type User = InferSelectModel<typeof users>;
export type NewUser = InferInsertModel<typeof users>;
export type Case = InferSelectModel<typeof cases>;
export type NewCase = InferInsertModel<typeof cases>;
export type Criminal = InferSelectModel<typeof criminals>;
export type NewCriminal = InferInsertModel<typeof criminals>;
export type Evidence = InferSelectModel<typeof evidence>;
export type NewEvidence = InferInsertModel<typeof evidence>;
export type Statute = InferSelectModel<typeof statutes>;
export type NewStatute = InferInsertModel<typeof statutes>;
export type Crime = InferSelectModel<typeof crimes>;
export type NewCrime = InferInsertModel<typeof crimes>;

// === SCHEMA EXPORT ===
export const schema = {
  users,
  sessions,
  criminals,
  cases,
  evidence,
  statutes,
  crimes,
  caseActivities,
  lawParagraphs,
  contentEmbeddings,
  caseLawLinks,
  account,
  verificationToken
};
