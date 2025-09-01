import { pgTable, index, uuid, varchar, text, boolean, jsonb, timestamp, foreignKey, integer, numeric, vector, unique, inet, serial, bigint } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"



export const reports = pgTable("reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	title: varchar({ length: 255 }).notNull(),
	content: text(),
	reportType: varchar("report_type", { length: 50 }).default('case_summary'),
	status: varchar({ length: 50 }).default('draft').notNull(),
	isPublic: boolean("is_public").default(false),
	tags: jsonb().default([]).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	index("idx_reports_case_id").using("btree", table.caseId.asc().nullsLast().op("uuid_ops")),
	index("idx_reports_type").using("btree", table.reportType.asc().nullsLast().op("text_ops")),
]);

export const citations = pgTable("citations", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	caseId: uuid("case_id"),
	citationText: text("citation_text").notNull(),
	citationType: varchar("citation_type", { length: 100 }),
	source: varchar({ length: 500 }),
	pageNumber: integer("page_number"),
	relevanceScore: numeric("relevance_score", { precision: 3, scale:  2 }),
	context: text(),
	verified: boolean().default(false),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	updatedAt: timestamp("updated_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	metadata: jsonb().default({}),
}, (table) => [
	index("idx_citations_case_id").using("btree", table.caseId.asc().nullsLast().op("uuid_ops")),
	index("idx_citations_relevance").using("btree", table.relevanceScore.asc().nullsLast().op("numeric_ops")),
	index("idx_citations_text_fts").using("gin", sql`to_tsvector('english'::regconfig, citation_text)`),
	index("idx_citations_type").using("btree", table.citationType.asc().nullsLast().op("text_ops")),
	foreignKey({
			columns: [table.caseId],
			foreignColumns: [cases.id],
			name: "citations_case_id_fkey"
		}).onDelete("cascade"),
]);

export const documentVectors = pgTable("document_vectors", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	documentId: uuid("document_id").notNull(),
	documentType: varchar("document_type", { length: 50 }).notNull(),
	chunkIndex: integer("chunk_index").default(0),
	content: text().notNull(),
	embedding: vector({ dimensions: 768 }),
	metadata: jsonb().default({}),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => [
	index("idx_document_vectors_document_id").using("btree", table.documentId.asc().nullsLast().op("uuid_ops")),
	index("idx_document_vectors_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
	index("idx_document_vectors_type").using("btree", table.documentType.asc().nullsLast().op("text_ops")),
]);

export const userSessions = pgTable("user_sessions", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	userId: uuid("user_id"),
	sessionToken: varchar("session_token", { length: 255 }).notNull(),
	expiresAt: timestamp("expires_at", { mode: 'string' }).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	ipAddress: inet("ip_address"),
	userAgent: text("user_agent"),
}, (table) => [
	index("idx_user_sessions_expires").using("btree", table.expiresAt.asc().nullsLast().op("timestamp_ops")),
	index("idx_user_sessions_token").using("btree", table.sessionToken.asc().nullsLast().op("text_ops")),
	index("idx_user_sessions_user_id").using("btree", table.userId.asc().nullsLast().op("uuid_ops")),
	foreignKey({
			columns: [table.userId],
			foreignColumns: [users.id],
			name: "user_sessions_user_id_fkey"
		}).onDelete("cascade"),
	unique("user_sessions_session_token_key").on(table.sessionToken),
]);

export const activityLogs = pgTable("activity_logs", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	userId: uuid("user_id"),
	action: varchar({ length: 100 }).notNull(),
	entityType: varchar("entity_type", { length: 50 }),
	entityId: uuid("entity_id"),
	details: jsonb().default({}),
	ipAddress: inet("ip_address"),
	userAgent: text("user_agent"),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => [
	index("idx_activity_logs_action").using("btree", table.action.asc().nullsLast().op("text_ops")),
	index("idx_activity_logs_created_at").using("btree", table.createdAt.asc().nullsLast().op("timestamp_ops")),
	index("idx_activity_logs_user_id").using("btree", table.userId.asc().nullsLast().op("uuid_ops")),
	foreignKey({
			columns: [table.userId],
			foreignColumns: [users.id],
			name: "activity_logs_user_id_fkey"
		}).onDelete("cascade"),
]);

export const searchCache = pgTable("search_cache", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	queryHash: varchar("query_hash", { length: 64 }).notNull(),
	queryText: text("query_text").notNull(),
	results: jsonb().notNull(),
	expiresAt: timestamp("expires_at", { mode: 'string' }).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => [
	unique("search_cache_query_hash_key").on(table.queryHash),
]);

export const autoTags = pgTable("auto_tags", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	entityId: uuid("entity_id").notNull(),
	entityType: varchar("entity_type", { length: 50 }).notNull(),
	tag: varchar({ length: 100 }).notNull(),
	confidence: numeric({ precision: 3, scale:  2 }).notNull(),
	source: varchar({ length: 50 }).default('ai_analysis').notNull(),
	model: varchar({ length: 100 }),
	extractedAt: timestamp("extracted_at", { mode: 'string' }).defaultNow().notNull(),
	isConfirmed: boolean("is_confirmed").default(false).notNull(),
	confirmedBy: uuid("confirmed_by"),
	confirmedAt: timestamp("confirmed_at", { mode: 'string' }),
});

export const vectorMetadata = pgTable("vector_metadata", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	documentId: uuid("document_id").notNull(),
	collectionName: varchar("collection_name", { length: 100 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	contentHash: text("content_hash").notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow(),
}, (table) => [
	index("vector_metadata_document_id_idx").using("btree", table.documentId.asc().nullsLast().op("uuid_ops")),
]);

export const legalDocuments = pgTable("legal_documents", {
	id: serial().primaryKey().notNull(),
	filename: varchar({ length: 255 }).notNull(),
	originalPath: text("original_path"),
	s3Bucket: varchar("s3_bucket", { length: 100 }),
	s3Key: text("s3_key"),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	fileSize: bigint("file_size", { mode: "number" }),
	mimeType: varchar("mime_type", { length: 100 }),
	uploadDate: timestamp("upload_date", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	documentType: varchar("document_type", { length: 50 }),
	title: text(),
	contentPreview: text("content_preview"),
	fullText: text("full_text"),
	metadata: jsonb(),
	processingStatus: varchar("processing_status", { length: 20 }).default('uploaded'),
	errorMessage: text("error_message"),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	updatedAt: timestamp("updated_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
});

export const users = pgTable("users", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	email: varchar({ length: 255 }).notNull(),
	hashedPassword: varchar("hashed_password", { length: 255 }),
	username: varchar({ length: 100 }),
	firstName: varchar("first_name", { length: 100 }),
	lastName: varchar("last_name", { length: 100 }),
	role: varchar({ length: 50 }).default('user').notNull(),
	department: varchar({ length: 100 }),
	jurisdiction: varchar({ length: 100 }),
	permissions: jsonb().default([]).notNull(),
	isActive: boolean("is_active").default(true).notNull(),
	emailVerified: boolean("email_verified").default(false).notNull(),
	avatarUrl: varchar("avatar_url", { length: 500 }),
	lastLoginAt: timestamp("last_login_at", { withTimezone: true, mode: 'string' }),
	practiceAreas: jsonb("practice_areas").default([]),
	barNumber: varchar("bar_number", { length: 50 }),
	firmName: varchar("firm_name", { length: 200 }),
	profileEmbedding: vector("profile_embedding", { dimensions: 384 }),
	metadata: jsonb().default({}),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	deletedAt: timestamp("deleted_at", { withTimezone: true, mode: 'string' }),
}, (table) => [
	index("users_active_idx").using("btree", table.isActive.asc().nullsLast().op("bool_ops")),
	index("users_email_idx").using("btree", table.email.asc().nullsLast().op("text_ops")),
	index("users_role_idx").using("btree", table.role.asc().nullsLast().op("text_ops")),
	index("users_username_idx").using("btree", table.username.asc().nullsLast().op("text_ops")),
]);

export const sessions = pgTable("sessions", {
	id: varchar({ length: 255 }).primaryKey().notNull(),
	userId: uuid("user_id").notNull(),
	expiresAt: timestamp("expires_at", { withTimezone: true, mode: 'string' }).notNull(),
	ipAddress: varchar("ip_address", { length: 45 }),
	userAgent: text("user_agent"),
	sessionContext: jsonb("session_context").default({}),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	index("sessions_expires_at_idx").using("btree", table.expiresAt.asc().nullsLast().op("timestamptz_ops")),
	index("sessions_user_id_idx").using("btree", table.userId.asc().nullsLast().op("uuid_ops")),
]);

export const evidence = pgTable("evidence", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	title: varchar({ length: 255 }).notNull(),
	description: text(),
	evidenceType: varchar("evidence_type", { length: 50 }).notNull(),
	fileUrl: text("file_url"),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	userId: uuid("user_id"),
	titleEmbedding: vector("title_embedding", { dimensions: 384 }),
	contentEmbedding: vector("content_embedding", { dimensions: 384 }),
	subType: varchar("sub_type", { length: 50 }),
	fileName: varchar("file_name", { length: 255 }),
	fileSize: integer("file_size"),
	mimeType: varchar("mime_type", { length: 100 }),
	hash: varchar({ length: 128 }),
	collectedAt: timestamp("collected_at", { mode: 'string' }),
	collectedBy: varchar("collected_by", { length: 255 }),
	location: varchar({ length: 255 }),
	chainOfCustody: jsonb("chain_of_custody").default([]),
	tags: jsonb().default([]).notNull(),
	isAdmissible: boolean("is_admissible").default(true),
	confidentialityLevel: varchar("confidentiality_level", { length: 50 }).default('internal'),
	aiAnalysis: jsonb("ai_analysis").default({}),
	aiTags: jsonb("ai_tags").default([]),
	aiSummary: text("ai_summary"),
	summary: text(),
	summaryType: varchar("summary_type", { length: 50 }),
	boardPosition: jsonb("board_position").default({}),
}, (table) => [
	index("idx_evidence_case_id").using("btree", table.caseId.asc().nullsLast().op("uuid_ops")),
	index("idx_evidence_tags").using("gin", table.tags.asc().nullsLast().op("jsonb_ops")),
	index("idx_evidence_title_fts").using("gin", sql`to_tsvector('english'::regconfig, (title)::text)`),
	index("idx_evidence_type").using("btree", table.evidenceType.asc().nullsLast().op("text_ops")),
]);

export const cases = pgTable("cases", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	title: varchar({ length: 500 }).notNull(),
	description: text(),
	caseNumber: varchar("case_number", { length: 100 }),
	status: varchar({ length: 50 }).default('active').notNull(),
	priority: varchar({ length: 20 }).default('medium').notNull(),
	practiceArea: varchar("practice_area", { length: 100 }),
	jurisdiction: varchar({ length: 100 }),
	court: varchar({ length: 200 }),
	clientName: varchar("client_name", { length: 200 }),
	opposingParty: varchar("opposing_party", { length: 200 }),
	assignedAttorney: uuid("assigned_attorney"),
	filingDate: timestamp("filing_date", { withTimezone: true, mode: 'string' }),
	dueDate: timestamp("due_date", { withTimezone: true, mode: 'string' }),
	closedDate: timestamp("closed_date", { withTimezone: true, mode: 'string' }),
	caseEmbedding: vector("case_embedding", { dimensions: 384 }),
	qdrantId: uuid("qdrant_id"),
	qdrantCollection: varchar("qdrant_collection", { length: 100 }).default('cases'),
	metadata: jsonb().default({}),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	index("idx_cases_created_at").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_cases_description_fts").using("gin", sql`to_tsvector('english'::regconfig, description)`),
	index("idx_cases_metadata").using("gin", table.metadata.asc().nullsLast().op("jsonb_ops")),
	index("idx_cases_status").using("btree", table.status.asc().nullsLast().op("text_ops")),
	index("idx_cases_title_fts").using("gin", sql`to_tsvector('english'::regconfig, (title)::text)`),
]);

export const canvasStates = pgTable("canvas_states", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	name: varchar({ length: 255 }).notNull(),
	canvasData: jsonb("canvas_data").notNull(),
	version: integer().default(1),
	isDefault: boolean("is_default").default(false),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const caseScores = pgTable("case_scores", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id").notNull(),
	score: numeric({ precision: 5, scale:  2 }).notNull(),
	riskLevel: varchar("risk_level", { length: 20 }).notNull(),
	breakdown: jsonb().default({}).notNull(),
	criteria: jsonb().default({}).notNull(),
	recommendations: jsonb().default([]).notNull(),
	calculatedBy: uuid("calculated_by"),
	calculatedAt: timestamp("calculated_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const criminals = pgTable("criminals", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	firstName: varchar("first_name", { length: 100 }).notNull(),
	lastName: varchar("last_name", { length: 100 }).notNull(),
	middleName: varchar("middle_name", { length: 100 }),
	aliases: jsonb().default([]).notNull(),
	dateOfBirth: timestamp("date_of_birth", { mode: 'string' }),
	placeOfBirth: varchar("place_of_birth", { length: 200 }),
	address: text(),
	phone: varchar({ length: 20 }),
	email: varchar({ length: 255 }),
	ssn: varchar({ length: 11 }),
	driversLicense: varchar("drivers_license", { length: 50 }),
	height: integer(),
	weight: integer(),
	eyeColor: varchar("eye_color", { length: 20 }),
	hairColor: varchar("hair_color", { length: 20 }),
	distinguishingMarks: text("distinguishing_marks"),
	photoUrl: text("photo_url"),
	fingerprints: jsonb().default({}),
	threatLevel: varchar("threat_level", { length: 20 }).default('low').notNull(),
	status: varchar({ length: 20 }).default('active').notNull(),
	notes: text(),
	aiSummary: text("ai_summary"),
	aiTags: jsonb("ai_tags").default([]).notNull(),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const documentChunks = pgTable("document_chunks", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	documentId: uuid("document_id").notNull(),
	documentType: varchar("document_type", { length: 50 }).notNull(),
	chunkIndex: integer("chunk_index").notNull(),
	content: text().notNull(),
	embedding: vector({ dimensions: 768 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const embeddingCache = pgTable("embedding_cache", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	textHash: text("text_hash").notNull(),
	embedding: vector({ dimensions: 768 }).notNull(),
	model: varchar({ length: 100 }).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const legalAnalysisSessions = pgTable("legal_analysis_sessions", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	userId: uuid("user_id"),
	sessionType: varchar("session_type", { length: 50 }).default('case_analysis'),
	analysisPrompt: text("analysis_prompt"),
	analysisResult: text("analysis_result"),
	confidenceLevel: numeric("confidence_level", { precision: 3, scale:  2 }),
	sourcesUsed: jsonb("sources_used").default([]).notNull(),
	model: varchar({ length: 100 }).default('gemma3-legal'),
	processingTime: integer("processing_time"),
	isActive: boolean("is_active").default(true),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const keys = pgTable("keys", {
	id: varchar({ length: 255 }).primaryKey().notNull(),
	userId: uuid("user_id").notNull(),
	hashedPassword: varchar("hashed_password", { length: 255 }),
	providerId: varchar("provider_id", { length: 255 }),
	providerUserId: varchar("provider_user_id", { length: 255 }),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	foreignKey({
			columns: [table.userId],
			foreignColumns: [users.id],
			name: "keys_user_id_users_id_fk"
		}).onDelete("cascade"),
]);

export const personsOfInterest = pgTable("persons_of_interest", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	name: varchar({ length: 255 }).notNull(),
	aliases: jsonb().default([]).notNull(),
	relationship: varchar({ length: 100 }),
	threatLevel: varchar("threat_level", { length: 20 }).default('low'),
	status: varchar({ length: 20 }).default('active'),
	profileData: jsonb("profile_data").default({}).notNull(),
	tags: jsonb().default([]).notNull(),
	position: jsonb().default({}).notNull(),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const ragMessages = pgTable("rag_messages", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	sessionId: varchar("session_id", { length: 255 }).notNull(),
	messageIndex: integer("message_index").notNull(),
	role: varchar({ length: 20 }).notNull(),
	content: text().notNull(),
	retrievedSources: jsonb("retrieved_sources").default([]).notNull(),
	sourceCount: integer("source_count").default(0).notNull(),
	retrievalScore: varchar("retrieval_score", { length: 10 }),
	processingTime: integer("processing_time"),
	model: varchar({ length: 100 }),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const ragSessions = pgTable("rag_sessions", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	sessionId: varchar("session_id", { length: 255 }).notNull(),
	userId: uuid("user_id"),
	title: varchar({ length: 255 }),
	model: varchar({ length: 100 }),
	isActive: boolean("is_active").default(true).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const statutes = pgTable("statutes", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	title: varchar({ length: 255 }).notNull(),
	code: varchar({ length: 100 }).notNull(),
	description: text(),
	category: varchar({ length: 100 }),
	jurisdiction: varchar({ length: 100 }),
	isActive: boolean("is_active").default(true),
	penalties: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const userAiQueries = pgTable("user_ai_queries", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	userId: uuid("user_id").notNull(),
	caseId: uuid("case_id"),
	query: text().notNull(),
	response: text().notNull(),
	model: varchar({ length: 100 }).default('gemma3-legal').notNull(),
	queryType: varchar("query_type", { length: 50 }).default('general'),
	confidence: numeric({ precision: 3, scale:  2 }),
	tokensUsed: integer("tokens_used"),
	processingTime: integer("processing_time"),
	contextUsed: jsonb("context_used").default([]).notNull(),
	embedding: vector({ dimensions: 768 }),
	metadata: jsonb().default({}).notNull(),
	isSuccessful: boolean("is_successful").default(true).notNull(),
	errorMessage: text("error_message"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const userProfiles = pgTable("user_profiles", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	userId: uuid("user_id").notNull(),
	bio: text(),
	phone: varchar({ length: 20 }),
	address: text(),
	preferences: jsonb().default({}).notNull(),
	permissions: jsonb().default([]).notNull(),
	specializations: jsonb().default([]).notNull(),
	certifications: jsonb().default([]).notNull(),
	experienceLevel: varchar("experience_level", { length: 20 }).default('junior'),
	workPatterns: jsonb("work_patterns").default({}).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});
