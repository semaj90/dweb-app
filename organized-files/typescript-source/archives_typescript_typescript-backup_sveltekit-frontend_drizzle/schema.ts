import { pgTable, uuid, varchar, text, boolean, jsonb, timestamp, index, unique, serial, vector, doublePrecision, integer, uniqueIndex, numeric, bigint, foreignKey, bigserial, real } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"



export const reports = pgTable("reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	title: varchar({ length: 255 }).notNull(),
	content: text(),
	reportType: varchar("report_type", { length: 50 }).default('case_summary'),
	status: varchar({ length: 20 }).default('draft'),
	isPublic: boolean("is_public").default(false),
	tags: jsonb().default([]).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const vscodeProblems = pgTable("vscode_problems", {
	id: serial().primaryKey().notNull(),
	filePath: text("file_path").notNull(),
	problemHash: text("problem_hash").notNull(),
	problemData: jsonb("problem_data").notNull(),
	semanticFeatures: jsonb("semantic_features"),
	embeddings: vector({ dimensions: 384 }),
	solutions: jsonb().default([]),
	confidenceScore: doublePrecision("confidence_score").default(0),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow(),
	solvedAt: timestamp("solved_at", { mode: 'string' }),
	workerId: text("worker_id"),
	processingTimeMs: integer("processing_time_ms"),
}, (table) => [
	index("idx_problems_data").using("gin", table.problemData.asc().nullsLast().op("jsonb_ops")),
	index("idx_problems_hash").using("btree", table.problemHash.asc().nullsLast().op("text_ops")),
	index("idx_problems_path").using("btree", table.filePath.asc().nullsLast().op("text_ops")),
	unique("vscode_problems_problem_hash_key").on(table.problemHash),
]);

export const semanticCache = pgTable("semantic_cache", {
	id: serial().primaryKey().notNull(),
	contentHash: text("content_hash").notNull(),
	contentText: text("content_text").notNull(),
	language: text(),
	semanticAnalysis: jsonb("semantic_analysis").notNull(),
	entities: jsonb().default([]),
	relationships: jsonb().default([]),
	embeddings: vector({ dimensions: 384 }),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_semantic_analysis").using("gin", table.semanticAnalysis.asc().nullsLast().op("jsonb_ops")),
	index("idx_semantic_hash").using("btree", table.contentHash.asc().nullsLast().op("text_ops")),
	unique("semantic_cache_content_hash_key").on(table.contentHash),
]);

export const solutionPatterns = pgTable("solution_patterns", {
	id: serial().primaryKey().notNull(),
	problemType: text("problem_type").notNull(),
	patternData: jsonb("pattern_data").notNull(),
	solutionTemplate: jsonb("solution_template").notNull(),
	successRate: doublePrecision("success_rate").default(0),
	usageCount: integer("usage_count").default(0),
	lastUsed: timestamp("last_used", { mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_patterns_type").using("btree", table.problemType.asc().nullsLast().op("text_ops")),
]);

export const solverMetrics = pgTable("solver_metrics", {
	id: serial().primaryKey().notNull(),
	timestamp: timestamp({ mode: 'string' }).defaultNow(),
	workerId: text("worker_id"),
	problemsProcessed: integer("problems_processed"),
	solutionsFound: integer("solutions_found"),
	processingTimeMs: integer("processing_time_ms"),
	memoryUsageMb: integer("memory_usage_mb"),
	gpuUtilization: doublePrecision("gpu_utilization"),
	metricsData: jsonb("metrics_data"),
});

export const vectors = pgTable("vectors", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	ownerType: text("owner_type").notNull(),
	ownerId: uuid("owner_id").notNull(),
	embedding: vector({ dimensions: 384 }).default('[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]').notNull(),
	payload: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_vectors_created_at").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_vectors_embedding_cosine").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")).with({lists: "100"}),
	index("idx_vectors_embedding_l2").using("ivfflat", table.embedding.asc().nullsLast().op("vector_l2_ops")).with({lists: "100"}),
	index("idx_vectors_owner_id").using("btree", table.ownerId.asc().nullsLast().op("uuid_ops")),
	index("idx_vectors_owner_type").using("btree", table.ownerType.asc().nullsLast().op("text_ops")),
	uniqueIndex("idx_vectors_owner_unique").using("btree", table.ownerType.asc().nullsLast().op("uuid_ops"), table.ownerId.asc().nullsLast().op("text_ops")),
	index("vectors_embedding_idx").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")).with({lists: "100"}),
]);

export const enhancedDocuments = pgTable("enhanced_documents", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	filename: varchar({ length: 500 }).notNull(),
	documentType: varchar("document_type", { length: 100 }),
	caseType: varchar("case_type", { length: 100 }),
	jurisdiction: varchar({ length: 200 }),
	year: integer(),
	content: text(),
	embedding: vector({ dimensions: 384 }),
	keyEntities: jsonb("key_entities"),
	legalConcepts: jsonb("legal_concepts"),
	citedCases: jsonb("cited_cases"),
	statutes: jsonb(),
	semanticSummary: text("semantic_summary"),
	contextualRank: numeric("contextual_rank"),
	userRelevance: numeric("user_relevance"),
	processingTime: timestamp("processing_time", { withTimezone: true, mode: 'string' }).defaultNow(),
	xstateContext: jsonb("xstate_context"),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_enhanced_documents_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
	index("idx_enhanced_documents_type").using("btree", table.documentType.asc().nullsLast().op("text_ops")),
	index("idx_enhanced_documents_year").using("btree", table.year.asc().nullsLast().op("int4_ops")),
]);

export const documentProcessing = pgTable("document_processing", {
	id: serial().primaryKey().notNull(),
	documentId: uuid("document_id").notNull(),
	originalName: varchar("original_name", { length: 255 }).notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	fileSize: bigint("file_size", { mode: "number" }).notNull(),
	fileType: varchar("file_type", { length: 100 }).notNull(),
	documentType: varchar("document_type", { length: 50 }),
	caseId: varchar("case_id", { length: 100 }),
	practiceArea: varchar("practice_area", { length: 100 }),
	jurisdiction: varchar({ length: 100 }),
	extractedText: text("extracted_text"),
	textLength: integer("text_length"),
	summary: text(),
	keyPoints: jsonb("key_points"),
	embeddings: jsonb(),
	metadata: jsonb(),
	performanceMetrics: jsonb("performance_metrics"),
	processedAt: timestamp("processed_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => [
	index("idx_case_id").using("btree", table.caseId.asc().nullsLast().op("text_ops")),
	index("idx_document_id").using("btree", table.documentId.asc().nullsLast().op("uuid_ops")),
	index("idx_document_type").using("btree", table.documentType.asc().nullsLast().op("text_ops")),
	index("idx_processed_at").using("btree", table.processedAt.asc().nullsLast().op("timestamp_ops")),
	unique("document_processing_document_id_key").on(table.documentId),
]);

export const legalCases = pgTable("legal_cases", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	caseNumber: varchar("case_number", { length: 255 }).notNull(),
	title: varchar({ length: 500 }).notNull(),
	status: varchar({ length: 100 }).default('active'),
	prosecutor: varchar({ length: 255 }),
	defendant: varchar({ length: 255 }),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => [
	index("idx_legal_cases_number").using("btree", table.caseNumber.asc().nullsLast().op("text_ops")),
	unique("legal_cases_case_number_key").on(table.caseNumber),
]);

export const userXstates = pgTable("user_xstates", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	userId: varchar("user_id", { length: 255 }).notNull(),
	sessionId: varchar("session_id", { length: 255 }).notNull(),
	currentState: varchar("current_state", { length: 100 }),
	previousStates: jsonb("previous_states"),
	typingPatterns: jsonb("typing_patterns"),
	uploadHistory: jsonb("upload_history"),
	searchQueries: jsonb("search_queries"),
	documentInteractions: jsonb("document_interactions"),
	learningContext: jsonb("learning_context"),
	timestamp: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_user_xstates_session").using("btree", table.sessionId.asc().nullsLast().op("text_ops")),
	index("idx_user_xstates_user_id").using("btree", table.userId.asc().nullsLast().op("text_ops")),
]);

export const realtimeTrainingData = pgTable("realtime_training_data", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	userXstateId: uuid("user_xstate_id"),
	documentId: uuid("document_id"),
	queryEmbedding: vector("query_embedding", { dimensions: 384 }),
	responseQuality: numeric("response_quality"),
	contextualFit: numeric("contextual_fit"),
	trainingWeight: numeric("training_weight"),
	timestamp: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_training_data_timestamp").using("btree", table.timestamp.asc().nullsLast().op("timestamptz_ops")),
	foreignKey({
			columns: [table.userXstateId],
			foreignColumns: [userXstates.id],
			name: "realtime_training_data_user_xstate_id_fkey"
		}),
	foreignKey({
			columns: [table.documentId],
			foreignColumns: [enhancedDocuments.id],
			name: "realtime_training_data_document_id_fkey"
		}),
]);

export const documentMetadata = pgTable("document_metadata", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	caseId: varchar("case_id", { length: 255 }).notNull(),
	filename: varchar({ length: 500 }).notNull(),
	objectName: varchar("object_name", { length: 1000 }).notNull(),
	contentType: varchar("content_type", { length: 100 }),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	sizeBytes: bigint("size_bytes", { mode: "number" }),
	uploadTime: timestamp("upload_time", { withTimezone: true, mode: 'string' }).defaultNow(),
	documentType: varchar("document_type", { length: 100 }),
	tags: jsonb(),
	metadata: jsonb(),
	processingStatus: varchar("processing_status", { length: 50 }).default('uploaded'),
	embedding: vector({ dimensions: 384 }),
	extractedText: text("extracted_text"),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow(),
	originalFilename: varchar("original_filename", { length: 255 }),
	uploadStatus: varchar("upload_status", { length: 20 }).default('pending'),
	jurisdiction: varchar({ length: 100 }),
	priority: integer().default(1),
	ingestSource: varchar("ingest_source", { length: 100 }).default('manual'),
	summary: text(),
}, (table) => [
	index("idx_document_case_id").using("btree", table.caseId.asc().nullsLast().op("text_ops")),
	index("idx_document_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
	index("idx_document_metadata").using("gin", table.metadata.asc().nullsLast().op("jsonb_ops")),
	index("idx_document_metadata_case_id").using("btree", table.caseId.asc().nullsLast().op("text_ops")),
	index("idx_document_status").using("btree", table.processingStatus.asc().nullsLast().op("text_ops")),
	index("idx_document_tags").using("gin", table.tags.asc().nullsLast().op("jsonb_ops")),
	unique("document_metadata_object_name_key").on(table.objectName),
]);

export const recommendationModels = pgTable("recommendation_models", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	userId: varchar("user_id", { length: 255 }).notNull(),
	// TODO: failed to parse database type 'bytea'
	modelData: unknown("model_data"),
	trainingIterations: integer("training_iterations").default(0),
	lastTrained: timestamp("last_trained", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	performanceMetrics: jsonb("performance_metrics").default({}),
}, (table) => [
	index("idx_recommendation_models_user").using("btree", table.userId.asc().nullsLast().op("text_ops")),
]);

export const indexedFiles = pgTable("indexed_files", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	filePath: varchar("file_path", { length: 1000 }).notNull(),
	content: text(),
	embedding: vector({ dimensions: 384 }),
	summary: text(),
	indexedAt: timestamp("indexed_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	processingMethod: varchar("processing_method", { length: 50 }).default('gpu'),
	gpuProcessingTimeMs: integer("gpu_processing_time_ms"),
	metadata: jsonb().default({}),
}, (table) => [
	index("idx_indexed_files_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
	index("idx_indexed_files_path").using("btree", table.filePath.asc().nullsLast().op("text_ops")),
	unique("indexed_files_file_path_key").on(table.filePath),
]);

export const documentEmbeddings = pgTable("document_embeddings", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	documentId: uuid("document_id").notNull(),
	chunkNumber: integer("chunk_number").notNull(),
	chunkText: text("chunk_text").notNull(),
	embedding: vector({ dimensions: 384 }),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_document_embeddings_document_id").using("btree", table.documentId.asc().nullsLast().op("uuid_ops")),
	index("idx_document_embeddings_vector").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")).with({lists: "100"}),
	foreignKey({
			columns: [table.documentId],
			foreignColumns: [documentMetadata.id],
			name: "document_embeddings_document_id_fkey"
		}).onDelete("cascade"),
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

export const aiReports = pgTable("ai_reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	reportType: varchar("report_type", { length: 50 }).notNull(),
	title: varchar({ length: 255 }).notNull(),
	content: text().notNull(),
	richTextContent: jsonb("rich_text_content"),
	metadata: jsonb().default({}).notNull(),
	canvasElements: jsonb("canvas_elements").default([]).notNull(),
	generatedBy: varchar("generated_by", { length: 100 }).default('gemma3-legal'),
	confidence: numeric({ precision: 3, scale:  2 }).default('0.85'),
	isActive: boolean("is_active").default(true),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const legalPrecedents = pgTable("legal_precedents", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseTitle: varchar("case_title", { length: 255 }).notNull(),
	citation: varchar({ length: 255 }).notNull(),
	court: varchar({ length: 100 }),
	year: integer(),
	jurisdiction: varchar({ length: 50 }),
	summary: text(),
	fullText: text("full_text"),
	embedding: text(),
	relevanceScore: numeric("relevance_score", { precision: 3, scale:  2 }),
	legalPrinciples: jsonb("legal_principles").default([]).notNull(),
	linkedCases: jsonb("linked_cases").default([]).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const vectorMetadata = pgTable("vector_metadata", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	documentId: text("document_id").notNull(),
	collectionName: varchar("collection_name", { length: 100 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	contentHash: text("content_hash").notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow(),
}, (table) => [
	unique("vector_metadata_document_id_unique").on(table.documentId),
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

export const qdrantCollections = pgTable("qdrant_collections", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	name: varchar({ length: 100 }).notNull(),
	description: text(),
	vectorSize: integer("vector_size").default(384).notNull(),
	distance: varchar({ length: 20 }).default('Cosine').notNull(),
	status: varchar({ length: 20 }).default('active').notNull(),
	isOptimized: boolean("is_optimized").default(false).notNull(),
	pointsCount: integer("points_count").default(0).notNull(),
	lastSynced: timestamp("last_synced", { withTimezone: true, mode: 'string' }),
	config: jsonb().default({}),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	uniqueIndex("qdrant_collections_name_idx").using("btree", table.name.asc().nullsLast().op("text_ops")),
	index("qdrant_collections_status_idx").using("btree", table.status.asc().nullsLast().op("text_ops")),
	unique("qdrant_collections_name_key").on(table.name),
]);

export const drizzleMigrations = pgTable("__drizzle_migrations__", {
	id: serial().primaryKey().notNull(),
	hash: text().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	createdAt: bigint("created_at", { mode: "number" }),
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
	index("users_profile_embedding_hnsw_idx").using("hnsw", table.profileEmbedding.asc().nullsLast().op("vector_cosine_ops")).with({m: "16",ef_construction: "64"}),
	index("users_role_idx").using("btree", table.role.asc().nullsLast().op("text_ops")),
	index("users_username_idx").using("btree", table.username.asc().nullsLast().op("text_ops")),
	unique("users_email_key").on(table.email),
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
	foreignKey({
			columns: [table.userId],
			foreignColumns: [users.id],
			name: "sessions_user_id_fkey"
		}).onDelete("cascade"),
]);

export const processingJobs = pgTable("processing_jobs", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	jobId: varchar("job_id", { length: 255 }).notNull(),
	jobType: varchar("job_type", { length: 100 }).notNull(),
	status: varchar({ length: 50 }).default('pending'),
	payload: jsonb(),
	result: jsonb(),
	createdAt: timestamp("created_at", { mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	startedAt: timestamp("started_at", { mode: 'string' }),
	completedAt: timestamp("completed_at", { mode: 'string' }),
	errorMessage: text("error_message"),
	retryCount: integer("retry_count").default(0),
}, (table) => [
	index("idx_processing_jobs_status").using("btree", table.status.asc().nullsLast().op("text_ops"), table.createdAt.asc().nullsLast().op("text_ops")),
	unique("processing_jobs_job_id_key").on(table.jobId),
]);

export const userActivities = pgTable("user_activities", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	userId: varchar("user_id", { length: 255 }).notNull(),
	sessionId: varchar("session_id", { length: 255 }),
	action: varchar({ length: 100 }).notNull(),
	query: text(),
	results: jsonb(),
	timestamp: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	feedback: varchar({ length: 50 }),
	processingTimeMs: integer("processing_time_ms"),
}, (table) => [
	index("idx_user_activities_user_timestamp").using("btree", table.userId.asc().nullsLast().op("text_ops"), table.timestamp.desc().nullsFirst().op("text_ops")),
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
	index("evidence_user_id_idx").using("btree", table.userId.asc().nullsLast().op("uuid_ops")),
	index("idx_evidence_case_id").using("btree", table.caseId.asc().nullsLast().op("uuid_ops")),
]);

export const embeddingJobs = pgTable("embedding_jobs", {
	id: bigserial({ mode: "bigint" }).primaryKey().notNull(),
	jobId: text("job_id"),
	type: text(),
	status: text(),
	inputData: text("input_data"),
	result: text(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }),
}, (table) => [
	uniqueIndex("idx_embedding_jobs_job_id").using("btree", table.jobId.asc().nullsLast().op("text_ops")),
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
	index("cases_assigned_attorney_idx").using("btree", table.assignedAttorney.asc().nullsLast().op("uuid_ops")),
	index("cases_case_embedding_hnsw_idx").using("hnsw", table.caseEmbedding.asc().nullsLast().op("vector_cosine_ops")).with({m: "16",ef_construction: "64"}),
	uniqueIndex("cases_case_number_idx").using("btree", table.caseNumber.asc().nullsLast().op("text_ops")),
	index("cases_practice_area_idx").using("btree", table.practiceArea.asc().nullsLast().op("text_ops")),
	index("cases_status_idx").using("btree", table.status.asc().nullsLast().op("text_ops")),
	foreignKey({
			columns: [table.assignedAttorney],
			foreignColumns: [users.id],
			name: "cases_assigned_attorney_fkey"
		}),
	unique("cases_case_number_key").on(table.caseNumber),
]);

export const attachmentVerifications = pgTable("attachment_verifications", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	attachmentId: uuid("attachment_id").notNull(),
	verifiedBy: uuid("verified_by").notNull(),
	verificationStatus: varchar("verification_status", { length: 50 }).default('pending').notNull(),
	verificationNotes: text("verification_notes"),
	verifiedAt: timestamp("verified_at", { mode: 'string' }).defaultNow().notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const canvasAnnotations = pgTable("canvas_annotations", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	evidenceId: uuid("evidence_id"),
	fabricData: jsonb("fabric_data").notNull(),
	annotationType: varchar("annotation_type", { length: 50 }),
	coordinates: jsonb(),
	boundingBox: jsonb("bounding_box"),
	text: text(),
	color: varchar({ length: 20 }),
	layerOrder: integer("layer_order").default(0),
	isVisible: boolean("is_visible").default(true),
	metadata: jsonb().default({}),
	version: integer().default(1),
	parentAnnotationId: uuid("parent_annotation_id"),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

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

export const caseActivities = pgTable("case_activities", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id").notNull(),
	activityType: varchar("activity_type", { length: 50 }).notNull(),
	title: varchar({ length: 255 }).notNull(),
	description: text(),
	scheduledFor: timestamp("scheduled_for", { mode: 'string' }),
	completedAt: timestamp("completed_at", { mode: 'string' }),
	status: varchar({ length: 20 }).default('pending').notNull(),
	priority: varchar({ length: 20 }).default('medium').notNull(),
	assignedTo: uuid("assigned_to"),
	relatedEvidence: jsonb("related_evidence").default([]).notNull(),
	relatedCriminals: jsonb("related_criminals").default([]).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const caseEmbeddings = pgTable("case_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	content: text().notNull(),
	embedding: text().notNull(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
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

export const chatEmbeddings = pgTable("chat_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	conversationId: uuid("conversation_id").notNull(),
	messageId: uuid("message_id").notNull(),
	content: text().notNull(),
	embedding: text().notNull(),
	role: varchar({ length: 20 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const citations = pgTable("citations", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	documentId: uuid("document_id"),
	citationType: varchar("citation_type", { length: 50 }).notNull(),
	relevanceScore: numeric("relevance_score", { precision: 3, scale:  2 }),
	pageNumber: integer("page_number"),
	pinpointCitation: varchar("pinpoint_citation", { length: 100 }),
	quotedText: text("quoted_text"),
	contextBefore: text("context_before"),
	contextAfter: text("context_after"),
	annotation: text(),
	legalPrinciple: text("legal_principle"),
	citationFormat: varchar("citation_format", { length: 20 }).default('bluebook'),
	formattedCitation: text("formatted_citation"),
	shepardsTreatment: varchar("shepards_treatment", { length: 50 }),
	isKeyAuthority: boolean("is_key_authority").default(false),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow().notNull(),
});

export const contentEmbeddings = pgTable("content_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	contentId: uuid("content_id").notNull(),
	contentType: varchar("content_type", { length: 50 }).notNull(),
	textContent: text("text_content").notNull(),
	embedding: text(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
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
	embedding: vector({ dimensions: 384 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const emailVerificationCodes = pgTable("email_verification_codes", {
	id: serial().primaryKey().notNull(),
	userId: uuid("user_id").notNull(),
	email: varchar({ length: 255 }).notNull(),
	code: varchar({ length: 8 }).notNull(),
	expiresAt: timestamp("expires_at", { withTimezone: true, mode: 'string' }).notNull(),
}, (table) => [
	unique("email_verification_codes_user_id_unique").on(table.userId),
]);

export const embeddingCache = pgTable("embedding_cache", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	textHash: text("text_hash").notNull(),
	embedding: vector({ dimensions: 384 }).notNull(),
	model: varchar({ length: 100 }).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	unique("embedding_cache_text_hash_unique").on(table.textHash),
]);

export const evidenceVectors = pgTable("evidence_vectors", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	evidenceId: uuid("evidence_id"),
	content: text().notNull(),
	embedding: text().notNull(),
	metadata: jsonb().default({}).notNull(),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const hashVerifications = pgTable("hash_verifications", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	evidenceId: uuid("evidence_id"),
	verifiedHash: varchar("verified_hash", { length: 64 }).notNull(),
	storedHash: varchar("stored_hash", { length: 64 }),
	result: boolean().notNull(),
	verificationMethod: varchar("verification_method", { length: 50 }).default('manual'),
	verifiedBy: uuid("verified_by"),
	verifiedAt: timestamp("verified_at", { mode: 'string' }).defaultNow(),
	notes: text(),
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

export const legalResearch = pgTable("legal_research", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	caseId: uuid("case_id"),
	query: text().notNull(),
	searchTerms: jsonb("search_terms").default([]),
	jurisdiction: varchar({ length: 100 }),
	dateRange: jsonb("date_range"),
	courtLevel: varchar("court_level", { length: 50 }),
	practiceArea: varchar("practice_area", { length: 100 }),
	resultsCount: integer("results_count").default(0),
	searchResults: jsonb("search_results").default([]),
	aiSummary: text("ai_summary"),
	keyFindings: jsonb("key_findings").default([]),
	recommendedCitations: jsonb("recommended_citations").default([]),
	searchDuration: integer("search_duration"),
	dataSource: varchar("data_source", { length: 50 }),
	isBookmarked: boolean("is_bookmarked").default(false),
	createdBy: uuid("created_by"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const passwordResetTokens = pgTable("password_reset_tokens", {
	tokenHash: varchar("token_hash", { length: 63 }).primaryKey().notNull(),
	userId: uuid("user_id").notNull(),
	expiresAt: timestamp("expires_at", { withTimezone: true, mode: 'string' }).notNull(),
});

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
}, (table) => [
	unique("rag_sessions_session_id_unique").on(table.sessionId),
]);

export const savedReports = pgTable("saved_reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	title: varchar({ length: 300 }).notNull(),
	caseId: uuid("case_id"),
	reportType: varchar("report_type", { length: 50 }).notNull(),
	templateId: uuid("template_id"),
	content: jsonb().notNull(),
	htmlContent: text("html_content"),
	generatedBy: varchar("generated_by", { length: 50 }).default('manual'),
	aiModel: varchar("ai_model", { length: 50 }),
	aiPrompt: text("ai_prompt"),
	exportFormat: varchar("export_format", { length: 20 }).default('pdf'),
	status: varchar({ length: 20 }).default('draft'),
	version: integer().default(1),
	wordCount: integer("word_count"),
	tags: jsonb().default([]),
	metadata: jsonb().default({}),
	sharedWith: jsonb("shared_with").default([]),
	lastExported: timestamp("last_exported", { mode: 'string' }),
	createdBy: uuid("created_by"),
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

export const themes = pgTable("themes", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	name: varchar({ length: 100 }).notNull(),
	description: text(),
	cssVariables: jsonb("css_variables").notNull(),
	colorPalette: jsonb("color_palette").notNull(),
	isSystem: boolean("is_system").default(false).notNull(),
	isPublic: boolean("is_public").default(false).notNull(),
	createdBy: uuid("created_by"),
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
	embedding: vector({ dimensions: 384 }),
	metadata: jsonb().default({}).notNull(),
	isSuccessful: boolean("is_successful").default(true).notNull(),
	errorMessage: text("error_message"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow().notNull(),
});

export const userEmbeddings = pgTable("user_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	userId: uuid("user_id"),
	content: text().notNull(),
	embedding: text().notNull(),
	metadata: jsonb().default({}).notNull(),
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
}, (table) => [
	unique("user_profiles_user_id_key").on(table.userId),
]);

export const documents = pgTable("documents", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	filename: text().notNull(),
	content: text().notNull(),
	originalContent: text("original_content"),
	metadata: jsonb(),
	confidence: real(),
	legalAnalysis: jsonb("legal_analysis"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow(),
	updatedAt: timestamp("updated_at", { mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_documents_content").using("gin", sql`to_tsvector('english'::regconfig, content)`),
]);

export const legalEmbeddings = pgTable("legal_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	documentId: uuid("document_id"),
	content: text().notNull(),
	embedding: vector({ dimensions: 384 }),
	metadata: jsonb(),
	model: text().default('nomic-embed-text'),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow(),
}, (table) => [
	index("idx_embeddings_vector").using("hnsw", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
	index("idx_legal_embeddings_vector").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")).with({lists: "100"}),
	foreignKey({
			columns: [table.documentId],
			foreignColumns: [documents.id],
			name: "legal_embeddings_document_id_fkey"
		}).onDelete("cascade"),
]);

export const searchSessions = pgTable("search_sessions", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	query: text().notNull(),
	queryEmbedding: vector("query_embedding", { dimensions: 384 }),
	results: jsonb(),
	searchType: text("search_type").default('hybrid'),
	resultCount: integer("result_count"),
	createdAt: timestamp("created_at", { mode: 'string' }).defaultNow(),
});

export const vectorOperations = pgTable("vector_operations", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	operationType: varchar("operation_type", { length: 50 }).notNull(),
	entityType: varchar("entity_type", { length: 50 }).notNull(),
	entityId: uuid("entity_id").notNull(),
	modelName: varchar("model_name", { length: 100 }).default('nomic-embed-text').notNull(),
	dimensions: integer().default(384).notNull(),
	similarity: varchar({ length: 20 }).default('cosine'),
	processingTimeMs: integer("processing_time_ms"),
	similarityScore: real("similarity_score"),
	qdrantSynced: boolean("qdrant_synced").default(false).notNull(),
	qdrantSyncedAt: timestamp("qdrant_synced_at", { withTimezone: true, mode: 'string' }),
	qdrantError: text("qdrant_error"),
	status: varchar({ length: 20 }).default('pending').notNull(),
	metadata: jsonb().default({}),
	error: text(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
	completedAt: timestamp("completed_at", { withTimezone: true, mode: 'string' }),
}, (table) => [
	index("vector_operations_entity_id_idx").using("btree", table.entityId.asc().nullsLast().op("uuid_ops")),
	index("vector_operations_entity_type_idx").using("btree", table.entityType.asc().nullsLast().op("text_ops")),
	index("vector_operations_operation_type_idx").using("btree", table.operationType.asc().nullsLast().op("text_ops")),
	index("vector_operations_qdrant_synced_idx").using("btree", table.qdrantSynced.asc().nullsLast().op("bool_ops")),
	index("vector_operations_status_idx").using("btree", table.status.asc().nullsLast().op("text_ops")),
]);
