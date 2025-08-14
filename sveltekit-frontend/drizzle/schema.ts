import { pgTable, serial, text, bigint, index, uuid, varchar, jsonb, timestamp, integer, unique, boolean, foreignKey, vector, numeric, doublePrecision } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"



export const __drizzle_migrations__ = pgTable("__drizzle_migrations__", {
	id: serial().primaryKey().notNull(),
	hash: text().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	created_at: bigint({ mode: "number" }),
});

export const user_activities = pgTable("user_activities", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	user_id: varchar({ length: 255 }).notNull(),
	session_id: varchar({ length: 255 }),
	action: varchar({ length: 100 }).notNull(),
	query: text(),
	results: jsonb(),
	timestamp: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	feedback: varchar({ length: 50 }),
	processing_time_ms: integer(),
}, (table) => {
	return {
		idx_user_activities_user_timestamp: index("idx_user_activities_user_timestamp").using("btree", table.user_id.asc().nullsLast().op("text_ops"), table.timestamp.desc().nullsFirst().op("text_ops")),
	}
});

export const processing_jobs = pgTable("processing_jobs", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	job_id: varchar({ length: 255 }).notNull(),
	job_type: varchar({ length: 100 }).notNull(),
	status: varchar({ length: 50 }).default('pending'),
	payload: jsonb(),
	result: jsonb(),
	created_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	started_at: timestamp({ mode: 'string' }),
	completed_at: timestamp({ mode: 'string' }),
	error_message: text(),
	retry_count: integer().default(0),
}, (table) => {
	return {
		idx_processing_jobs_status: index("idx_processing_jobs_status").using("btree", table.status.asc().nullsLast().op("text_ops"), table.created_at.asc().nullsLast().op("text_ops")),
		processing_jobs_job_id_key: unique("processing_jobs_job_id_key").on(table.job_id),
	}
});

export const users = pgTable("users", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	email: varchar({ length: 255 }).notNull(),
	name: text(),
	role: varchar({ length: 50 }).default('prosecutor').notNull(),
	created_at: timestamp({ withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	updated_at: timestamp({ withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	hashed_password: text(),
	is_active: boolean().default(true),
	first_name: varchar({ length: 100 }),
	last_name: varchar({ length: 100 }),
	email_verified: timestamp({ mode: 'string' }),
	avatar_url: text(),
}, (table) => {
	return {
		users_email_key: unique("users_email_key").on(table.email),
	}
});

export const user_xstates = pgTable("user_xstates", {
	id: uuid().default(sql`uuid_generate_v4()`).notNull(),
	user_id: varchar({ length: 255 }).notNull(),
	session_id: varchar({ length: 255 }).notNull(),
	current_state: varchar({ length: 100 }),
	previous_states: jsonb(),
	typing_patterns: jsonb(),
	upload_history: jsonb(),
	search_queries: jsonb(),
	document_interactions: jsonb(),
	learning_context: jsonb(),
	timestamp: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		idx_user_xstates_session: index("idx_user_xstates_session").using("btree", table.session_id.asc().nullsLast().op("text_ops")),
		idx_user_xstates_user_id: index("idx_user_xstates_user_id").using("btree", table.user_id.asc().nullsLast().op("text_ops")),
	}
});

export const realtime_training_data = pgTable("realtime_training_data", {
	id: uuid().default(sql`uuid_generate_v4()`).notNull(),
	user_xstate_id: uuid(),
	document_id: uuid(),
	query_embedding: vector({ dimensions: 768 }),
	response_quality: numeric(),
	contextual_fit: numeric(),
	training_weight: numeric(),
	timestamp: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		idx_training_data_timestamp: index("idx_training_data_timestamp").using("btree", table.timestamp.asc().nullsLast().op("timestamptz_ops")),
		realtime_training_data_user_xstate_id_fkey: foreignKey({
			columns: [table.user_xstate_id],
			foreignColumns: [user_xstates.id],
			name: "realtime_training_data_user_xstate_id_fkey"
		}),
		realtime_training_data_document_id_fkey: foreignKey({
			columns: [table.document_id],
			foreignColumns: [enhanced_documents.id],
			name: "realtime_training_data_document_id_fkey"
		}),
	}
});

export const document_metadata = pgTable("document_metadata", {
	id: uuid().default(sql`uuid_generate_v4()`).notNull(),
	case_id: varchar({ length: 255 }).notNull(),
	filename: varchar({ length: 500 }).notNull(),
	object_name: varchar({ length: 1000 }).notNull(),
	content_type: varchar({ length: 100 }),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	size_bytes: bigint({ mode: "number" }),
	upload_time: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
	document_type: varchar({ length: 100 }),
	tags: jsonb(),
	metadata: jsonb(),
	processing_status: varchar({ length: 50 }).default('uploaded'),
	embedding: vector({ dimensions: 384 }),
	extracted_text: text(),
	created_at: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
	updated_at: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
	original_filename: varchar({ length: 255 }),
	upload_status: varchar({ length: 20 }).default('pending'),
}, (table) => {
	return {
		idx_document_case_id: index("idx_document_case_id").using("btree", table.case_id.asc().nullsLast().op("text_ops")),
		idx_document_embedding: index("idx_document_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
		idx_document_metadata: index("idx_document_metadata").using("gin", table.metadata.asc().nullsLast().op("jsonb_ops")),
		idx_document_metadata_case_id: index("idx_document_metadata_case_id").using("btree", table.case_id.asc().nullsLast().op("text_ops")),
		idx_document_status: index("idx_document_status").using("btree", table.processing_status.asc().nullsLast().op("text_ops")),
		idx_document_tags: index("idx_document_tags").using("gin", table.tags.asc().nullsLast().op("jsonb_ops")),
	}
});

export const enhanced_documents = pgTable("enhanced_documents", {
	id: uuid().default(sql`uuid_generate_v4()`).notNull(),
	filename: varchar({ length: 500 }).notNull(),
	document_type: varchar({ length: 100 }),
	case_type: varchar({ length: 100 }),
	jurisdiction: varchar({ length: 200 }),
	year: integer(),
	content: text(),
	embedding: vector({ dimensions: 768 }),
	key_entities: jsonb(),
	legal_concepts: jsonb(),
	cited_cases: jsonb(),
	statutes: jsonb(),
	semantic_summary: text(),
	contextual_rank: numeric(),
	user_relevance: numeric(),
	processing_time: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
	xstate_context: jsonb(),
	created_at: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
	updated_at: timestamp({ withTimezone: true, mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		idx_enhanced_documents_embedding: index("idx_enhanced_documents_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
		idx_enhanced_documents_type: index("idx_enhanced_documents_type").using("btree", table.document_type.asc().nullsLast().op("text_ops")),
		idx_enhanced_documents_year: index("idx_enhanced_documents_year").using("btree", table.year.asc().nullsLast().op("int4_ops")),
	}
});

export const cases = pgTable("cases", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_number: varchar({ length: 50 }).notNull(),
	title: varchar({ length: 255 }).notNull(),
	description: text(),
	status: varchar({ length: 20 }).default('open').notNull(),
	priority: varchar({ length: 20 }).default('medium').notNull(),
	created_by: uuid(),
	created_at: timestamp({ withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	updated_at: timestamp({ withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => {
	return {
		cases_created_by_fkey: foreignKey({
			columns: [table.created_by],
			foreignColumns: [users.id],
			name: "cases_created_by_fkey"
		}),
		cases_case_number_key: unique("cases_case_number_key").on(table.case_number),
	}
});

export const document_processing = pgTable("document_processing", {
	id: serial().primaryKey().notNull(),
	document_id: uuid().notNull(),
	original_name: varchar({ length: 255 }).notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	file_size: bigint({ mode: "number" }).notNull(),
	file_type: varchar({ length: 100 }).notNull(),
	document_type: varchar({ length: 50 }),
	case_id: varchar({ length: 100 }),
	practice_area: varchar({ length: 100 }),
	jurisdiction: varchar({ length: 100 }),
	extracted_text: text(),
	text_length: integer(),
	summary: text(),
	key_points: jsonb(),
	embeddings: jsonb(),
	metadata: jsonb(),
	performance_metrics: jsonb(),
	processed_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	created_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => {
	return {
		idx_case_id: index("idx_case_id").using("btree", table.case_id.asc().nullsLast().op("text_ops")),
		idx_document_id: index("idx_document_id").using("btree", table.document_id.asc().nullsLast().op("uuid_ops")),
		idx_document_type: index("idx_document_type").using("btree", table.document_type.asc().nullsLast().op("text_ops")),
		idx_processed_at: index("idx_processed_at").using("btree", table.processed_at.asc().nullsLast().op("timestamp_ops")),
		document_processing_document_id_key: unique("document_processing_document_id_key").on(table.document_id),
	}
});

export const evidence = pgTable("evidence", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	title: varchar({ length: 255 }).notNull(),
	description: text(),
	evidence_type: varchar({ length: 50 }).notNull(),
	file_url: text(),
	created_at: timestamp({ withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	updated_at: timestamp({ withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => {
	return {
		idx_evidence_case_id: index("idx_evidence_case_id").using("btree", table.case_id.asc().nullsLast().op("uuid_ops")),
		evidence_case_id_fkey: foreignKey({
			columns: [table.case_id],
			foreignColumns: [cases.id],
			name: "evidence_case_id_fkey"
		}).onDelete("cascade"),
	}
});

export const legal_cases = pgTable("legal_cases", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	case_number: varchar({ length: 255 }).notNull(),
	title: varchar({ length: 500 }).notNull(),
	status: varchar({ length: 100 }).default('active'),
	prosecutor: varchar({ length: 255 }),
	defendant: varchar({ length: 255 }),
	created_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => {
	return {
		idx_legal_cases_number: index("idx_legal_cases_number").using("btree", table.case_number.asc().nullsLast().op("text_ops")),
		legal_cases_case_number_key: unique("legal_cases_case_number_key").on(table.case_number),
	}
});

export const legal_documents = pgTable("legal_documents", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	title: varchar({ length: 500 }).notNull(),
	content: text(),
	case_id: varchar({ length: 255 }),
	embedding: vector({ dimensions: 384 }),
	metadata: jsonb().default({}),
	created_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	summary: text(),
	entities: text(),
	risk_score: doublePrecision().default(0),
	updated_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => {
	return {
		idx_legal_docs_case: index("idx_legal_docs_case").using("btree", table.case_id.asc().nullsLast().op("text_ops")),
		idx_legal_docs_created: index("idx_legal_docs_created").using("btree", table.created_at.desc().nullsFirst().op("timestamp_ops")),
		idx_legal_docs_embedding: index("idx_legal_docs_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
		idx_legal_documents_case_id: index("idx_legal_documents_case_id").using("btree", table.case_id.asc().nullsLast().op("text_ops")),
		idx_legal_documents_embedding_ivfflat: index("idx_legal_documents_embedding_ivfflat").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")).with({lists: "100"}),
	}
});

export const sessions = pgTable("sessions", {
	id: varchar({ length: 255 }).primaryKey().notNull(),
	user_id: uuid().notNull(),
	expires_at: timestamp({ mode: 'string' }).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		sessions_user_id_fkey: foreignKey({
			columns: [table.user_id],
			foreignColumns: [users.id],
			name: "sessions_user_id_fkey"
		}).onDelete("cascade"),
	}
});

export const recommendation_models = pgTable("recommendation_models", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	user_id: varchar({ length: 255 }).notNull(),
	// TODO: failed to parse database type 'bytea'
	model_data: unknown("model_data"),
	training_iterations: integer().default(0),
	last_trained: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	performance_metrics: jsonb().default({}),
}, (table) => {
	return {
		idx_recommendation_models_user: index("idx_recommendation_models_user").using("btree", table.user_id.asc().nullsLast().op("text_ops")),
	}
});

export const indexed_files = pgTable("indexed_files", {
	id: uuid().default(sql`uuid_generate_v4()`).primaryKey().notNull(),
	file_path: varchar({ length: 1000 }).notNull(),
	content: text(),
	embedding: vector({ dimensions: 768 }),
	summary: text(),
	indexed_at: timestamp({ mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	processing_method: varchar({ length: 50 }).default('gpu'),
	gpu_processing_time_ms: integer(),
	metadata: jsonb().default({}),
}, (table) => {
	return {
		idx_indexed_files_embedding: index("idx_indexed_files_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
		idx_indexed_files_path: index("idx_indexed_files_path").using("btree", table.file_path.asc().nullsLast().op("text_ops")),
		indexed_files_file_path_key: unique("indexed_files_file_path_key").on(table.file_path),
	}
});

export const documents = pgTable("documents", {
	id: varchar({ length: 255 }).primaryKey().notNull(),
	url: text(),
	content: text(),
	parsed: jsonb(),
	summary: text(),
	embedding: vector({ dimensions: 384 }),
	metadata: jsonb(),
	created_at: timestamp({ mode: 'string' }).defaultNow(),
	updated_at: timestamp({ mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		idx_docs_created: index("idx_docs_created").using("btree", table.created_at.desc().nullsFirst().op("timestamp_ops")),
		idx_docs_embedding: index("idx_docs_embedding").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")),
		idx_docs_metadata: index("idx_docs_metadata").using("gin", table.metadata.asc().nullsLast().op("jsonb_ops")),
	}
});

export const auto_tags = pgTable("auto_tags", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	entity_id: uuid().notNull(),
	entity_type: varchar({ length: 50 }).notNull(),
	tag: varchar({ length: 100 }).notNull(),
	confidence: numeric({ precision: 3, scale:  2 }).notNull(),
	source: varchar({ length: 50 }).default('ai_analysis').notNull(),
	model: varchar({ length: 100 }),
	extracted_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	is_confirmed: boolean().default(false).notNull(),
	confirmed_by: uuid(),
	confirmed_at: timestamp({ mode: 'string' }),
});

export const document_embeddings = pgTable("document_embeddings", {
	id: uuid().default(sql`uuid_generate_v4()`).notNull(),
	document_id: uuid().notNull(),
	chunk_number: integer().notNull(),
	chunk_text: text().notNull(),
	embedding: vector({ dimensions: 384 }),
	created_at: timestamp({ mode: 'string' }).defaultNow(),
	updated_at: timestamp({ mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		idx_document_embeddings_document_id: index("idx_document_embeddings_document_id").using("btree", table.document_id.asc().nullsLast().op("uuid_ops")),
		idx_document_embeddings_vector: index("idx_document_embeddings_vector").using("ivfflat", table.embedding.asc().nullsLast().op("vector_cosine_ops")).with({lists: "100"}),
		document_embeddings_document_id_fkey: foreignKey({
			columns: [table.document_id],
			foreignColumns: [document_metadata.id],
			name: "document_embeddings_document_id_fkey"
		}).onDelete("cascade"),
	}
});

export const ai_reports = pgTable("ai_reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	report_type: varchar({ length: 50 }).notNull(),
	title: varchar({ length: 255 }).notNull(),
	content: text().notNull(),
	rich_text_content: jsonb(),
	metadata: jsonb().default({}).notNull(),
	canvas_elements: jsonb().default([]).notNull(),
	generated_by: varchar({ length: 100 }).default('gemma3-legal'),
	confidence: numeric({ precision: 3, scale:  2 }).default('0.85'),
	is_active: boolean().default(true),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const attachment_verifications = pgTable("attachment_verifications", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	attachment_id: uuid().notNull(),
	verified_by: uuid().notNull(),
	verification_status: varchar({ length: 50 }).default('pending').notNull(),
	verification_notes: text(),
	verified_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const canvas_annotations = pgTable("canvas_annotations", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	evidence_id: uuid(),
	fabric_data: jsonb().notNull(),
	annotation_type: varchar({ length: 50 }),
	coordinates: jsonb(),
	bounding_box: jsonb(),
	text: text(),
	color: varchar({ length: 20 }),
	layer_order: integer().default(0),
	is_visible: boolean().default(true),
	metadata: jsonb().default({}),
	version: integer().default(1),
	parent_annotation_id: uuid(),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const canvas_states = pgTable("canvas_states", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	name: varchar({ length: 255 }).notNull(),
	canvas_data: jsonb().notNull(),
	version: integer().default(1),
	is_default: boolean().default(false),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const case_activities = pgTable("case_activities", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid().notNull(),
	activity_type: varchar({ length: 50 }).notNull(),
	title: varchar({ length: 255 }).notNull(),
	description: text(),
	scheduled_for: timestamp({ mode: 'string' }),
	completed_at: timestamp({ mode: 'string' }),
	status: varchar({ length: 20 }).default('pending').notNull(),
	priority: varchar({ length: 20 }).default('medium').notNull(),
	assigned_to: uuid(),
	related_evidence: jsonb().default([]).notNull(),
	related_criminals: jsonb().default([]).notNull(),
	metadata: jsonb().default({}).notNull(),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const case_embeddings = pgTable("case_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	content: text().notNull(),
	embedding: text().notNull(),
	metadata: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const case_scores = pgTable("case_scores", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid().notNull(),
	score: numeric({ precision: 5, scale:  2 }).notNull(),
	risk_level: varchar({ length: 20 }).notNull(),
	breakdown: jsonb().default({}).notNull(),
	criteria: jsonb().default({}).notNull(),
	recommendations: jsonb().default([]).notNull(),
	calculated_by: uuid(),
	calculated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const chat_embeddings = pgTable("chat_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	conversation_id: uuid().notNull(),
	message_id: uuid().notNull(),
	content: text().notNull(),
	embedding: text().notNull(),
	role: varchar({ length: 20 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const citations = pgTable("citations", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	document_id: uuid(),
	citation_type: varchar({ length: 50 }).notNull(),
	relevance_score: numeric({ precision: 3, scale:  2 }),
	page_number: integer(),
	pinpoint_citation: varchar({ length: 100 }),
	quoted_text: text(),
	context_before: text(),
	context_after: text(),
	annotation: text(),
	legal_principle: text(),
	citation_format: varchar({ length: 20 }).default('bluebook'),
	formatted_citation: text(),
	shepards_treatment: varchar({ length: 50 }),
	is_key_authority: boolean().default(false),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const content_embeddings = pgTable("content_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	content_id: uuid().notNull(),
	content_type: varchar({ length: 50 }).notNull(),
	text_content: text().notNull(),
	embedding: text(),
	metadata: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const criminals = pgTable("criminals", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	first_name: varchar({ length: 100 }).notNull(),
	last_name: varchar({ length: 100 }).notNull(),
	middle_name: varchar({ length: 100 }),
	aliases: jsonb().default([]).notNull(),
	date_of_birth: timestamp({ mode: 'string' }),
	place_of_birth: varchar({ length: 200 }),
	address: text(),
	phone: varchar({ length: 20 }),
	email: varchar({ length: 255 }),
	ssn: varchar({ length: 11 }),
	drivers_license: varchar({ length: 50 }),
	height: integer(),
	weight: integer(),
	eye_color: varchar({ length: 20 }),
	hair_color: varchar({ length: 20 }),
	distinguishing_marks: text(),
	photo_url: text(),
	fingerprints: jsonb().default({}),
	threat_level: varchar({ length: 20 }).default('low').notNull(),
	status: varchar({ length: 20 }).default('active').notNull(),
	notes: text(),
	ai_summary: text(),
	ai_tags: jsonb().default([]).notNull(),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const document_chunks = pgTable("document_chunks", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	document_id: uuid().notNull(),
	document_type: varchar({ length: 50 }).notNull(),
	chunk_index: integer().notNull(),
	content: text().notNull(),
	embedding: vector({ dimensions: 768 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const email_verification_codes = pgTable("email_verification_codes", {
	id: serial().primaryKey().notNull(),
	user_id: uuid().notNull(),
	email: varchar({ length: 255 }).notNull(),
	code: varchar({ length: 8 }).notNull(),
	expires_at: timestamp({ withTimezone: true, mode: 'string' }).notNull(),
}, (table) => {
	return {
		email_verification_codes_user_id_unique: unique("email_verification_codes_user_id_unique").on(table.user_id),
	}
});

export const embedding_cache = pgTable("embedding_cache", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	text_hash: text().notNull(),
	embedding: vector({ dimensions: 768 }).notNull(),
	model: varchar({ length: 100 }).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
}, (table) => {
	return {
		embedding_cache_text_hash_unique: unique("embedding_cache_text_hash_unique").on(table.text_hash),
	}
});

export const evidence_vectors = pgTable("evidence_vectors", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	evidence_id: uuid(),
	content: text().notNull(),
	embedding: text().notNull(),
	metadata: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const hash_verifications = pgTable("hash_verifications", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	evidence_id: uuid(),
	verified_hash: varchar({ length: 64 }).notNull(),
	stored_hash: varchar({ length: 64 }),
	result: boolean().notNull(),
	verification_method: varchar({ length: 50 }).default('manual'),
	verified_by: uuid(),
	verified_at: timestamp({ mode: 'string' }).defaultNow(),
	notes: text(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const legal_analysis_sessions = pgTable("legal_analysis_sessions", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	user_id: uuid(),
	session_type: varchar({ length: 50 }).default('case_analysis'),
	analysis_prompt: text(),
	analysis_result: text(),
	confidence_level: numeric({ precision: 3, scale:  2 }),
	sources_used: jsonb().default([]).notNull(),
	model: varchar({ length: 100 }).default('gemma3-legal'),
	processing_time: integer(),
	is_active: boolean().default(true),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const legal_precedents = pgTable("legal_precedents", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_title: varchar({ length: 255 }).notNull(),
	citation: varchar({ length: 255 }).notNull(),
	court: varchar({ length: 100 }),
	year: integer(),
	jurisdiction: varchar({ length: 50 }),
	summary: text(),
	full_text: text(),
	embedding: text(),
	relevance_score: numeric({ precision: 3, scale:  2 }),
	legal_principles: jsonb().default([]).notNull(),
	linked_cases: jsonb().default([]).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const legal_research = pgTable("legal_research", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	query: text().notNull(),
	search_terms: jsonb().default([]),
	jurisdiction: varchar({ length: 100 }),
	date_range: jsonb(),
	court_level: varchar({ length: 50 }),
	practice_area: varchar({ length: 100 }),
	results_count: integer().default(0),
	search_results: jsonb().default([]),
	ai_summary: text(),
	key_findings: jsonb().default([]),
	recommended_citations: jsonb().default([]),
	search_duration: integer(),
	data_source: varchar({ length: 50 }),
	is_bookmarked: boolean().default(false),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const password_reset_tokens = pgTable("password_reset_tokens", {
	token_hash: varchar({ length: 63 }).primaryKey().notNull(),
	user_id: uuid().notNull(),
	expires_at: timestamp({ withTimezone: true, mode: 'string' }).notNull(),
});

export const persons_of_interest = pgTable("persons_of_interest", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	name: varchar({ length: 255 }).notNull(),
	aliases: jsonb().default([]).notNull(),
	relationship: varchar({ length: 100 }),
	threat_level: varchar({ length: 20 }).default('low'),
	status: varchar({ length: 20 }).default('active'),
	profile_data: jsonb().default({}).notNull(),
	tags: jsonb().default([]).notNull(),
	position: jsonb().default({}).notNull(),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const rag_messages = pgTable("rag_messages", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	session_id: varchar({ length: 255 }).notNull(),
	message_index: integer().notNull(),
	role: varchar({ length: 20 }).notNull(),
	content: text().notNull(),
	retrieved_sources: jsonb().default([]).notNull(),
	source_count: integer().default(0).notNull(),
	retrieval_score: varchar({ length: 10 }),
	processing_time: integer(),
	model: varchar({ length: 100 }),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const rag_sessions = pgTable("rag_sessions", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	session_id: varchar({ length: 255 }).notNull(),
	user_id: uuid(),
	title: varchar({ length: 255 }),
	model: varchar({ length: 100 }),
	is_active: boolean().default(true).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
}, (table) => {
	return {
		rag_sessions_session_id_unique: unique("rag_sessions_session_id_unique").on(table.session_id),
	}
});

export const reports = pgTable("reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	case_id: uuid(),
	title: varchar({ length: 255 }).notNull(),
	content: text(),
	report_type: varchar({ length: 50 }).default('case_summary'),
	status: varchar({ length: 20 }).default('draft'),
	is_public: boolean().default(false),
	tags: jsonb().default([]).notNull(),
	metadata: jsonb().default({}).notNull(),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const saved_reports = pgTable("saved_reports", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	title: varchar({ length: 300 }).notNull(),
	case_id: uuid(),
	report_type: varchar({ length: 50 }).notNull(),
	template_id: uuid(),
	content: jsonb().notNull(),
	html_content: text(),
	generated_by: varchar({ length: 50 }).default('manual'),
	ai_model: varchar({ length: 50 }),
	ai_prompt: text(),
	export_format: varchar({ length: 20 }).default('pdf'),
	status: varchar({ length: 20 }).default('draft'),
	version: integer().default(1),
	word_count: integer(),
	tags: jsonb().default([]),
	metadata: jsonb().default({}),
	shared_with: jsonb().default([]),
	last_exported: timestamp({ mode: 'string' }),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const statutes = pgTable("statutes", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	title: varchar({ length: 255 }).notNull(),
	code: varchar({ length: 100 }).notNull(),
	description: text(),
	category: varchar({ length: 100 }),
	jurisdiction: varchar({ length: 100 }),
	is_active: boolean().default(true),
	penalties: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const themes = pgTable("themes", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	name: varchar({ length: 100 }).notNull(),
	description: text(),
	css_variables: jsonb().notNull(),
	color_palette: jsonb().notNull(),
	is_system: boolean().default(false).notNull(),
	is_public: boolean().default(false).notNull(),
	created_by: uuid(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const user_ai_queries = pgTable("user_ai_queries", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	user_id: uuid().notNull(),
	case_id: uuid(),
	query: text().notNull(),
	response: text().notNull(),
	model: varchar({ length: 100 }).default('gemma3-legal').notNull(),
	query_type: varchar({ length: 50 }).default('general'),
	confidence: numeric({ precision: 3, scale:  2 }),
	tokens_used: integer(),
	processing_time: integer(),
	context_used: jsonb().default([]).notNull(),
	embedding: vector({ dimensions: 768 }),
	metadata: jsonb().default({}).notNull(),
	is_successful: boolean().default(true).notNull(),
	error_message: text(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const user_embeddings = pgTable("user_embeddings", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	user_id: uuid(),
	content: text().notNull(),
	embedding: text().notNull(),
	metadata: jsonb().default({}).notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const vector_metadata = pgTable("vector_metadata", {
	id: uuid().defaultRandom().primaryKey().notNull(),
	document_id: text().notNull(),
	collection_name: varchar({ length: 100 }).notNull(),
	metadata: jsonb().default({}).notNull(),
	content_hash: text().notNull(),
	created_at: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updated_at: timestamp({ mode: 'string' }).defaultNow(),
}, (table) => {
	return {
		vector_metadata_document_id_unique: unique("vector_metadata_document_id_unique").on(table.document_id),
	}
});
