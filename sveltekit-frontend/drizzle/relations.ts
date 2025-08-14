import { relations } from "drizzle-orm/relations";
import { user_xstates, realtime_training_data, enhanced_documents, users, cases, evidence, sessions, document_metadata, document_embeddings } from "./schema";

export const realtime_training_dataRelations = relations(realtime_training_data, ({one}) => ({
	user_xstate: one(user_xstates, {
		fields: [realtime_training_data.user_xstate_id],
		references: [user_xstates.id]
	}),
	enhanced_document: one(enhanced_documents, {
		fields: [realtime_training_data.document_id],
		references: [enhanced_documents.id]
	}),
}));

export const user_xstatesRelations = relations(user_xstates, ({many}) => ({
	realtime_training_data: many(realtime_training_data),
}));

export const enhanced_documentsRelations = relations(enhanced_documents, ({many}) => ({
	realtime_training_data: many(realtime_training_data),
}));

export const casesRelations = relations(cases, ({one, many}) => ({
	user: one(users, {
		fields: [cases.created_by],
		references: [users.id]
	}),
	evidences: many(evidence),
}));

export const usersRelations = relations(users, ({many}) => ({
	cases: many(cases),
	sessions: many(sessions),
}));

export const evidenceRelations = relations(evidence, ({one}) => ({
	case: one(cases, {
		fields: [evidence.case_id],
		references: [cases.id]
	}),
}));

export const sessionsRelations = relations(sessions, ({one}) => ({
	user: one(users, {
		fields: [sessions.user_id],
		references: [users.id]
	}),
}));

export const document_embeddingsRelations = relations(document_embeddings, ({one}) => ({
	document_metadatum: one(document_metadata, {
		fields: [document_embeddings.document_id],
		references: [document_metadata.id]
	}),
}));

export const document_metadataRelations = relations(document_metadata, ({many}) => ({
	document_embeddings: many(document_embeddings),
}));