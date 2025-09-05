import { relations } from "drizzle-orm/relations";
import { userXstates, realtimeTrainingData, enhancedDocuments, documentMetadata, documentEmbeddings, users, sessions, cases, documents, legalEmbeddings } from "./schema";

export const realtimeTrainingDataRelations = relations(realtimeTrainingData, ({one}) => ({
	userXstate: one(userXstates, {
		fields: [realtimeTrainingData.userXstateId],
		references: [userXstates.id]
	}),
	enhancedDocument: one(enhancedDocuments, {
		fields: [realtimeTrainingData.documentId],
		references: [enhancedDocuments.id]
	}),
}));

export const userXstatesRelations = relations(userXstates, ({many}) => ({
	realtimeTrainingData: many(realtimeTrainingData),
}));

export const enhancedDocumentsRelations = relations(enhancedDocuments, ({many}) => ({
	realtimeTrainingData: many(realtimeTrainingData),
}));

export const documentEmbeddingsRelations = relations(documentEmbeddings, ({one}) => ({
	documentMetadatum: one(documentMetadata, {
		fields: [documentEmbeddings.documentId],
		references: [documentMetadata.id]
	}),
}));

export const documentMetadataRelations = relations(documentMetadata, ({many}) => ({
	documentEmbeddings: many(documentEmbeddings),
}));

export const sessionsRelations = relations(sessions, ({one}) => ({
	user: one(users, {
		fields: [sessions.userId],
		references: [users.id]
	}),
}));

export const usersRelations = relations(users, ({many}) => ({
	sessions: many(sessions),
	cases: many(cases),
}));

export const casesRelations = relations(cases, ({one}) => ({
	user: one(users, {
		fields: [cases.assignedAttorney],
		references: [users.id]
	}),
}));

export const legalEmbeddingsRelations = relations(legalEmbeddings, ({one}) => ({
	document: one(documents, {
		fields: [legalEmbeddings.documentId],
		references: [documents.id]
	}),
}));

export const documentsRelations = relations(documents, ({many}) => ({
	legalEmbeddings: many(legalEmbeddings),
}));