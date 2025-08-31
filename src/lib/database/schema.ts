// Compatibility shim - forward to the canonical schema index
// This makes imports from "$lib/database/schema" consistent whether they use the directory or direct file.
export * from './schema/index';
export { default } from './schema/index';

// Compatibility shim: re-export Postgres schema with camelCase aliases
// This file maps snake_case table exports from the server schema to the
// camelCase names used across the frontend. Add more aliases here as we
// discover missing exported members during incremental TypeScript fixes.

import * as pg from '$lib/server/db/schema-postgres';

// Direct re-exports (keep original names available too)
export const users = pg.users;
export const sessions = pg.sessions;
export const cases = pg.cases;
export const evidence = pg.evidence;
export const legalDocuments = pg.legal_documents;
export const documentChunks = pg.document_chunks;
export const vectorMetadata = pg.vector_metadata;
export const embeddingCache = pg.embedding_cache;
export const statutes = pg.statutes;
export const legalAnalysisSessions = pg.legal_analysis_sessions;
export const userAiQueries = pg.user_ai_queries;
export const autoTags = pg.auto_tags;
export const caseScores = pg.case_scores;
export const ragSessions = pg.rag_sessions;
export const ragMessages = pg.rag_messages;
export const canvasStates = pg.canvas_states;
export const keys = pg.keys;

// Fallback aliases commonly referenced across frontend code (best-effort mapping)
// These ensure older imports keep working while we migrate call sites incrementally.
export const contentEmbeddings = pg.embedding_cache; // historically a content_embeddings table
export const caseEmbeddings = pg.vector_metadata; // alias for per-case vector metadata
export const evidenceVectors = pg.vector_metadata; // evidence vector metadata alias
export const legalDocs = pg.legal_documents;
export const documents = pg.legal_documents;

// Re-export types with camelCase names used around the codebase
export type Evidence = pg.Evidence;
export type NewEvidence = pg.NewEvidence;
export type LegalDocument = pg.LegalDocument;
export type NewLegalDocument = pg.NewLegalDocument;
export type DocumentChunk = pg.DocumentChunk;
export type NewDocumentChunk = pg.NewDocumentChunk;
export type VectorMetadata = pg.VectorMetadata;
export type NewVectorMetadata = pg.NewVectorMetadata;
export type EmbeddingCache = pg.EmbeddingCache;
export type NewEmbeddingCache = pg.NewEmbeddingCache;
export type Statute = pg.Statute;
export type NewStatute = pg.NewStatute;
export type LegalAnalysisSession = pg.LegalAnalysisSession;
export type NewLegalAnalysisSession = pg.NewLegalAnalysisSession;
export type UserAiQuery = pg.UserAiQuery;
export type NewUserAiQuery = pg.NewUserAiQuery;
export type AutoTag = pg.AutoTag;
export type NewAutoTag = pg.NewAutoTag;
export type CaseScore = pg.CaseScore;
export type NewCaseScore = pg.NewCaseScore;
export type RagSession = pg.RagSession;
export type NewRagSession = pg.NewRagSession;
export type RagMessage = pg.RagMessage;
export type NewRagMessage = pg.NewRagMessage;
export type CanvasState = pg.CanvasState;
export type NewCanvasState = pg.NewCanvasState;
export type Key = pg.Key;
export type NewKey = pg.NewKey;
export type User = pg.User;
export type NewUser = pg.NewUser;
export type Session = pg.Session;
export type NewSession = pg.NewSession;
export type Case = pg.Case;
export type NewCase = pg.NewCase;

// Helper: named export of the raw server schema to support advanced usage.
export const serverSchema = pg;
