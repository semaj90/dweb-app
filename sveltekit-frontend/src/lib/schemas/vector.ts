// Zod schemas for vector search & RAG endpoints
import { z } from 'zod';

export const vectorSearchRequestSchema = z.object({
  query: z.string().min(1),
  userId: z.string().min(1),
  caseId: z.string().optional(),
  limit: z.number().int().positive().max(100).default(10),
  threshold: z.number().min(0).max(1).default(0.7)
});

export const vectorSearchResponseSchema = z.object({
  results: z.array(z.object({
    id: z.string().optional(),
    score: z.number().optional(),
    content: z.string().optional(),
    metadata: z.record(z.any()).optional()
  })).default([])
});

export const ragRequestSchema = z.object({
  query: z.string().min(1),
  userId: z.string().min(1),
  caseId: z.string().optional(),
  useContext: z.boolean().default(true),
  model: z.string().default('gemma3-legal')
});

export const ragResponseSchema = z.object({
  response: z.string().nullable(),
  context: vectorSearchResponseSchema.shape.results.optional()
});

export const similarCasesRequestSchema = z.object({
  caseId: z.string().min(1),
  userId: z.string().min(1),
  limit: z.number().int().positive().max(25).default(5)
});

export const healthResponseSchema = z.object({
  status: z.enum(['healthy','degraded','unhealthy']).default('healthy')
});

export type VectorSearchRequest = z.infer<typeof vectorSearchRequestSchema>;
export type VectorSearchResponse = z.infer<typeof vectorSearchResponseSchema>;
export type RAGRequest = z.infer<typeof ragRequestSchema>;
export type RAGResponse = z.infer<typeof ragResponseSchema>;
export type SimilarCasesRequest = z.infer<typeof similarCasesRequestSchema>;
