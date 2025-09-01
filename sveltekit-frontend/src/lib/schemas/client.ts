// Client-safe schema definitions
// These mirror server schemas but are safe for browser import

import { z } from 'zod';
import { URL } from "url";

export const evidenceSchema = z.object({
  title: z.string().min(1, 'Title is required'),
  description: z.string().min(1, 'Description is required'), 
  tags: z.array(z.string()).optional(),
  caseId: z.string().uuid().optional(),
  fileUrl: z.string().url().optional(),
  metadata: z.record(z.any()).optional(),
});

export type EvidenceFormData = z.infer<typeof evidenceSchema>;

// Vector search result type (client-safe)
export interface VectorSearchResult {
  id: string;
  content: string;
  similarity: number;
  metadata?: Record<string, any>;
  embedding?: number[];
}

// Export common form schemas
export const caseSchema = z.object({
  title: z.string().min(1, 'Title is required'),
  description: z.string().min(1, 'Description is required'),
  status: z.enum(['active', 'pending', 'closed']).optional(),
  priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
});

export type CaseFormData = z.infer<typeof caseSchema>;