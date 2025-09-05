import { z } from 'zod';

export const fileUploadSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255),
  description: z.string().optional(),
  caseId: z.string().uuid('Invalid case ID'),
  tags: z.array(z.string()).optional().default([]),
  file: z.any().refine((file) => file instanceof File && file.size > 0, 'File is required'),
  documentType: z.enum(['contract', 'evidence', 'correspondence', 'report', 'legal_brief', 'other']).default('evidence'),
  confidentialityLevel: z.enum(['public', 'internal', 'confidential', 'restricted']).default('internal'),
});

export const evidenceUploadSchema = z.object({
  id: z.string().uuid().optional(),
  caseId: z.string().uuid('Case ID is required'),
  title: z.string().min(1, 'Title is required').max(255),
  description: z.string().max(2000).optional(),
  evidenceType: z.enum(['document', 'photo', 'video', 'audio', 'physical', 'digital', 'other']).default('document'),
  collectedBy: z.string().min(1, 'Collected by is required').max(100),
  collectedDate: z.string().datetime('Invalid date format'),
  chainOfCustody: z.array(z.object({
    handedBy: z.string(),
    receivedBy: z.string(),
    timestamp: z.string().datetime(),
    notes: z.string().optional()
  })).optional(),
  tags: z.array(z.string()).optional().default([]),
  isConfidential: z.boolean().default(false),
  retentionPeriod: z.number().positive().optional(),
  metadata: z.record(z.any()).optional()
});

export const bulkUploadSchema = z.object({
  files: z.array(z.instanceof(File)).min(1, 'At least one file is required'),
  caseId: z.string().uuid('Invalid case ID'),
  description: z.string().max(2000).optional(),
  tags: z.array(z.string()).optional().default([]),
  documentType: z.enum(['contract', 'evidence', 'correspondence', 'report', 'legal_brief', 'other']).default('evidence'),
  confidentialityLevel: z.enum(['public', 'internal', 'confidential', 'restricted']).default('internal'),
});

export type FileUpload = z.infer<typeof fileUploadSchema>;
export type EvidenceUpload = z.infer<typeof evidenceUploadSchema>;
export type BulkUpload = z.infer<typeof bulkUploadSchema>;