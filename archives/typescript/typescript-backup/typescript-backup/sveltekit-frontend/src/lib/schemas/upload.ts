import { z } from "zod";


// Allowed file types for evidence upload
const ALLOWED_MIME_TYPES = [
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'image/jpeg',
  'image/png',
  'image/gif',
  'video/mp4',
  'audio/mpeg',
  'audio/wav'
] as const;

// Maximum file size (50MB)
const MAX_FILE_SIZE = 50 * 1024 * 1024;

// File upload schema
export const fileUploadSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255),
  description: z.string().optional(),
  type: z.enum(['document', 'image', 'video', 'audio', 'physical', 'digital']),
  caseId: z.string().uuid('Invalid case ID'),
  file: z
    .instanceof(File, { message: 'Please upload a file' })
    .refine((file) => file.size > 0, 'File is empty')
    .refine((file) => file.size <= MAX_FILE_SIZE, 'File size must be less than 50MB')
    .refine(
      (file) => ALLOWED_MIME_TYPES.includes(file.type as any),
      'File type not supported'
    ),
  tags: z.array(z.string()).optional(),
  isPrivate: z.boolean().default(false),
  aiAnalysis: z.boolean().default(true) // Whether to run AI analysis on upload
});

// Document processing options schema
export const documentProcessingSchema = z.object({
  extractText: z.boolean().default(true),
  generateEmbeddings: z.boolean().default(true),
  generateSummary: z.boolean().default(true),
  extractEntities: z.boolean().default(false),
  analyzeSentiment: z.boolean().default(false),
  classifyDocument: z.boolean().default(false),
  chunkSize: z.number().min(100).max(2000).default(1000),
  chunkOverlap: z.number().min(0).max(500).default(200)
});

// Batch upload schema
export const batchUploadSchema = z.object({
  caseId: z.string().uuid('Invalid case ID'),
  files: z.array(
    z.object({
      file: z.instanceof(File),
      title: z.string().optional(),
      type: z.enum(['document', 'image', 'video', 'audio', 'physical', 'digital']).optional()
    })
  ).min(1, 'At least one file is required').max(10, 'Maximum 10 files at once'),
  processingOptions: documentProcessingSchema.optional()
});

// Search upload schema for finding existing uploads
export const searchUploadSchema = z.object({
  query: z.string().optional(),
  caseId: z.string().uuid().optional(),
  type: z.enum(['document', 'image', 'video', 'audio', 'physical', 'digital']).optional(),
  dateFrom: z.date().optional(),
  dateTo: z.date().optional(),
  tags: z.array(z.string()).optional(),
  limit: z.number().min(1).max(100).default(20),
  offset: z.number().min(0).default(0)
});

// Helper function to validate file upload
export function validateFileUpload(file: File) {
  const errors: string[] = [];

  if (file.size === 0) {
    errors.push('File is empty');
  }
  
  if (file.size > MAX_FILE_SIZE) {
    errors.push(`File size exceeds ${MAX_FILE_SIZE / (1024 * 1024)}MB limit`);
  }
  
  if (!ALLOWED_MIME_TYPES.includes(file.type as any)) {
    errors.push(`File type '${file.type}' is not supported`);
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

// Export types
export type FileUploadInput = z.input<typeof fileUploadSchema>;
export type FileUploadData = z.output<typeof fileUploadSchema>;
export type DocumentProcessingOptions = z.output<typeof documentProcessingSchema>;
export type BatchUploadData = z.output<typeof batchUploadSchema>;
export type SearchUploadData = z.output<typeof searchUploadSchema>;