// @ts-nocheck
/**
 * File Upload Schemas with Zod Validation
 * Comprehensive validation for legal document and evidence upload
 */
import { z } from 'zod';

// File type enumeration
export const fileTypeEnum = z.enum([
  'document',
  'image',
  'video',
  'audio',
  'physical',
  'digital',
  'testimony',
  'contract',
  'evidence'
]);

// Evidence type classification
export const evidenceTypeEnum = z.enum([
  'physical_evidence',
  'digital_evidence',
  'witness_testimony',
  'expert_opinion',
  'documents',
  'photographs',
  'video_recording',
  'audio_recording',
  'forensic_analysis',
  'chain_of_custody'
]);

// Confidentiality levels
export const confidentialityLevelEnum = z.enum([
  'public',
  'standard',
  'confidential',
  'classified',
  'restricted'
]);

// Case priority levels
export const casePriorityEnum = z.enum([
  'low',
  'medium',
  'high',
  'urgent',
  'critical'
]);

// Chain of custody entry schema
export const chainOfCustodyEntrySchema = z.object({
  timestamp: z.string().datetime(),
  officer: z.string().min(1, 'Officer name is required'),
  action: z.enum(['collected', 'transferred', 'analyzed', 'stored', 'returned']),
  location: z.string().min(1, 'Location is required'),
  notes: z.string().optional(),
  signature: z.string().optional()
});

// File metadata schema
export const fileMetadataSchema = z.object({
  originalName: z.string().min(1, 'Original filename is required'),
  size: z.number().positive('File size must be positive'),
  mimeType: z.string().min(1, 'MIME type is required'),
  hash: z.string().optional(),
  dimensions: z.object({
    width: z.number().optional(),
    height: z.number().optional()
  }).optional(),
  duration: z.number().optional(), // for audio/video files
  pages: z.number().optional(), // for PDF files
  extractedText: z.string().optional(),
  ocrConfidence: z.number().min(0).max(1).optional()
});

// Single file upload schema
export const fileUploadSchema = z.object({
  file: z.instanceof(File, { message: 'File is required' })
    .refine((file) => file.size > 0, 'File cannot be empty')
    .refine((file) => file.size <= 100 * 1024 * 1024, 'File size cannot exceed 100MB')
    .refine((file) => {
      const allowedTypes = [
        'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/webm',
        'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/aac',
        'application/pdf', 
        'application/msword', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'text/plain', 'text/csv', 'application/json',
        'application/zip', 'application/x-rar-compressed'
      ];
      return allowedTypes.includes(file.type);
    }, 'File type not supported'),
  
  title: z.string()
    .min(1, 'Title is required')
    .max(200, 'Title cannot exceed 200 characters'),
  
  description: z.string()
    .max(1000, 'Description cannot exceed 1000 characters')
    .optional(),
  
  caseId: z.string()
    .uuid('Case ID must be a valid UUID')
    .optional(),
  
  evidenceType: evidenceTypeEnum,
  
  fileType: fileTypeEnum,
  
  tags: z.array(z.string().min(1)).max(20, 'Cannot have more than 20 tags').default([]),
  
  confidentialityLevel: confidentialityLevelEnum.default('standard'),
  
  isAdmissible: z.boolean().default(true),
  
  collectedAt: z.string().datetime().optional(),
  
  collectedBy: z.string()
    .min(1, 'Collector name is required')
    .max(100, 'Collector name cannot exceed 100 characters')
    .optional(),
  
  location: z.string()
    .max(500, 'Location cannot exceed 500 characters')
    .optional(),
  
  chainOfCustody: z.array(chainOfCustodyEntrySchema).default([]),
  
  // AI processing options
  enableAiAnalysis: z.boolean().default(true),
  enableOcr: z.boolean().default(true),
  enableEmbeddings: z.boolean().default(true),
  enableSummarization: z.boolean().default(true),
  
  // Additional metadata
  metadata: z.record(z.string(), z.any()).default({})
});

// Multiple file upload schema
export const multipleFileUploadSchema = z.object({
  files: z.array(z.instanceof(File))
    .min(1, 'At least one file is required')
    .max(10, 'Cannot upload more than 10 files at once')
    .refine((files) => {
      const totalSize = files.reduce((sum, file) => sum + file.size, 0);
      return totalSize <= 500 * 1024 * 1024; // 500MB total limit
    }, 'Total file size cannot exceed 500MB'),
  
  caseId: z.string()
    .uuid('Case ID must be a valid UUID')
    .optional(),
  
  commonTags: z.array(z.string()).default([]),
  confidentialityLevel: confidentialityLevelEnum.default('standard'),
  collectedBy: z.string().optional(),
  location: z.string().optional(),
  enableAiAnalysis: z.boolean().default(true)
});

// Case creation with file upload schema
export const caseWithFilesSchema = z.object({
  // Case information
  title: z.string()
    .min(1, 'Case title is required')
    .max(200, 'Case title cannot exceed 200 characters'),
  
  caseNumber: z.string()
    .min(1, 'Case number is required')
    .max(50, 'Case number cannot exceed 50 characters'),
  
  description: z.string()
    .min(10, 'Case description must be at least 10 characters')
    .max(2000, 'Case description cannot exceed 2000 characters'),
  
  category: z.string()
    .min(1, 'Case category is required')
    .max(100, 'Case category cannot exceed 100 characters'),
  
  priority: casePriorityEnum.default('medium'),
  
  incidentDate: z.string().datetime().optional(),
  
  location: z.string()
    .max(500, 'Location cannot exceed 500 characters')
    .optional(),
  
  jurisdiction: z.string()
    .max(200, 'Jurisdiction cannot exceed 200 characters')
    .optional(),
  
  leadProsecutor: z.string()
    .max(100, 'Lead prosecutor name cannot exceed 100 characters')
    .optional(),
  
  assignedTeam: z.array(z.string()).default([]),
  
  tags: z.array(z.string()).max(20, 'Cannot have more than 20 tags').default([]),
  
  // Files to upload with the case
  files: z.array(z.instanceof(File))
    .min(0, 'Files are optional')
    .max(20, 'Cannot upload more than 20 files when creating a case'),
  
  // File descriptions (matching array indices)
  fileDescriptions: z.array(z.string()).default([]),
  fileTitles: z.array(z.string()).default([]),
  fileTypes: z.array(fileTypeEnum).default([]),
  fileEvidenceTypes: z.array(evidenceTypeEnum).default([])
});

// Search and filter schema
export const fileSearchSchema = z.object({
  query: z.string().max(200, 'Search query cannot exceed 200 characters').optional(),
  caseId: z.string().uuid().optional(),
  fileType: fileTypeEnum.optional(),
  evidenceType: evidenceTypeEnum.optional(),
  confidentialityLevel: confidentialityLevelEnum.optional(),
  tags: z.array(z.string()).default([]),
  dateFrom: z.string().datetime().optional(),
  dateTo: z.string().datetime().optional(),
  isAdmissible: z.boolean().optional(),
  collectedBy: z.string().optional(),
  minSize: z.number().positive().optional(),
  maxSize: z.number().positive().optional(),
  hasAiAnalysis: z.boolean().optional(),
  sortBy: z.enum(['created_at', 'updated_at', 'title', 'size', 'relevance']).default('created_at'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
  limit: z.number().min(1).max(100).default(20),
  offset: z.number().min(0).default(0)
});

// AI analysis result schema
export const aiAnalysisResultSchema = z.object({
  summary: z.string().optional(),
  keyPoints: z.array(z.string()).default([]),
  entities: z.array(z.object({
    name: z.string(),
    type: z.string(),
    confidence: z.number().min(0).max(1)
  })).default([]),
  sentiment: z.object({
    score: z.number().min(-1).max(1),
    label: z.enum(['positive', 'negative', 'neutral'])
  }).optional(),
  categories: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1).default(0),
  processingTime: z.number().positive().optional(),
  model: z.string().optional(),
  embedding: z.array(z.number()).optional()
});

// Export types
export type FileUpload = z.infer<typeof fileUploadSchema>;
export type MultipleFileUpload = z.infer<typeof multipleFileUploadSchema>;
export type CaseWithFiles = z.infer<typeof caseWithFilesSchema>;
export type FileSearch = z.infer<typeof fileSearchSchema>;
export type FileMetadata = z.infer<typeof fileMetadataSchema>;
export type ChainOfCustodyEntry = z.infer<typeof chainOfCustodyEntrySchema>;
export type AiAnalysisResult = z.infer<typeof aiAnalysisResultSchema>;

// Export enums as types
export type FileType = z.infer<typeof fileTypeEnum>;
export type EvidenceType = z.infer<typeof evidenceTypeEnum>;
export type ConfidentialityLevel = z.infer<typeof confidentialityLevelEnum>;
export type CasePriority = z.infer<typeof casePriorityEnum>;

// Helper functions for file validation
export const validateFileSize = (file: File, maxSizeMB: number = 100): boolean => {
  return file.size <= maxSizeMB * 1024 * 1024;
};

export const validateFileType = (file: File, allowedTypes: string[]): boolean => {
  return allowedTypes.includes(file.type);
};

export const getFileCategory = (mimeType: string): FileType => {
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType.startsWith('video/')) return 'video';
  if (mimeType.startsWith('audio/')) return 'audio';
  if (mimeType.includes('pdf') || mimeType.includes('document') || mimeType.includes('text')) return 'document';
  return 'digital';
};

export const formatFileSize = (bytes: number): string => {
  const sizes = ['B', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 B';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
};

// Default form values
export const defaultFileUploadValues: Partial<FileUpload> = {
  title: '',
  description: '',
  evidenceType: 'documents',
  fileType: 'document',
  tags: [],
  confidentialityLevel: 'standard',
  isAdmissible: true,
  enableAiAnalysis: true,
  enableOcr: true,
  enableEmbeddings: true,
  enableSummarization: true,
  chainOfCustody: [],
  metadata: {}
};

export const defaultCaseWithFilesValues: Partial<CaseWithFiles> = {
  title: '',
  caseNumber: '',
  description: '',
  category: '',
  priority: 'medium',
  assignedTeam: [],
  tags: [],
  files: [],
  fileDescriptions: [],
  fileTitles: [],
  fileTypes: [],
  fileEvidenceTypes: []
};