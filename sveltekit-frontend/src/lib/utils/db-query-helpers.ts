
// Fixed database query utilities with proper field mappings
import { sql, eq, and, or, like, desc, asc } from 'drizzle-orm';
import type { SQL } from 'drizzle-orm';
import type { PgColumn } from 'drizzle-orm/pg-core';

// Database field mapping utilities
export const fieldMap = {
  // User fields
  user: {
    createdAt: 'created_at',
    updatedAt: 'updated_at',
    emailVerified: 'email_verified',
    hashedPassword: 'hashed_password',
    firstName: 'first_name',
    lastName: 'last_name',
    avatarUrl: 'avatar_url',
    isActive: 'is_active'
  },
  // Case fields
  case: {
    caseNumber: 'case_number',
    incidentDate: 'incident_date',
    dangerScore: 'danger_score',
    estimatedValue: 'estimated_value',
    leadProsecutor: 'lead_prosecutor',
    assignedTeam: 'assigned_team',
    aiSummary: 'ai_summary',
    aiTags: 'ai_tags',
    titleEmbedding: 'title_embedding',
    descriptionEmbedding: 'description_embedding',
    fullTextEmbedding: 'full_text_embedding',
    createdBy: 'created_by',
    createdAt: 'created_at',
    updatedAt: 'updated_at',
    closedAt: 'closed_at'
  },
  // Evidence fields
  evidence: {
    caseId: 'case_id',
    criminalId: 'criminal_id',
    evidenceType: 'evidence_type',
    fileType: 'file_type',
    subType: 'sub_type',
    fileUrl: 'file_url',
    fileName: 'file_name',
    fileSize: 'file_size',
    mimeType: 'mime_type',
    chainOfCustody: 'chain_of_custody',
    collectedAt: 'collected_at',
    collectedBy: 'collected_by',
    labAnalysis: 'lab_analysis',
    aiAnalysis: 'ai_analysis',
    aiTags: 'ai_tags',
    aiSummary: 'ai_summary',
    isAdmissible: 'is_admissible',
    confidentialityLevel: 'confidentiality_level',
    canvasPosition: 'canvas_position',
    posX: 'pos_x',
    posY: 'pos_y',
    zIndex: 'z_index',
    canvasState: 'canvas_state',
    titleEmbedding: 'title_embedding',
    descriptionEmbedding: 'description_embedding',
    contentEmbedding: 'content_embedding',
    summaryEmbedding: 'summary_embedding',
    uploadedBy: 'uploaded_by',
    uploadedAt: 'uploaded_at',
    updatedAt: 'updated_at'
  }
} as const;

// Query builder helpers with proper type safety
export function buildFilters(filters: SQL[]): SQL | undefined {
  return filters.length > 0 ? and(...filters) : undefined;
}

export function buildSearchFilters(
  searchColumns: PgColumn[],
  searchTerm: string
): SQL {
  const searchFilters = searchColumns.map((col: any) => like(col, `%${searchTerm}%`)
  );
  return or(...searchFilters);
}

export function applySorting(
  column: PgColumn,
  order: 'asc' | 'desc' = 'desc'
): SQL {
  return order === 'asc' ? asc(column) : desc(column);
}

// Type-safe filter builders
export const filterBuilders = {
  textFilter: (column: PgColumn, value: string) =>
    eq(column, value),

  searchFilter: (columns: PgColumn[], term: string) =>
    or(...columns.map((col: any) => like(col, `%${term}%`))),

  dateRangeFilter: (column: PgColumn, start: Date, end: Date) =>
    and(
      // Use SQL for date comparisons
      like(column, `%${start.toISOString().split('T')[0]}%`),
      like(column, `%${end.toISOString().split('T')[0]}%`)
    )
};

// Pagination helpers
export interface PaginationParams {
  page: number;
  limit: number;
}

export function getPaginationParams(
  page: string | null,
  limit: string | null
): PaginationParams {
  const pageNum = Math.max(1, parseInt(page || '1'));
  const limitNum = Math.min(100, Math.max(1, parseInt(limit || '20')));

  return {
    page: pageNum,
    limit: limitNum
  };
}

export function applyPagination(params: PaginationParams) {
  const offset = (params.page - 1) * params.limit;
  return {
    limit: params.limit,
    offset
  };
}

// Common query patterns
export const queryPatterns = {
  // Safe case filtering
  caseFilters: (filters: {
    search?: string;
    status?: string;
    priority?: string;
  }) => {
    const conditions: SQL[] = [];

    if (filters.search) {
      // Add search conditions - implement based on actual schema
    }
    if (filters.status) {
      // Add status filter - implement based on actual schema
    }
    if (filters.priority) {
      // Add priority filter - implement based on actual schema
    }

    return buildFilters(conditions);
  },

  // Safe evidence filtering
  evidenceFilters: (filters: {
    caseId?: string;
    evidenceType?: string;
    search?: string;
  }) => {
    const conditions: SQL[] = [];

    if (filters.caseId) {
      // Add caseId filter
    }
    if (filters.evidenceType) {
      // Add type filter
    }
    if (filters.search) {
      // Add search filter
    }

    return buildFilters(conditions);
  }
};

export default {
  fieldMap,
  buildFilters,
  buildSearchFilters,
  applySorting,
  filterBuilders,
  getPaginationParams,
  applyPagination,
  queryPatterns
};
