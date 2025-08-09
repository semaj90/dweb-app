// Legal Documents Schema
// Defines the structure for legal documents in the database

/**
 * Legal document type definition
 */
export const DocumentTypes = {
  CONTRACT: 'contract',
  BRIEF: 'brief',
  MOTION: 'motion',
  PLEADING: 'pleading',
  OPINION: 'opinion',
  STATUTE: 'statute',
  MEMORANDUM: 'memorandum',
  DISCOVERY: 'discovery',
  CORRESPONDENCE: 'correspondence',
  OTHER: 'other'
};

/**
 * Document status types
 */
export const DocumentStatus = {
  DRAFT: 'draft',
  PENDING_REVIEW: 'pending_review',
  REVIEWED: 'reviewed',
  APPROVED: 'approved',
  FILED: 'filed',
  ARCHIVED: 'archived'
};

/**
 * Legal document interface
 */
export class LegalDocument {
  constructor(data = {}) {
    this.id = data.id || null;
    this.title = data.title || '';
    this.content = data.content || '';
    this.documentType = data.documentType || DocumentTypes.OTHER;
    this.status = data.status || DocumentStatus.DRAFT;
    this.caseId = data.caseId || null;
    this.clientName = data.clientName || '';
    this.attorneyName = data.attorneyName || '';
    this.jurisdiction = data.jurisdiction || '';
    this.practiceArea = data.practiceArea || '';
    this.tags = data.tags || [];
    this.metadata = data.metadata || {};
    this.createdAt = data.createdAt || new Date();
    this.updatedAt = data.updatedAt || new Date();
    this.embedding = data.embedding || null;
    this.summary = data.summary || '';
    this.keyTerms = data.keyTerms || [];
    this.citations = data.citations || [];
    this.risks = data.risks || [];
    this.processingStatus = data.processingStatus || 'pending';
    this.confidenceScore = data.confidenceScore || 0;
  }

  /**
   * Validate document
   */
  validate() {
    const errors = [];
    
    if (!this.title || this.title.trim() === '') {
      errors.push('Title is required');
    }
    
    if (!this.content || this.content.trim() === '') {
      errors.push('Content is required');
    }
    
    if (!Object.values(DocumentTypes).includes(this.documentType)) {
      errors.push('Invalid document type');
    }
    
    if (!Object.values(DocumentStatus).includes(this.status)) {
      errors.push('Invalid document status');
    }
    
    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Convert to plain object
   */
  toObject() {
    return {
      id: this.id,
      title: this.title,
      content: this.content,
      documentType: this.documentType,
      status: this.status,
      caseId: this.caseId,
      clientName: this.clientName,
      attorneyName: this.attorneyName,
      jurisdiction: this.jurisdiction,
      practiceArea: this.practiceArea,
      tags: this.tags,
      metadata: this.metadata,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      embedding: this.embedding,
      summary: this.summary,
      keyTerms: this.keyTerms,
      citations: this.citations,
      risks: this.risks,
      processingStatus: this.processingStatus,
      confidenceScore: this.confidenceScore
    };
  }

  /**
   * Convert to JSON
   */
  toJSON() {
    return JSON.stringify(this.toObject());
  }

  /**
   * Create from JSON
   */
  static fromJSON(json) {
    const data = typeof json === 'string' ? JSON.parse(json) : json;
    return new LegalDocument(data);
  }
}

/**
 * New legal document type (for insertion)
 */
export class NewLegalDocument extends LegalDocument {
  constructor(data = {}) {
    super(data);
    delete this.id; // Remove ID for new documents
  }
}

/**
 * Legal documents table schema (for Drizzle ORM compatibility)
 */
export const legalDocuments = {
  id: 'serial',
  title: 'varchar(255)',
  content: 'text',
  documentType: 'varchar(100)',
  status: 'varchar(50)',
  caseId: 'integer',
  clientName: 'varchar(255)',
  attorneyName: 'varchar(255)',
  jurisdiction: 'varchar(100)',
  practiceArea: 'varchar(100)',
  tags: 'jsonb',
  metadata: 'jsonb',
  createdAt: 'timestamp',
  updatedAt: 'timestamp',
  embedding: 'vector(384)',
  summary: 'text',
  keyTerms: 'jsonb',
  citations: 'jsonb',
  risks: 'jsonb',
  processingStatus: 'varchar(50)',
  confidenceScore: 'float'
};

/**
 * Query builder helpers
 */
export const queryHelpers = {
  /**
   * Build WHERE clause from filters
   */
  buildWhereClause(filters) {
    const conditions = [];
    const values = [];
    let paramIndex = 1;

    if (filters.documentType) {
      conditions.push(`document_type = $${paramIndex++}`);
      values.push(filters.documentType);
    }

    if (filters.status) {
      conditions.push(`status = $${paramIndex++}`);
      values.push(filters.status);
    }

    if (filters.caseId) {
      conditions.push(`case_id = $${paramIndex++}`);
      values.push(filters.caseId);
    }

    if (filters.clientName) {
      conditions.push(`client_name ILIKE $${paramIndex++}`);
      values.push(`%${filters.clientName}%`);
    }

    if (filters.practiceArea) {
      conditions.push(`practice_area = $${paramIndex++}`);
      values.push(filters.practiceArea);
    }

    if (filters.jurisdiction) {
      conditions.push(`jurisdiction = $${paramIndex++}`);
      values.push(filters.jurisdiction);
    }

    if (filters.searchTerm) {
      conditions.push(`(title ILIKE $${paramIndex} OR content ILIKE $${paramIndex})`);
      values.push(`%${filters.searchTerm}%`);
      paramIndex++;
    }

    return {
      whereClause: conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '',
      values
    };
  },

  /**
   * Build ORDER BY clause
   */
  buildOrderByClause(orderBy, orderDirection = 'DESC') {
    const validColumns = Object.keys(legalDocuments);
    const column = validColumns.includes(orderBy) ? orderBy : 'created_at';
    const direction = orderDirection.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';
    
    return `ORDER BY ${column} ${direction}`;
  }
};

// Export everything
export default {
  LegalDocument,
  NewLegalDocument,
  DocumentTypes,
  DocumentStatus,
  legalDocuments,
  queryHelpers
};