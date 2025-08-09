// @ts-nocheck
// @ts-nocheck
// Legal documents schema stub

export const legalDocuments = {
  id: 'legalDocuments',
  tableName: 'legal_documents',
  columns: {
    id: { type: 'string', primaryKey: true },
    title: { type: 'string' },
    content: { type: 'text' },
    type: { type: 'string' },
    status: { type: 'string' },
    createdAt: { type: 'timestamp' },
    updatedAt: { type: 'timestamp' }
  }
};

export interface NewLegalDocument {
  id?: string;
  title: string;
  content: string;
  type: string;
  status?: string;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface LegalDocument extends NewLegalDocument {
  id: string;
  createdAt: Date;
  updatedAt: Date;
}

export default legalDocuments;