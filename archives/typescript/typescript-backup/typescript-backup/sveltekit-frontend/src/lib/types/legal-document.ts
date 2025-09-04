// Canonical unified LegalDocument type
// Consolidates many scattered interface variants across the codebase.
// Future refactors: replace other duplicate declarations with imports from this file.
export interface LegalDocumentUnified {
  id: string;
  title: string;
  content?: string;
  type?: string; // e.g. 'legal', 'contract', etc.
  summary?: string;
  excerpt?: string;
  score?: number;
  tags?: string[];
  jurisdiction?: string;
  court?: string;
  citation?: string;
  metadata?: Record<string, any>;
  createdAt?: Date;
  updatedAt?: Date;
}

export type LegalDocument = LegalDocumentUnified;

export function isLegalDocument(value: any): value is LegalDocumentUnified {
  return (
    value &&
    typeof value === 'object' &&
    typeof value.id === 'string' &&
    typeof value.title === 'string'
  );
}

export function mergeLegalDocuments<T extends Partial<LegalDocumentUnified>>(
  base: LegalDocumentUnified,
  patch: T
): LegalDocumentUnified {
  return { ...base, ...patch };
}
