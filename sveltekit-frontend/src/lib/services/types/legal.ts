// Basic LegalDocument type placeholder to satisfy imports
export interface LegalDocument {
  id: string;
  title: string;
  content?: string;
  score?: number;
  summary?: string;
  excerpt?: string;
  metadata?: Record<string, any>;
  type?: string;
  createdAt?: Date;
  updatedAt?: Date;
}
