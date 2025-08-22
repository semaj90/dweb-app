
/**
 * Authentication Types
 * Standardized types for user sessions and authentication
 */

export interface SessionUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  isActive: boolean;
}

export interface UserSession {
  user: SessionUser | null;
}

export interface SessionValidationResult {
  user: SessionUser | null;
  isValid: boolean;
}

// Type guards for safe type checking
export function isSessionUser(user: any): user is SessionUser {
  return user && 
    typeof user.id === 'string' &&
    typeof user.email === 'string' &&
    typeof user.role === 'string' &&
    typeof user.isActive === 'boolean';
}

export function hasValidSession(locals: App.Locals): locals is App.Locals & { user: SessionUser } {
  return locals.user !== null && isSessionUser(locals.user);
}

export function validateUserSession(locals: App.Locals): SessionUser {
  if (!locals.user) {
    throw new Error('Authentication required');
  }
  
  if (!locals.user.isActive) {
    throw new Error('Account is inactive');
  }
  
  return locals.user;
}

// Additional types for full CRUD system
export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
}

export interface Case {
  id: string;
  userId: string;
  title: string;
  description: string;
  caseNumber: string;
  status: string;
  createdAt: string;
  updatedAt: string;
}

export interface DocumentMetadata {
  id: string;
  caseId: string;
  userId: string;
  originalFilename: string;
  fileSize: number;
  fileType: string;
  minioBucket: string;
  minioKey: string;
  extractedText?: string;
  summary?: string;
  uploadStatus: string;
  processingStatus: string;
  createdAt: string;
}

export interface Evidence {
  id: string;
  caseId: string;
  documentId?: string;
  evidenceType: string;
  title: string;
  description: string;
  relevanceScore: number;
  createdAt: string;
}
