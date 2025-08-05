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
