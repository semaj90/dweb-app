// Modern authentication store using Svelte 5 runes
// Integrates with Lucia, MCP GPU orchestrator, and legal AI features

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator.js';

export interface User {
  id: string;
  email: string;
  name?: string;
  firstName?: string;
  lastName?: string;
  role: string;
  isActive: boolean;
  avatarUrl?: string;
  emailVerified?: Date | boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface AuthState {
  user: User | null;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
}

// Reactive authentication state using $state rune (browser-only)
const authState = browser ? $state<AuthState>({
  user: null,
  loading: false,
  error: null,
  isAuthenticated: false
}) : {
  user: null,
  loading: false,
  error: null,
  isAuthenticated: false
};

// Derived state functions for common auth checks
export function isAdmin(): boolean {
  return authState.user?.role === 'admin' || authState.user?.role === 'lead_prosecutor' || false;
}

export function canCreateCases(): boolean {
  return authState.user?.role === 'admin' || 
         authState.user?.role === 'lead_prosecutor' || 
         authState.user?.role === 'prosecutor' || false;
}

export function canViewEvidence(): boolean {
  return authState.isAuthenticated;
}

// Auth service with reactive methods
export class AuthService {
  // Get current state (reactive)
  get state() {
    return authState;
  }

  // Initialize auth state (checks existing session)
  async initialize() {
    if (!browser) return;
    
    authState.loading = true;
    authState.error = null;

    try {
      const response = await fetch('/api/auth/me', {
        credentials: 'include'
      });

      if (response.ok) {
        const { user } = await response.json();
        authState.user = user;
        authState.isAuthenticated = true;
        
        // Log successful session restore with AI analytics
        await mcpGPUOrchestrator.routeAPIRequest(
          '/api/analytics/session-restore',
          { userId: user.id, timestamp: new Date().toISOString() },
          { userId: user.id, analyticsLevel: 'session' }
        );
      } else {
        authState.user = null;
        authState.isAuthenticated = false;
      }
    } catch (error: any) {
      console.error('Auth initialization error:', error);
      authState.error = 'Failed to initialize authentication';
      authState.user = null;
      authState.isAuthenticated = false;
    } finally {
      authState.loading = false;
    }
  }

  // Login with email and password
  async login(email: string, password: string): Promise<{ success: boolean; error?: string }> {
    authState.loading = true;
    authState.error = null;

    try {
      // Import session manager dynamically to avoid circular imports
      const { sessionManager } = await import('./sessionManager.svelte.js');

      // Pre-login security analysis using GPU orchestrator
      const securityContext = {
        email,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        ipInfo: await this.getClientInfo()
      };

      await mcpGPUOrchestrator.routeAPIRequest(
        '/api/security/pre-login-analysis',
        securityContext,
        { securityLevel: 'authentication' }
      );

      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok) {
        authState.user = result.user;
        authState.isAuthenticated = true;
        authState.error = null;

        // Start session management with XState
        const sessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        await sessionManager.startSession(result.user, sessionId);

        // Post-login AI analysis for user behavior patterns
        await mcpGPUOrchestrator.processLegalDocument(
          `Successful login: ${email} at ${new Date().toISOString()}`,
          {
            userId: result.user.id,
            includeRAG: false,
            includeGraph: true,
            generateSummary: false
          }
        );

        return { success: true };
      } else {
        authState.error = result.error;
        
        // Log failed login attempt for security monitoring
        await mcpGPUOrchestrator.routeAPIRequest(
          '/api/security/failed-login',
          { ...securityContext, error: result.error },
          { securityLevel: 'high' }
        );

        return { success: false, error: result.error };
      }
    } catch (error: any) {
      const errorMessage = 'Network error during login';
      authState.error = errorMessage;
      console.error('Login error:', error);
      return { success: false, error: errorMessage };
    } finally {
      authState.loading = false;
    }
  }

  // Register new user
  async register(userData: {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
  }): Promise<{ success: boolean; error?: string }> {
    authState.loading = true;
    authState.error = null;

    try {
      // Pre-registration analysis
      const registrationContext = {
        email: userData.email,
        firstName: userData.firstName,
        lastName: userData.lastName,
        timestamp: new Date().toISOString()
      };

      await mcpGPUOrchestrator.routeAPIRequest(
        '/api/analytics/registration-attempt',
        registrationContext,
        { analyticsLevel: 'registration' }
      );

      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok) {
        authState.user = result.user;
        authState.isAuthenticated = true;
        authState.error = null;

        // Log successful registration with AI context
        await mcpGPUOrchestrator.processLegalDocument(
          `New user registration: ${userData.email}`,
          {
            userId: result.user.id,
            includeRAG: false,
            includeGraph: true,
            generateSummary: true
          }
        );

        return { success: true };
      } else {
        authState.error = result.error;
        return { success: false, error: result.error };
      }
    } catch (error: any) {
      const errorMessage = 'Network error during registration';
      authState.error = errorMessage;
      console.error('Registration error:', error);
      return { success: false, error: errorMessage };
    } finally {
      authState.loading = false;
    }
  }

  // Logout current user
  async logout() {
    authState.loading = true;

    try {
      // Import session manager dynamically
      const { sessionManager } = await import('./sessionManager.svelte.js');

      // Log logout event for analytics
      if (authState.user) {
        await mcpGPUOrchestrator.routeAPIRequest(
          '/api/analytics/logout',
          { 
            userId: authState.user.id,
            timestamp: new Date().toISOString(),
            sessionDuration: this.calculateSessionDuration()
          },
          { userId: authState.user.id, analyticsLevel: 'session' }
        );
      }

      // End session management
      await sessionManager.endSession();

      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include'
      });

      // Clear state regardless of response
      authState.user = null;
      authState.isAuthenticated = false;
      authState.error = null;

      // Redirect to login page
      await goto('/login');
    } catch (error: any) {
      console.error('Logout error:', error);
      // Still clear state on error
      authState.user = null;
      authState.isAuthenticated = false;
    } finally {
      authState.loading = false;
    }
  }

  // Update user profile
  async updateProfile(updates: Partial<User>): Promise<{ success: boolean; error?: string }> {
    if (!authState.user) return { success: false, error: 'Not authenticated' };

    authState.loading = true;

    try {
      const response = await fetch('/api/auth/profile', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok) {
        authState.user = { ...authState.user, ...result.user };
        
        // Log profile update for audit trail
        await mcpGPUOrchestrator.routeAPIRequest(
          '/api/analytics/profile-update',
          { 
            userId: authState.user.id,
            changes: Object.keys(updates),
            timestamp: new Date().toISOString()
          },
          { userId: authState.user.id, analyticsLevel: 'profile' }
        );

        return { success: true };
      } else {
        authState.error = result.error;
        return { success: false, error: result.error };
      }
    } catch (error: any) {
      const errorMessage = 'Failed to update profile';
      authState.error = errorMessage;
      console.error('Profile update error:', error);
      return { success: false, error: errorMessage };
    } finally {
      authState.loading = false;
    }
  }

  // Check if user has specific permission
  hasPermission(permission: string): boolean {
    if (!authState.user) return false;

    const rolePermissions = {
      admin: ['all'],
      lead_prosecutor: ['manage_cases', 'view_all_evidence', 'assign_tasks', 'generate_reports'],
      prosecutor: ['create_cases', 'manage_own_cases', 'view_evidence', 'generate_reports'],
      investigator: ['view_cases', 'add_evidence', 'view_evidence'],
      analyst: ['view_cases', 'analyze_evidence', 'generate_reports'],
      viewer: ['view_cases', 'view_evidence']
    };

    const userPermissions = rolePermissions[authState.user.role as keyof typeof rolePermissions] || [];
    return userPermissions.includes('all') || userPermissions.includes(permission);
  }

  // Require authentication (for use in components)
  requireAuth(): User {
    if (!authState.isAuthenticated || !authState.user) {
      goto('/login');
      throw new Error('Authentication required');
    }
    return authState.user;
  }

  // Private helper methods
  private async getClientInfo() {
    try {
      // Get basic client information for security analysis
      return {
        screen: `${screen.width}x${screen.height}`,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        language: navigator.language,
        platform: navigator.platform
      };
    } catch {
      return {};
    }
  }

  private calculateSessionDuration(): number {
    // Calculate session duration in minutes
    // This would need to be tracked from login time
    return 0; // Placeholder
  }
}

// Create singleton auth service
export const authService = new AuthService();
;
// Initialize auth state when module loads
if (browser) {
  authService.initialize();
}

// Export reactive getters for use in components
export const user = () => authState.user;
export const isAuthenticated = () => authState.isAuthenticated;
export const isLoading = () => authState.loading;
export const authError = () => authState.error;