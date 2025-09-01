// Enhanced Authentication Store with Role-Based Access Control
// Manages user authentication state, permissions, and session management

import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { User } from '../server/db/schema-postgres';
import { AccessControl, type UserRole, type Permission } from './roles';

export interface AuthUser extends Partial<User> {
  id: string;
  email: string;
  role: UserRole;
  name?: string;
  firstName?: string;
  lastName?: string;
  isActive: boolean;
  avatarUrl?: string;
  emailVerified?: boolean;
}

export interface AuthSession {
  id: string;
  userId: string;
  expiresAt: Date;
}

export interface AuthState {
  user: AuthUser | null;
  session: AuthSession | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  permissions: Permission[];
  lastActivity: Date | null;
  csrfToken?: string;
}

// Initial auth state
const initialState: AuthState = {
  user: null,
  session: null,
  isLoading: true,
  isAuthenticated: false,
  permissions: [],
  lastActivity: null,
};

// Create writable store for auth state
export const authState = writable<AuthState>(initialState);
;
// Create derived stores for common auth checks
export const isAuthenticated = derived(authState, $auth => $auth.isAuthenticated);
export const currentUser = derived(authState, $auth => $auth.user);
export const userRole = derived(authState, $auth => $auth.user?.role || 'viewer');
export const userPermissions = derived(authState, $auth => $auth.permissions);
export const isLoading = derived(authState, $auth => $auth.isLoading);
;
// Session management constants
const SESSION_CHECK_INTERVAL = 60000; // Check session every minute
const SESSION_WARNING_TIME = 5 * 60 * 1000; // Warn 5 minutes before expiration
const ACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes of inactivity

export class AuthStore {
  private static sessionCheckInterval: NodeJS.Timeout | null = null;
  private static activityTimeout: NodeJS.Timeout | null = null;

  /**
   * Initialize the auth store and start session management
   */
  static async initialize(): Promise<void> {
    if (!browser) return;

    authState.update(state => ({ ...state, isLoading: true }));

    try {
      // Check if there's an existing session
      await this.checkSession();
      
      // Start session monitoring
      this.startSessionMonitoring();
      
      // Setup activity tracking
      this.setupActivityTracking();
      
    } catch (error: any) {
      console.error('Auth initialization failed:', error);
      this.clearAuth();
    } finally {
      authState.update(state => ({ ...state, isLoading: false }));
    }
  }

  /**
   * Login with email and password
   */
  static async login(email: string, password: string, rememberMe = false): Promise<{ success: boolean; error?: string; requiresMFA?: boolean }> {
    authState.update(state => ({ ...state, isLoading: true }));

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, rememberMe }),
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok && result.success) {
        // Update auth state with user data
        await this.updateAuthState(result.user, result.session);
        
        // Track login activity
        this.trackActivity('login');
        
        return { success: true };
      } else {
        return { 
          success: false, 
          error: result.error || 'Login failed',
          requiresMFA: result.requiresMFA 
        };
      }
    } catch (error: any) {
      console.error('Login error:', error);
      return { success: false, error: 'Network error during login' };
    } finally {
      authState.update(state => ({ ...state, isLoading: false }));
    }
  }

  /**
   * Register a new user account
   */
  static async register(userData: {
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
    role?: UserRole;
  }): Promise<{ success: boolean; error?: string; requiresVerification?: boolean }> {
    authState.update(state => ({ ...state, isLoading: true }));

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok && result.success) {
        // If auto-login after registration
        if (result.user && result.session) {
          await this.updateAuthState(result.user, result.session);
        }
        
        return { 
          success: true, 
          requiresVerification: result.requiresVerification 
        };
      } else {
        return { 
          success: false, 
          error: result.error || 'Registration failed' 
        };
      }
    } catch (error: any) {
      console.error('Registration error:', error);
      return { success: false, error: 'Network error during registration' };
    } finally {
      authState.update(state => ({ ...state, isLoading: false }));
    }
  }

  /**
   * Logout and clear session
   */
  static async logout(): Promise<void> {
    authState.update(state => ({ ...state, isLoading: true }));

    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include'
      });
    } catch (error: any) {
      console.error('Logout error:', error);
    } finally {
      this.clearAuth();
      this.stopSessionMonitoring();
      
      // Redirect to login page
      if (browser) {
        window.location.href = '/login';
      }
    }
  }

  /**
   * Check current session validity
   */
  static async checkSession(): Promise<boolean> {
    if (!browser) return false;

    try {
      const response = await fetch('/api/auth/session', {
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok && result.user && result.session) {
        await this.updateAuthState(result.user, result.session);
        return true;
      } else {
        this.clearAuth();
        return false;
      }
    } catch (error: any) {
      console.error('Session check error:', error);
      this.clearAuth();
      return false;
    }
  }

  /**
   * Update user profile
   */
  static async updateProfile(updates: Partial<AuthUser>): Promise<{ success: boolean; error?: string }> {
    const currentState = get(authState);
    if (!currentState.isAuthenticated || !currentState.user) {
      return { success: false, error: 'Not authenticated' };
    }

    try {
      const response = await fetch('/api/user/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok && result.success) {
        // Update local user data
        authState.update(state => ({
          ...state,
          user: state.user ? { ...state.user, ...result.user } : null
        }));
        
        return { success: true };
      } else {
        return { success: false, error: result.error || 'Profile update failed' };
      }
    } catch (error: any) {
      console.error('Profile update error:', error);
      return { success: false, error: 'Network error during profile update' };
    }
  }

  /**
   * Change user password
   */
  static async changePassword(currentPassword: string, newPassword: string): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch('/api/auth/change-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ currentPassword, newPassword }),
        credentials: 'include'
      });

      const result = await response.json();
      return { 
        success: response.ok && result.success, 
        error: result.error 
      };
    } catch (error: any) {
      console.error('Password change error:', error);
      return { success: false, error: 'Network error during password change' };
    }
  }

  /**
   * Check if user has specific permission
   */
  static hasPermission(permission: Permission): boolean {
    const state = get(authState);
    return state.permissions.includes(permission);
  }

  /**
   * Check if user has any of the specified permissions
   */
  static hasAnyPermission(permissions: Permission[]): boolean {
    const state = get(authState);
    return permissions.some(permission => state.permissions.includes(permission));
  }

  /**
   * Check if user has all of the specified permissions
   */
  static hasAllPermissions(permissions: Permission[]): boolean {
    const state = get(authState);
    return permissions.every(permission => state.permissions.includes(permission));
  }

  /**
   * Check if user can access a resource
   */
  static canAccessResource(
    permission: Permission,
    resourceOwnerId?: string,
    isPublic = false
  ): boolean {
    const state = get(authState);
    if (!state.user) return false;

    return AccessControl.canAccessResource(
      state.user.role,
      permission,
      resourceOwnerId,
      state.user.id,
      isPublic
    );
  }

  /**
   * Get user's role hierarchy level
   */
  static getRoleHierarchy(): number {
    const state = get(authState);
    if (!state.user) return 0;
    
    const role = state.user.role;
    return AccessControl.getRolePermissions(role).length;
  }

  /**
   * Private: Update auth state with user and session data
   */
  private static async updateAuthState(user: AuthUser, session: AuthSession): Promise<void> {
    // Get user permissions based on role
    const permissions = AccessControl.getRolePermissions(user.role);

    authState.update(state => ({
      ...state,
      user,
      session,
      isAuthenticated: true,
      permissions,
      lastActivity: new Date(),
      isLoading: false
    }));
  }

  /**
   * Private: Clear authentication state
   */
  private static clearAuth(): void {
    authState.set({
      ...initialState,
      isLoading: false
    });
  }

  /**
   * Private: Start session monitoring
   */
  private static startSessionMonitoring(): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
    }

    this.sessionCheckInterval = setInterval(async () => {
      const state = get(authState);
      
      if (state.session) {
        const now = new Date();
        const expiresAt = new Date(state.session.expiresAt);
        const timeUntilExpiry = expiresAt.getTime() - now.getTime();

        // Warn user if session expires soon
        if (timeUntilExpiry <= SESSION_WARNING_TIME && timeUntilExpiry > 0) {
          this.showSessionWarning(timeUntilExpiry);
        }

        // Check session validity
        if (timeUntilExpiry <= 0) {
          await this.checkSession();
        }
      }
    }, SESSION_CHECK_INTERVAL);
  }

  /**
   * Private: Stop session monitoring
   */
  private static stopSessionMonitoring(): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
      this.sessionCheckInterval = null;
    }
    
    if (this.activityTimeout) {
      clearTimeout(this.activityTimeout);
      this.activityTimeout = null;
    }
  }

  /**
   * Private: Setup activity tracking
   */
  private static setupActivityTracking(): void {
    if (!browser) return;

    // Track user activity
    const trackActivity = () => {
      this.trackActivity('interaction');
      this.resetActivityTimeout();
    };

    // Add event listeners for user activity
    ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(event => {
      document.addEventListener(event, trackActivity, { passive: true });
    });

    this.resetActivityTimeout();
  }

  /**
   * Private: Track user activity
   */
  private static trackActivity(type: string): void {
    authState.update(state => ({
      ...state,
      lastActivity: new Date()
    }));

    // Send activity ping to server occasionally
    if (type === 'login' || Math.random() < 0.1) {
      fetch('/api/auth/activity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, timestamp: new Date() }),
        credentials: 'include'
      }).catch(() => {}); // Silently fail
    }
  }

  /**
   * Private: Reset activity timeout
   */
  private static resetActivityTimeout(): void {
    if (this.activityTimeout) {
      clearTimeout(this.activityTimeout);
    }

    this.activityTimeout = setTimeout(() => {
      // Auto-logout after inactivity
      this.logout();
    }, ACTIVITY_TIMEOUT);
  }

  /**
   * Private: Show session expiration warning
   */
  private static showSessionWarning(timeRemaining: number): void {
    const minutes = Math.ceil(timeRemaining / 60000);
    
    // You can replace this with a proper notification system
    if (browser && window.confirm(
      `Your session will expire in ${minutes} minute${minutes !== 1 ? 's' : ''}. ` +
      'Would you like to extend your session?'
    )) {
      // Refresh session by making a request
      this.checkSession();
    }
  }
}

// Export store instances and utilities
export { authState as default };