/**
 * Authentication Store - Enhanced with XState + Production Service Client
 * Integrates with Legal AI Platform authentication system
 */

import { browser } from '$app/environment';
import { createActor, type ActorRefFrom } from 'xstate';
import { authMachine, type AuthContext } from '$lib/machines/auth-machine';
import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator';

export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  name: string;
  role: string;
  department?: string;
  jurisdiction?: string;
  permissions?: string[];
  isActive?: boolean;
  emailVerified?: boolean;
  avatarUrl?: string;
}

export interface Session {
  id: string;
  expiresAt: Date;
  fresh: boolean;
  ipAddress?: string;
  userAgent?: string;
  deviceInfo?: any;
}

export interface AuthState {
  user: User | null;
  session: Session | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  twoFactorRequired: boolean;
  machineState: string;
  loginAttempts: number;
  lockoutUntil?: Date;
}

// Enhanced reactive auth store using Svelte 5 runes + XState
let authState = $state<AuthState>({
  user: null,
  session: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  twoFactorRequired: false,
  machineState: 'idle',
  loginAttempts: 0,
  lockoutUntil: undefined
});

// XState machine actor
let authActor: ActorRefFrom<typeof authMachine> | null = null;

// Initialize XState machine in browser
if (browser) {
  authActor = createActor(authMachine);
  
  // Subscribe to machine state changes
  authActor.subscribe((state) => {
    authState.machineState = state.value as string;
    authState.user = state.context.user as User | null;
    authState.session = state.context.session as Session | null;
    authState.isAuthenticated = state.context.user !== null;
    authState.isLoading = state.context.isLoading;
    authState.error = state.context.error || null;
    authState.twoFactorRequired = state.context.twoFactorRequired;
    authState.loginAttempts = state.context.loginAttempts;
    authState.lockoutUntil = state.context.lockoutUntil;
    
    // Handle side effects
    if (state.matches('authenticated')) {
      // Store session in localStorage if remember me is enabled
      if (state.context.session && browser) {
        localStorage.setItem('legal_ai_session', JSON.stringify(state.context.session));
      }
    } else if (state.matches('idle') && state.context.user === null) {
      // Clear stored session on logout
      if (browser) {
        localStorage.removeItem('legal_ai_session');
      }
    }
  });
  
  // Start the machine
  authActor.start();
  
  // Try to restore session from localStorage
  try {
    const storedSession = localStorage.getItem('legal_ai_session');
    if (storedSession) {
      const sessionData = JSON.parse(storedSession);
      // Check if session is still valid
      if (new Date(sessionData.expiresAt) > new Date()) {
        authActor.send({ type: 'VALIDATE_SESSION', data: sessionData });
      } else {
        localStorage.removeItem('legal_ai_session');
      }
    }
  } catch (err) {
    console.warn('Failed to restore session:', err);
    localStorage.removeItem('legal_ai_session');
  }
}

// Device fingerprinting for security
async function getDeviceFingerprint(): Promise<string> {
  if (!browser) return '';
  
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.textBaseline = 'top';
    ctx.font = '14px Arial';
    ctx.fillText('Legal AI Auth', 2, 2);
  }
  
  const fingerprint = {
    userAgent: navigator.userAgent,
    language: navigator.language,
    languages: navigator.languages,
    platform: navigator.platform,
    screenResolution: `${screen.width}x${screen.height}`,
    colorDepth: screen.colorDepth,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    canvas: canvas.toDataURL(),
    cookieEnabled: navigator.cookieEnabled,
    onlineStatus: navigator.onLine,
    doNotTrack: navigator.doNotTrack,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: (navigator as any).deviceMemory || 0,
    connection: (navigator as any).connection?.effectiveType || 'unknown'
  };
  
  return btoa(JSON.stringify(fingerprint));
}

export const authStore = {
  get state() {
    return authState;
  },
  
  get user() {
    return authState.user;
  },
  
  get session() {
    return authState.session;
  },
  
  get isAuthenticated() {
    return authState.isAuthenticated;
  },
  
  get isLoading() {
    return authState.isLoading;
  },
  
  get error() {
    return authState.error;
  },
  
  get twoFactorRequired() {
    return authState.twoFactorRequired;
  },
  
  get machineState() {
    return authState.machineState;
  },
  
  get loginAttempts() {
    return authState.loginAttempts;
  },
  
  get isLocked() {
    return authState.lockoutUntil ? new Date() < authState.lockoutUntil : false;
  },
  
  // Enhanced login with GPU security analysis
  login: async (email: string, password: string, options: {
    rememberMe?: boolean;
    twoFactorCode?: string;
    enableGPUAuth?: boolean;
  } = {}) => {
    if (!authActor) throw new Error('Auth machine not initialized');
    
    const deviceInfo = {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      fingerprint: await getDeviceFingerprint()
    };
    
    // GPU-enhanced authentication if enabled
    if (options.enableGPUAuth && browser) {
      try {
        const securityCheck = await mcpGPUOrchestrator.dispatchGPUTask({
          id: `login_security_${Date.now()}`,
          type: 'security_analysis',
          priority: 'high',
          data: {
            email,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            fingerprint: deviceInfo.fingerprint
          },
          context: {
            action: 'login_attempt',
            enhancedSecurity: true
          },
          config: {
            useGPU: true,
            model: 'gemma3-legal',
            protocol: 'quic'
          }
        });
        
        if (securityCheck.riskScore && securityCheck.riskScore > 0.8) {
          authState.error = 'Security verification failed. Please try again.';
          return;
        }
      } catch (error) {
        console.warn('GPU security check failed, proceeding with standard auth:', error);
      }
    }
    
    authActor.send({
      type: 'START_LOGIN',
      data: {
        email,
        password,
        rememberMe: options.rememberMe,
        twoFactorCode: options.twoFactorCode,
        deviceInfo
      }
    });
  },
  
  // Enhanced registration with legal professional validation
  register: async (data: {
    email: string;
    firstName: string;
    lastName: string;
    password: string;
    role: string;
    department: string;
    jurisdiction: string;
    badgeNumber?: string;
    enableTwoFactor?: boolean;
  }) => {
    if (!authActor) throw new Error('Auth machine not initialized');
    
    const deviceInfo = {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      fingerprint: await getDeviceFingerprint(),
      securityScore: 100 // Default high score for new registrations
    };
    
    // GPU-enhanced validation for legal professionals
    if (browser) {
      try {
        const validationCheck = await mcpGPUOrchestrator.dispatchGPUTask({
          id: `register_validation_${Date.now()}`,
          type: 'security_validation',
          priority: 'high',
          data: {
            email: data.email,
            firstName: data.firstName,
            lastName: data.lastName,
            role: data.role,
            department: data.department,
            jurisdiction: data.jurisdiction,
            badgeNumber: data.badgeNumber,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            fingerprint: deviceInfo.fingerprint
          },
          context: {
            action: 'registration_attempt',
            enhancedValidation: true,
            legalProfessionalCheck: true
          },
          config: {
            useGPU: true,
            model: 'gemma3-legal',
            protocol: 'quic'
          }
        });
        
        if (validationCheck.riskScore && validationCheck.riskScore > 0.9) {
          authState.error = 'Registration validation failed. Please verify your information.';
          return;
        }
        
        if (validationCheck.legalVerification && !validationCheck.legalVerification.verified) {
          authState.error = 'Unable to verify legal professional credentials. Please contact support.';
          return;
        }
      } catch (error) {
        console.warn('GPU validation failed, proceeding with standard registration:', error);
      }
    }
    
    authActor.send({
      type: 'START_REGISTRATION',
      data: {
        ...data,
        deviceInfo
      }
    });
  },
  
  logout: async () => {
    if (!authActor) {
      // Fallback for when machine is not initialized
      authState.user = null;
      authState.session = null;
      authState.isAuthenticated = false;
      authState.error = null;
      authState.twoFactorRequired = false;
      if (browser) {
        localStorage.removeItem('legal_ai_session');
      }
      return;
    }
    
    authActor.send({ type: 'LOGOUT' });
  },
  
  validateSession: async () => {
    if (!authActor) throw new Error('Auth machine not initialized');
    authActor.send({ type: 'VALIDATE_SESSION' });
  },
  
  submitTwoFactor: async (code: string) => {
    if (!authActor) throw new Error('Auth machine not initialized');
    authActor.send({
      type: 'TWO_FACTOR_SUCCESS',
      data: { code }
    });
  },
  
  updateProfile: async (data: Partial<User>) => {
    if (!authActor) throw new Error('Auth machine not initialized');
    authActor.send({
      type: 'UPDATE_PROFILE',
      data
    });
  },
  
  clearError: () => {
    authState.error = null;
  },
  
  // Session management utilities
  refreshSession: async () => {
    if (!browser || !authState.isAuthenticated) return;
    
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        credentials: 'include'
      });
      
      if (response.ok) {
        const sessionData = await response.json();
        if (authActor) {
          authActor.send({
            type: 'SESSION_REFRESHED',
            data: sessionData
          });
        }
      } else {
        if (authActor) {
          authActor.send({ type: 'SESSION_EXPIRED' });
        }
      }
    } catch (err) {
      console.error('Session refresh failed:', err);
      if (authActor) {
        authActor.send({ type: 'SESSION_EXPIRED' });
      }
    }
  },
  
  isSessionExpired: () => {
    if (!authState.session?.expiresAt) return true;
    return new Date() > new Date(authState.session.expiresAt);
  },
  
  getSessionTimeRemaining: () => {
    if (!authState.session?.expiresAt) return 0;
    return Math.max(0, new Date(authState.session.expiresAt).getTime() - Date.now());
  },
  
  formatSessionExpiry: () => {
    const remaining = authStore.getSessionTimeRemaining();
    if (remaining === 0) return 'Expired';
    
    const hours = Math.floor(remaining / (1000 * 60 * 60));
    const minutes = Math.floor((remaining % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }
};

// Auto-refresh session every 15 minutes
if (browser) {
  setInterval(() => {
    if (authState.isAuthenticated && !authStore.isSessionExpired()) {
      authStore.refreshSession().catch(console.error);
    }
  }, 15 * 60 * 1000); // 15 minutes
}