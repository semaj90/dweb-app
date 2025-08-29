// XState Machine for Session Management with Production Services Integration
import { createMachine, assign } from "xstate";
import { productionServiceClient, services } from '../services/productionServiceClient';
import type { User } from '../stores/auth.svelte';

// Session context interface
export interface SessionContext {
  user: User | null;
  sessionId: string | null;
  expiresAt: Date | null;
  lastActivity: Date | null;
  securityLevel: 'standard' | 'elevated' | 'secure';
  permissions: string[];
  currentRoute: string;
  deviceFingerprint?: string;
  sessionHealth: {
    isValid: boolean;
    warningCount: number;
    lastHealthCheck: Date | null;
  };
  analyticsData: {
    loginTime: Date | null;
    activityCount: number;
    featuresUsed: string[];
  };
}

// Session events
type SessionEvent =
  | { type: "AUTHENTICATE"; user: User; sessionId: string }
  | { type: "REFRESH_SESSION" }
  | { type: "LOGOUT" }
  | { type: "SESSION_EXPIRED" }
  | { type: "ACTIVITY"; route: string; action: string }
  | { type: "SECURITY_CHECK" }
  | { type: "PERMISSION_CHECK"; permission: string }
  | { type: "EXTEND_SESSION" }
  | { type: "ELEVATE_SECURITY"; reason: string }
  | { type: "HEALTH_CHECK" };

export const sessionMachine = createMachine({
  id: "sessionManager",
  initial: "unauthenticated",
  context: {
    user: null,
    sessionId: null,
    expiresAt: null,
    lastActivity: null,
    securityLevel: 'standard',
    permissions: [],
    currentRoute: '/',
    sessionHealth: {
      isValid: false,
      warningCount: 0,
      lastHealthCheck: null
    },
    analyticsData: {
      loginTime: null,
      activityCount: 0,
      featuresUsed: []
    }
  } as SessionContext,
  types: {} as {
    context: SessionContext;
    events: SessionEvent;
  },
  states: {
    unauthenticated: {
      entry: ["clearSessionData"],
      on: {
        AUTHENTICATE: {
          target: "authenticating",
          actions: assign({
            user: ({ event }) => (event as any).user,
            sessionId: ({ event }) => (event as any).sessionId,
            analyticsData: ({ context }) => ({
              ...context.analyticsData,
              loginTime: new Date()
            })
          })
        }
      }
    },

    authenticating: {
      invoke: {
        src: "validateSession",
        input: ({ context }) => ({
          user: context.user,
          sessionId: context.sessionId
        }),
        onDone: {
          target: "authenticated",
          actions: assign({
            expiresAt: ({ event }) => new Date((event as any).data.expiresAt),
            permissions: ({ event }) => (event as any).data.permissions,
            lastActivity: () => new Date(),
            sessionHealth: ({ context }) => ({
              ...context.sessionHealth,
              isValid: true,
              lastHealthCheck: new Date()
            })
          })
        },
        onError: "unauthenticated"
      }
    },

    authenticated: {
      initial: "active",
      entry: ["logSessionStart"],
      on: {
        LOGOUT: "logging_out",
        SESSION_EXPIRED: "expired",
        SECURITY_CHECK: "security_validation",
        HEALTH_CHECK: "health_checking"
      },
      states: {
        active: {
          on: {
            ACTIVITY: {
              actions: ["recordActivity", "updateLastActivity"]
            },
            REFRESH_SESSION: "refreshing",
            EXTEND_SESSION: "extending",
            PERMISSION_CHECK: "checking_permissions",
            ELEVATE_SECURITY: "elevating_security"
          },
          // Automatic session timeout check every 5 minutes
          after: {
            300000: {
              target: "checking_timeout",
              actions: ["checkSessionTimeout"]
            }
          }
        },

        refreshing: {
          invoke: {
            src: "refreshSession",
            input: ({ context }) => ({
              sessionId: context.sessionId,
              user: context.user
            }),
            onDone: {
              target: "active",
              actions: assign({
                expiresAt: ({ event }) => new Date((event as any).data.expiresAt),
                lastActivity: () => new Date(),
                sessionHealth: ({ context }) => ({
                  ...context.sessionHealth,
                  lastHealthCheck: new Date()
                })
              })
            },
            onError: "expired"
          }
        },

        extending: {
          invoke: {
            src: "extendSession",
            input: ({ context }) => ({
              sessionId: context.sessionId,
              requestedDuration: 3600000 // 1 hour
            }),
            onDone: {
              target: "active",
              actions: assign({
                expiresAt: ({ event }) => new Date((event as any).data.expiresAt)
              })
            },
            onError: "active"
          }
        },

        checking_permissions: {
          invoke: {
            src: "validatePermissions",
            input: ({ context, event }) => ({
              user: context.user,
              permission: (event as any).permission,
              securityLevel: context.securityLevel
            }),
            onDone: "active",
            onError: "permission_denied"
          }
        },

        permission_denied: {
          after: {
            2000: "active"
          }
        },

        elevating_security: {
          invoke: {
            src: "elevateSecurityLevel",
            input: ({ context, event }) => ({
              sessionId: context.sessionId,
              reason: (event as any).reason,
              currentLevel: context.securityLevel
            }),
            onDone: {
              target: "active",
              actions: assign({
                securityLevel: ({ event }) => (event as any).data.newLevel
              })
            },
            onError: "active"
          }
        },

        checking_timeout: {
          invoke: {
            src: "checkSessionValidity",
            input: ({ context }) => ({
              sessionId: context.sessionId,
              expiresAt: context.expiresAt,
              lastActivity: context.lastActivity
            }),
            onDone: "active",
            onError: "expired"
          }
        },

        security_validation: {
          invoke: {
            src: "performSecurityCheck",
            input: ({ context }) => ({
              user: context.user,
              deviceFingerprint: context.deviceFingerprint,
              sessionHealth: context.sessionHealth
            }),
            onDone: {
              target: "active",
              actions: assign({
                sessionHealth: ({ context, event }) => ({
                  ...context.sessionHealth,
                  warningCount: (event as any).data.warningCount,
                  lastHealthCheck: new Date()
                })
              })
            },
            onError: {
              target: "security_compromised",
              actions: assign({
                sessionHealth: ({ context }) => ({
                  ...context.sessionHealth,
                  isValid: false,
                  warningCount: context.sessionHealth.warningCount + 1
                })
              })
            }
          }
        },

        security_compromised: {
          entry: ["alertSecurityBreach"],
          after: {
            5000: "logging_out"
          }
        },

        health_checking: {
          invoke: {
            src: "performHealthCheck",
            input: ({ context }) => ({
              sessionId: context.sessionId,
              analyticsData: context.analyticsData
            }),
            onDone: {
              target: "active",
              actions: assign({
                sessionHealth: ({ context, event }) => ({
                  ...context.sessionHealth,
                  lastHealthCheck: new Date(),
                  isValid: (event as any).data.healthy
                })
              })
            },
            onError: "active"
          }
        }
      }
    },

    logging_out: {
      invoke: {
        src: "performLogout",
        input: ({ context }) => ({
          sessionId: context.sessionId,
          user: context.user,
          sessionDuration: context.analyticsData.loginTime ? 
            Date.now() - context.analyticsData.loginTime.getTime() : 0
        }),
        onDone: "unauthenticated",
        onError: "unauthenticated"
      }
    },

    expired: {
      entry: ["logSessionExpired"],
      after: {
        3000: "unauthenticated"
      }
    }
  }
});

// Service implementations for session management
export const sessionServices = {
  validateSession: async ({ user, sessionId }: { user: User; sessionId: string }) => {
    try {
      // Use production service client for session validation
      const result = await services.triggerXStateEvent('session.validate', {
        userId: user.id,
        sessionId
      });

      // Calculate expiration (24 hours from now)
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);

      // Get user permissions based on role
      const permissions = getUserPermissions(user.role);

      return {
        expiresAt: expiresAt.toISOString(),
        permissions,
        valid: true
      };
    } catch (error: any) {
      console.error('Session validation failed:', error);
      throw error;
    }
  },

  refreshSession: async ({ sessionId, user }: { sessionId: string; user: User }) => {
    try {
      const result = await services.triggerXStateEvent('session.refresh', {
        sessionId,
        userId: user.id
      });

      return {
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
      };
    } catch (error: any) {
      console.error('Session refresh failed:', error);
      throw error;
    }
  },

  extendSession: async ({ sessionId, requestedDuration }: { 
    sessionId: string; 
    requestedDuration: number 
  }) => {
    try {
      const result = await services.triggerXStateEvent('session.extend', {
        sessionId,
        duration: requestedDuration
      });

      return {
        expiresAt: new Date(Date.now() + requestedDuration).toISOString()
      };
    } catch (error: any) {
      console.error('Session extension failed:', error);
      throw error;
    }
  },

  validatePermissions: async ({ user, permission, securityLevel }: {
    user: User;
    permission: string;
    securityLevel: string;
  }) => {
    const userPermissions = getUserPermissions(user.role);
    const hasPermission = userPermissions.includes(permission) || userPermissions.includes('all');

    if (!hasPermission) {
      throw new Error(`Permission denied: ${permission}`);
    }

    return { valid: true };
  },

  elevateSecurityLevel: async ({ sessionId, reason, currentLevel }: {
    sessionId: string;
    reason: string;
    currentLevel: string;
  }) => {
    try {
      const result = await services.triggerXStateEvent('security.elevate', {
        sessionId,
        reason,
        currentLevel
      });

      return {
        newLevel: 'elevated' as const
      };
    } catch (error: any) {
      console.error('Security elevation failed:', error);
      throw error;
    }
  },

  checkSessionValidity: async ({ sessionId, expiresAt, lastActivity }: {
    sessionId: string;
    expiresAt: Date | null;
    lastActivity: Date | null;
  }) => {
    const now = new Date();
    
    if (expiresAt && now > expiresAt) {
      throw new Error('Session expired');
    }

    if (lastActivity && (now.getTime() - lastActivity.getTime()) > 4 * 60 * 60 * 1000) {
      throw new Error('Session inactive too long');
    }

    return { valid: true };
  },

  performSecurityCheck: async ({ user, deviceFingerprint, sessionHealth }: {
    user: User;
    deviceFingerprint?: string;
    sessionHealth: any;
  }) => {
    try {
      const result = await services.triggerXStateEvent('security.check', {
        userId: user.id,
        deviceFingerprint,
        healthStatus: sessionHealth
      });

      return {
        warningCount: 0,
        secure: true
      };
    } catch (error: any) {
      console.error('Security check failed:', error);
      throw error;
    }
  },

  performHealthCheck: async ({ sessionId, analyticsData }: {
    sessionId: string;
    analyticsData: any;
  }) => {
    try {
      const result = await services.triggerXStateEvent('session.health', {
        sessionId,
        analytics: analyticsData
      });

      return {
        healthy: true,
        metrics: result
      };
    } catch (error: any) {
      console.error('Health check failed:', error);
      return { healthy: false };
    }
  },

  performLogout: async ({ sessionId, user, sessionDuration }: {
    sessionId: string;
    user: User;
    sessionDuration: number;
  }) => {
    try {
      const result = await services.triggerXStateEvent('session.logout', {
        sessionId,
        userId: user.id,
        duration: sessionDuration
      });

      return { success: true };
    } catch (error: any) {
      console.error('Logout failed:', error);
      return { success: false };
    }
  }
};

// Action implementations
export const sessionActions = {
  clearSessionData: () => {
    // Clear any local session data
    if (typeof window !== 'undefined') {
      localStorage.removeItem('sessionHealth');
      sessionStorage.removeItem('currentRoute');
    }
  },

  logSessionStart: ({ context }: { context: SessionContext }) => {
    console.log('Session started for user:', context.user?.email);
    if (typeof window !== 'undefined') {
      localStorage.setItem('sessionStartTime', Date.now().toString());
    }
  },

  recordActivity: ({ context, event }: { context: SessionContext; event: any }) => {
    const activity = {
      route: event.route,
      action: event.action,
      timestamp: new Date()
    };

    // Log activity for analytics
    console.log('User activity:', activity);
  },

  updateLastActivity: assign({
    lastActivity: () => new Date(),
    currentRoute: ({ event }) => (event as any).route,
    analyticsData: ({ context }) => ({
      ...context.analyticsData,
      activityCount: context.analyticsData.activityCount + 1
    })
  }),

  checkSessionTimeout: ({ context }: { context: SessionContext }) => {
    const now = new Date();
    if (context.expiresAt && now > context.expiresAt) {
      console.warn('Session expired due to timeout');
    }
  },

  alertSecurityBreach: ({ context }: { context: SessionContext }) => {
    console.error('Security breach detected for session:', context.sessionId);
    // Could trigger notifications, alerts, etc.
  },

  logSessionExpired: ({ context }: { context: SessionContext }) => {
    console.log('Session expired for user:', context.user?.email);
  }
};

// Helper function to get user permissions based on role
function getUserPermissions(role: string): string[] {
  const permissions = {
    admin: ['all'],
    lead_prosecutor: ['manage_cases', 'view_all_evidence', 'assign_tasks', 'generate_reports'],
    prosecutor: ['create_cases', 'manage_own_cases', 'view_evidence', 'generate_reports'],
    investigator: ['view_cases', 'add_evidence', 'view_evidence'],
    analyst: ['view_cases', 'analyze_evidence', 'generate_reports'],
    viewer: ['view_cases', 'view_evidence']
  };

  return permissions[role as keyof typeof permissions] || permissions.viewer;
}