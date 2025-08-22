// Application State Machine with XState
// Manages global app state, authentication, and navigation

import { createMachine, assign, type InterpreterFrom } from 'xstate';
import { writable } from 'svelte/store';
import { browser } from '$app/environment';

// Types for application state
export interface User {
  id: string;
  email: string;
  role: 'admin' | 'prosecutor' | 'detective' | 'user';
  name: string;
  avatar?: string;
  preferences: {
    theme: 'light' | 'dark' | 'yorha';
    language: string;
    notifications: boolean;
  };
}

export interface AppContext {
  user: User | null;
  isAuthenticated: boolean;
  currentRoute: string;
  notifications: Array<{
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    title: string;
    message: string;
    timestamp: number;
    read: boolean;
  }>;
  settings: {
    apiEndpoint: string;
    enableGPUAcceleration: boolean;
    enableMultiProtocol: boolean;
    maxConcurrentRequests: number;
    cacheEnabled: boolean;
  };
  performance: {
    startTime: number;
    pageLoadTime: number;
    apiResponseTimes: number[];
    errorCount: number;
  };
}

// Application events
export type AppEvent =
  | { type: 'LOGIN'; credentials: { email: string; password: string } }
  | { type: 'LOGOUT' }
  | { type: 'LOGIN_SUCCESS'; user: User }
  | { type: 'LOGIN_FAILURE'; error: string }
  | { type: 'NAVIGATE'; route: string }
  | { type: 'ADD_NOTIFICATION'; notification: Omit<AppContext['notifications'][0], 'id' | 'timestamp'> }
  | { type: 'MARK_NOTIFICATION_READ'; id: string }
  | { type: 'CLEAR_NOTIFICATIONS' }
  | { type: 'UPDATE_SETTINGS'; settings: Partial<AppContext['settings']> }
  | { type: 'UPDATE_USER_PREFERENCES'; preferences: Partial<User['preferences']> }
  | { type: 'RECORD_PERFORMANCE'; metric: string; value: number }
  | { type: 'RESET_SESSION' };

// Default context
const defaultContext: AppContext = {
  user: null,
  isAuthenticated: false,
  currentRoute: '/',
  notifications: [],
  settings: {
    apiEndpoint: browser ? window.location.origin : 'http://localhost:5173',
    enableGPUAcceleration: true,
    enableMultiProtocol: true,
    maxConcurrentRequests: 5,
    cacheEnabled: true
  },
  performance: {
    startTime: Date.now(),
    pageLoadTime: 0,
    apiResponseTimes: [],
    errorCount: 0
  }
};

// Services for the state machine
const appServices = {
  authenticateUser: async (context: AppContext, event: unknown) => {
    const { email, password } = event.credentials;
    
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) {
        throw new Error('Authentication failed');
      }

      const { user, token } = await response.json();
      
      // Store token in localStorage
      if (browser) {
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user', JSON.stringify(user));
      }

      return user;
    } catch (error) {
      throw error;
    }
  },

  loadUserSession: async () => {
    if (!browser) return null;

    try {
      const token = localStorage.getItem('auth_token');
      const userData = localStorage.getItem('user');

      if (!token || !userData) return null;

      // Verify token is still valid
      const response = await fetch('/api/auth/verify', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
        return null;
      }

      return JSON.parse(userData);
    } catch (error) {
      console.error('Session load error:', error);
      return null;
    }
  },

  persistSettings: async (context: AppContext) => {
    if (browser) {
      localStorage.setItem('app_settings', JSON.stringify(context.settings));
      localStorage.setItem('user_preferences', JSON.stringify(context.user?.preferences));
    }
  }
};

// Main application state machine
export const appStateMachine = createMachine({
  id: 'appState',
  initial: 'initializing',
  context: defaultContext,
  states: {
    initializing: {
      invoke: {
        id: 'loadSession',
        src: 'loadUserSession',
        onDone: [
          {
            target: 'authenticated',
            guard: (_, event) => event.data !== null,
            actions: assign({
              user: (_, event) => event.data,
              isAuthenticated: true
            })
          },
          {
            target: 'unauthenticated'
          }
        ],
        onError: {
          target: 'unauthenticated'
        }
      }
    },

    unauthenticated: {
      on: {
        LOGIN: {
          target: 'authenticating'
        },
        NAVIGATE: {
          actions: assign({
            currentRoute: (_, event) => event.route
          })
        }
      }
    },

    authenticating: {
      invoke: {
        id: 'authenticate',
        src: 'authenticateUser',
        onDone: {
          target: 'authenticated',
          actions: [
            assign({
              user: (_, event) => event.data,
              isAuthenticated: true
            }),
            'addSuccessNotification'
          ]
        },
        onError: {
          target: 'unauthenticated',
          actions: [
            assign({
              user: null,
              isAuthenticated: false
            }),
            'addErrorNotification'
          ]
        }
      }
    },

    authenticated: {
      entry: ['loadUserSettings'],
      on: {
        LOGOUT: {
          target: 'unauthenticated',
          actions: [
            assign({
              user: null,
              isAuthenticated: false
            }),
            'clearSession'
          ]
        },
        NAVIGATE: {
          actions: assign({
            currentRoute: (_, event) => event.route
          })
        },
        UPDATE_USER_PREFERENCES: {
          actions: [
            assign({
              user: (context, event) => ({
                ...context.user!,
                preferences: {
                  ...context.user!.preferences,
                  ...event.preferences
                }
              })
            }),
            'persistUserPreferences'
          ]
        }
      }
    }
  },

  on: {
    ADD_NOTIFICATION: {
      actions: assign({
        notifications: (context, event) => [
          {
            id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: Date.now(),
            read: false,
            ...event.notification
          },
          ...context.notifications.slice(0, 19) // Keep last 20 notifications
        ]
      })
    },

    MARK_NOTIFICATION_READ: {
      actions: assign({
        notifications: (context, event) =>
          context.notifications.map(notif =>
            notif.id === event.id ? { ...notif, read: true } : notif
          )
      })
    },

    CLEAR_NOTIFICATIONS: {
      actions: assign({
        notifications: []
      })
    },

    UPDATE_SETTINGS: {
      actions: [
        assign({
          settings: (context, event) => ({
            ...context.settings,
            ...event.settings
          })
        }),
        'persistSettings'
      ]
    },

    RECORD_PERFORMANCE: {
      actions: assign({
        performance: (context, event) => {
          const { metric, value } = event;
          
          switch (metric) {
            case 'pageLoad':
              return { ...context.performance, pageLoadTime: value };
            case 'apiResponse':
              return {
                ...context.performance,
                apiResponseTimes: [...context.performance.apiResponseTimes, value].slice(-50)
              };
            case 'error':
              return { ...context.performance, errorCount: context.performance.errorCount + 1 };
            default:
              return context.performance;
          }
        }
      })
    },

    RESET_SESSION: {
      target: 'initializing',
      actions: assign(defaultContext)
    }
  }
}, {
  services: appServices,
  actions: {
    addSuccessNotification: assign({
      notifications: (context) => [{
        id: `notif_${Date.now()}`,
        type: 'success' as const,
        title: 'Welcome!',
        message: `Logged in as ${context.user?.email}`,
        timestamp: Date.now(),
        read: false
      }, ...context.notifications]
    }),

    addErrorNotification: assign({
      notifications: (context, event) => [{
        id: `notif_${Date.now()}`,
        type: 'error' as const,
        title: 'Authentication Failed',
        message: 'Invalid credentials. Please try again.',
        timestamp: Date.now(),
        read: false
      }, ...context.notifications]
    }),

    clearSession: () => {
      if (browser) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
      }
    },

    loadUserSettings: assign({
      settings: (context) => {
        if (browser) {
          const saved = localStorage.getItem('app_settings');
          if (saved) {
            try {
              return { ...context.settings, ...JSON.parse(saved) };
            } catch (e) {
              console.warn('Failed to load settings:', e);
            }
          }
        }
        return context.settings;
      }
    }),

    persistUserPreferences: (context) => {
      if (browser && context.user) {
        localStorage.setItem('user_preferences', JSON.stringify(context.user.preferences));
      }
    },

    persistSettings: (context) => {
      if (browser) {
        localStorage.setItem('app_settings', JSON.stringify(context.settings));
      }
    }
  }
});

// Type for the app service
export type AppService = InterpreterFrom<typeof appStateMachine>;

// Svelte store for app state
export const appState = writable<AppContext>(defaultContext);
export const appActor = writable<AppService | null>(null);

// Helper functions for common operations
export const appActions = {
  login: (credentials: { email: string; password: string }) => ({
    type: 'LOGIN' as const,
    credentials
  }),

  logout: () => ({
    type: 'LOGOUT' as const
  }),

  navigate: (route: string) => ({
    type: 'NAVIGATE' as const,
    route
  }),

  notify: (notification: Omit<AppContext['notifications'][0], 'id' | 'timestamp'>) => ({
    type: 'ADD_NOTIFICATION' as const,
    notification
  }),

  updateSettings: (settings: Partial<AppContext['settings']>) => ({
    type: 'UPDATE_SETTINGS' as const,
    settings
  }),

  recordPerformance: (metric: string, value: number) => ({
    type: 'RECORD_PERFORMANCE' as const,
    metric,
    value
  })
};

// Selectors for derived state
export const appSelectors = {
  isAuthenticated: (context: AppContext) => context.isAuthenticated,
  currentUser: (context: AppContext) => context.user,
  unreadNotifications: (context: AppContext) => 
    context.notifications.filter(n => !n.read),
  averageApiResponseTime: (context: AppContext) => {
    const times = context.performance.apiResponseTimes;
    return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  },
  isGPUEnabled: (context: AppContext) => context.settings.enableGPUAcceleration,
  theme: (context: AppContext) => context.user?.preferences.theme || 'yorha'
};