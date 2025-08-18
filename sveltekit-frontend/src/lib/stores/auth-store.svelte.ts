/**
 * Global Authentication Store - SvelteKit 2 + Svelte 5 Best Practices
 * Uses modern reactive patterns with $state runes and XState integration
 */

import { type User } from '$lib/server/auth';
import { createMachine, assign, fromPromise  } from "xstate";
import { browser } from "$app/environment";
import { goto } from "$app/navigation";
import { page } from "$app/stores";

// Authentication context
interface AuthContext {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  redirectPath: string | null;
}

// Authentication events
type AuthEvents =
  | { type: 'LOGIN'; email: string; password: string }
  | { type: 'REGISTER'; userData: RegisterData }
  | { type: 'LOGOUT' }
  | { type: 'CHECK_AUTH' }
  | { type: 'CLEAR_ERROR' }
  | { type: 'SET_REDIRECT'; path: string }
  | { type: 'AUTHENTICATED'; user: User }
  | { type: 'UNAUTHENTICATED' };

interface RegisterData {
  email: string;
  password: string;
  firstName?: string;
  lastName?: string;
  role?: string;
}

// API services
const loginService = fromPromise(
  async ({ input }: { input: { email: string; password: string } }) => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Login failed');
    }

    return await response.json();
  }
);

const registerService = fromPromise(async ({ input }: { input: RegisterData }) => {
  const response = await fetch('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Registration failed');
  }

  return await response.json();
});

const logoutService = fromPromise(async () => {
  const response = await fetch('/api/auth/logout', {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error('Logout failed');
  }

  return true;
});

const checkAuthService = fromPromise(async () => {
  const response = await fetch('/api/auth/me');

  if (!response.ok) {
    throw new Error('Not authenticated');
  }

  return await response.json();
});

// Authentication state machine
const authMachine = createMachine({
  id: 'auth',
  types: {
    context: {} as AuthContext,
    events: {} as AuthEvents,
  },
  context: {
    user: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
    redirectPath: null,
  },
  initial: 'checking',
  states: {
    checking: {
      entry: assign({ isLoading: true }),
      invoke: {
        src: checkAuthService,
        onDone: {
          target: 'authenticated',
          actions: [
            assign({
              user: ({ event }) => event.output.user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            }),
          ],
        },
        onError: {
          target: 'unauthenticated',
          actions: assign({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
          }),
        },
      },
    },

    unauthenticated: {
      on: {
        LOGIN: {
          target: 'loggingIn',
        },
        REGISTER: {
          target: 'registering',
        },
        SET_REDIRECT: {
          actions: assign({
            redirectPath: ({ event }) => event.path,
          }),
        },
      },
    },

    loggingIn: {
      entry: assign({
        isLoading: true,
        error: null,
      }),
      invoke: {
        src: loginService,
        // fromPromise passes original event as event in invoke meta; route it via event for LOGIN
        input: ({ event }: any) => ({
          email: (event as any).email,
          password: (event as any).password,
        }),
        onDone: {
          target: 'authenticated',
          actions: [
            assign({
              user: ({ event }) => event.output.user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            }),
            // Handle redirect after successful login
            ({ context }) => {
              if (browser && context.redirectPath) {
                goto(context.redirectPath);
              } else if (browser) {
                goto('/dashboard');
              }
            },
          ],
        },
        onError: {
          target: 'unauthenticated',
          actions: assign({
            isLoading: false,
            error: ({ event }: any) => (event as any).error?.message || 'Login failed',
          }),
        },
      },
    },

    registering: {
      entry: assign({
        isLoading: true,
        error: null,
      }),
      invoke: {
        src: registerService,
        input: ({ event }: any) => (event as any).userData,
        onDone: {
          target: 'authenticated',
          actions: [
            assign({
              user: ({ event }) => event.output.user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            }),
            // Redirect to dashboard after registration
            () => {
              if (browser) {
                goto('/dashboard');
              }
            },
          ],
        },
        onError: {
          target: 'unauthenticated',
          actions: assign({
            isLoading: false,
            error: ({ event }: any) => (event as any).error?.message || 'Registration failed',
          }),
        },
      },
    },

    authenticated: {
      entry: assign({
        isAuthenticated: true,
        isLoading: false,
        redirectPath: null,
      }),
      on: {
        LOGOUT: {
          target: 'loggingOut',
        },
        UNAUTHENTICATED: {
          target: 'unauthenticated',
          actions: assign({
            user: null,
            isAuthenticated: false,
          }),
        },
      },
    },

    loggingOut: {
      entry: assign({ isLoading: true }),
      invoke: {
        src: logoutService,
        onDone: {
          target: 'unauthenticated',
          actions: [
            assign({
              user: null,
              isAuthenticated: false,
              isLoading: false,
              error: null,
            }),
            // Redirect to home after logout
            () => {
              if (browser) {
                goto('/');
              }
            },
          ],
        },
        onError: {
          target: 'authenticated',
          actions: assign({
            isLoading: false,
            error: 'Logout failed',
          }),
        },
      },
    },
  },

  on: {
    CLEAR_ERROR: {
      actions: assign({ error: null }),
    },
    CHECK_AUTH: {
      target: 'checking',
    },
  },
});

// Create global auth store using SvelteKit 2 patterns
class AuthStore {
  #machineState: any = $state({
    context: {
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      redirectPath: null,
    },
    value: 'checking',
  });
  #actor: any = $state();

  constructor() {
    if (browser) {
      // Initialize actor only in browser
      import('xstate').then(({ createActor }) => {
        this.#actor = createActor(authMachine);
        this.#actor.start();

        // Subscribe to authentication state changes
        this.#actor.subscribe((state: any) => {
          // Update reactive state
          this.#machineState = {
            context: state.context,
            value: state.value,
          };
        });

        // Check initial auth state
        this.#actor.send({ type: 'CHECK_AUTH' });
      });
    }
  }

  // Reactive getters using proper typing
  get isAuthenticated() {
    return this.#machineState?.context?.isAuthenticated ?? false;
  }

  get user() {
    return this.#machineState?.context?.user ?? null;
  }

  get isLoading() {
    return this.#machineState?.context?.isLoading ?? false;
  }

  get error() {
    return this.#machineState?.context?.error ?? null;
  }

  get currentState() {
    return this.#machineState?.value ?? 'checking';
  }

  // User role helpers
  get isAdmin() {
    return this.user?.role === 'admin';
  }

  get isProsecutor() {
    return this.user?.role === 'prosecutor';
  }

  get isDetective() {
    return this.user?.role === 'detective';
  }

  // Actions
  login = (email: string, password: string) => {
    this.#actor?.send({ type: 'LOGIN', email, password });
  };

  register = (userData: RegisterData) => {
    this.#actor?.send({ type: 'REGISTER', userData });
  };

  logout = () => {
    this.#actor?.send({ type: 'LOGOUT' });
  };

  clearError = () => {
    this.#actor?.send({ type: 'CLEAR_ERROR' });
  };

  setRedirect = (path: string) => {
    this.#actor?.send({ type: 'SET_REDIRECT', path });
  };

  checkAuth = () => {
    this.#actor?.send({ type: 'CHECK_AUTH' });
  };

  // Permission helpers
  hasPermission = (permission: string): boolean => {
    if (!this.user) return false;

    const rolePermissions = {
      admin: ['read', 'write', 'delete', 'manage_users', 'manage_cases', 'manage_evidence'],
      prosecutor: ['read', 'write', 'manage_cases', 'manage_evidence'],
      detective: ['read', 'write', 'manage_evidence'],
      user: ['read'],
    };

    return (
      rolePermissions[this.user.role as keyof typeof rolePermissions]?.includes(permission) ?? false
    );
  };

  canAccessCase = (caseId: string): boolean => {
    // Implement case-specific access control
    return this.hasPermission('read');
  };

  canEditCase = (caseId: string): boolean => {
    return this.hasPermission('write');
  };

  canDeleteCase = (caseId: string): boolean => {
    return this.hasPermission('delete');
  };
}

// Export singleton instance
export const authStore = new AuthStore();

// Auth guards for routes
export function requireAuth() {
  return {
    beforeNavigate: ({ to }: { to: any }) => {
      if (!authStore.isAuthenticated) {
        authStore.setRedirect(to.pathname);
        goto('/auth/login');
        return false;
      }
      return true;
    },
  };
}

export function requireRole(role: string) {
  return {
    beforeNavigate: ({ to }: { to: any }) => {
      if (!authStore.isAuthenticated) {
        authStore.setRedirect(to.pathname);
        goto('/auth/login');
        return false;
      }

      if (authStore.user?.role !== role && authStore.user?.role !== 'admin') {
        goto('/unauthorized');
        return false;
      }

      return true;
    },
  };
}

export function requirePermission(permission: string) {
  return {
    beforeNavigate: ({ to }: { to: any }) => {
      if (!authStore.isAuthenticated) {
        authStore.setRedirect(to.pathname);
        goto('/auth/login');
        return false;
      }

      if (!authStore.hasPermission(permission)) {
        goto('/unauthorized');
        return false;
      }

      return true;
    },
  };
}

// Utility for components
export function useAuth() {
  return {
    user: $derived(authStore.user),
    isAuthenticated: $derived(authStore.isAuthenticated),
    isLoading: $derived(authStore.isLoading),
    error: $derived(authStore.error),
    currentState: $derived(authStore.currentState),
    login: authStore.login,
    register: authStore.register,
    logout: authStore.logout,
    clearError: authStore.clearError,
    hasPermission: authStore.hasPermission,
    canAccessCase: authStore.canAccessCase,
    canEditCase: authStore.canEditCase,
    canDeleteCase: authStore.canDeleteCase,
  };
}


