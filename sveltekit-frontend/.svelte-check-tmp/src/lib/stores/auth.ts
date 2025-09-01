
import type { User } from "$lib/types/user";
import { writable, type Writable } from "svelte/store";
import { setContext, getContext } from "svelte";

export interface AuthUser {
  id: string;
  email: string;
  name?: string;
  role: string;
  avatarUrl?: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: AuthUser | null;
  isLoading: boolean;
}

const createAuthStore = () => {
  const { subscribe, set, update } = writable<AuthState>({
    isAuthenticated: false,
    user: null,
    isLoading: true,
  });

  return {
    subscribe,
    login: async (email: string, password: string) => {
      update((state) => ({ ...state, isLoading: true }));

      try {
        const response = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        if (response.ok) {
          const { user } = await response.json();
          set({ isAuthenticated: true, user, isLoading: false });
          return { success: true };
        } else {
          const { error } = await response.json();
          set({ isAuthenticated: false, user: null, isLoading: false });
          return { success: false, error };
        }
      } catch (error: any) {
        set({ isAuthenticated: false, user: null, isLoading: false });
        return { success: false, error: "Network error" };
      }
    },

    logout: async () => {
      try {
        await fetch("/api/auth/logout", { method: "POST" });
      } catch (error: any) {
        console.error("Logout error:", error);
      }
      set({ isAuthenticated: false, user: null, isLoading: false });
    },

    checkAuth: async () => {
      update((state) => ({ ...state, isLoading: true }));

      try {
        const response = await fetch("/api/auth/me");
        if (response.ok) {
          const { user } = await response.json();
          set({ isAuthenticated: true, user, isLoading: false });
        } else {
          set({ isAuthenticated: false, user: null, isLoading: false });
        }
      } catch (error: any) {
        set({ isAuthenticated: false, user: null, isLoading: false });
      }
    },

    updateUser: (userData: Partial<AuthUser>) => {
      update((state) => ({
        ...state,
        user: state.user ? { ...state.user, ...userData } : null,
      }));
    },
  };
};

export type AuthStore = ReturnType<typeof createAuthStore>;

// Context key for the auth store
const AUTH_CONTEXT_KEY = Symbol("auth");

// Set the auth context (call this in your root layout)
export const setAuthContext = (): AuthStore => {
  const authStore = createAuthStore();
  setContext(AUTH_CONTEXT_KEY, authStore);
  return authStore;
};

// Get the auth context (call this in components that need auth)
export const getAuthContext = (): AuthStore => {
  const authStore = getContext<AuthStore>(AUTH_CONTEXT_KEY);
  if (!authStore) {
    throw new Error(
      "Auth context not found. Make sure to call setAuthContext in your root layout.",
    );
  }
  return authStore;
};

// Utility to check if user has specific role
export const hasRole = (user: AuthUser | null, role: string): boolean => {
  return user?.role === role;
};

// Utility to check if user has any of the specified roles
export const hasAnyRole = (user: AuthUser | null, roles: string[]): boolean => {
  return user ? roles.includes(user.role) : false;
};

// Create and export default auth store
const authStore = createAuthStore();
export default authStore;
