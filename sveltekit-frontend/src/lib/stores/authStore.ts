import crypto from "crypto";

// Global authentication store with AI assistant integration
import { writable, derived, get } from "svelte/store";
import { browser } from "$app/environment";
import { goto } from "$app/navigation";

export interface AuthUser {
  id: string;
  email: string;
  name?: string;
  firstName?: string;
  lastName?: string;
  role: "admin" | "prosecutor" | "detective" | "user";
  avatarUrl?: string;
  isActive: boolean;
  createdAt: Date;
}

export interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  sessionId?: string;
  lastActivity?: Date;
}

// Core authentication state
const createAuthStore = () => {
  const { subscribe, set, update } = writable<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
  });

  return {
    subscribe,

    // Initialize authentication state from server
    async init() {
      if (!browser) return;

      try {
        const response = await fetch("/api/auth/me", {
          credentials: "include",
        });

        if (response.ok) {
          const userData = await response.json();
          set({
            user: userData.user,
            isAuthenticated: true,
            isLoading: false,
            sessionId: userData.sessionId,
            lastActivity: new Date(),
          });
        } else {
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      } catch (error: any) {
        console.error("Failed to initialize auth:", error);
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        });
      }
    },

    // Login user
    async login(email: string, password: string) {
      update((state: any) => ({ ...state, isLoading: true }));

      try {
        const response = await fetch("/api/auth/login", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ email, password }),
          credentials: "include",
        });

        if (response.ok) {
          const userData = await response.json();
          set({
            user: userData.user,
            isAuthenticated: true,
            isLoading: false,
            sessionId: userData.sessionId,
            lastActivity: new Date(),
          });

          // Initialize AI assistant for user
          aiAssistantStore.initializeForUser(userData.user);

          return { success: true };
        } else {
          const error = await response.json();
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          });
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        });
        return { success: false, error: "Network error" };
      }
    },

    // Register new user
    async register(userData: {
      email: string;
      password: string;
      firstName?: string;
      lastName?: string;
      role?: string;
    }) {
      update((state: any) => ({ ...state, isLoading: true }));

      try {
        const response = await fetch("/api/auth/register", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(userData),
          credentials: "include",
        });

        if (response.ok) {
          const result = await response.json();
          set({
            user: result.user,
            isAuthenticated: true,
            isLoading: false,
            sessionId: result.sessionId,
            lastActivity: new Date(),
          });

          // Initialize AI assistant for new user
          aiAssistantStore.initializeForUser(result.user);

          return { success: true };
        } else {
          const error = await response.json();
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          });
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        });
        return { success: false, error: "Network error" };
      }
    },

    // Logout user
    async logout() {
      try {
        await fetch("/api/auth/logout", {
          method: "POST",
          credentials: "include",
        });
      } catch (error: any) {
        console.error("Logout error:", error);
      }

      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
      });

      // Clear AI assistant state
      aiAssistantStore.clear();

      // Redirect to login
      goto("/login");
    },

    // Update user profile
    async updateProfile(updates: Partial<AuthUser>) {
      const currentState = get({ subscribe });
      if (!currentState.user)
        return { success: false, error: "Not authenticated" };

      try {
        const response = await fetch("/api/auth/profile", {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updates),
          credentials: "include",
        });

        if (response.ok) {
          const updatedUser = await response.json();
          update((state: any) => ({
            ...state,
            user: { ...state.user!, ...updatedUser },
            lastActivity: new Date(),
          }));

          return { success: true };
        } else {
          const error = await response.json();
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        return { success: false, error: "Network error" };
      }
    },

    // Update last activity
    updateActivity() {
      update((state: any) => ({
        ...state,
        lastActivity: new Date(),
      }));
    },
  };
};

export const authStore = createAuthStore();
// Derived stores for common checks (repaired syntax)
export const isAuthenticated = derived(authStore, ($auth: any) => $auth.isAuthenticated);
export const currentUser = derived(authStore, ($auth: any) => $auth.user);
export const userRole = derived(authStore, ($auth: any) => $auth.user?.role);
export const isAdmin = derived(authStore, ($auth: any) => $auth.user?.role === "admin");
export const isProsecutor = derived(authStore, ($auth: any) => $auth.user?.role === "prosecutor");
export const isDetective = derived(authStore, ($auth: any) => $auth.user?.role === "detective");

// AI Assistant integration store
export interface AIAssistantState {
  isEnabled: boolean;
  userId?: string;
  preferences: {
    autoSuggest: boolean;
    contextAwareness: boolean;
    legalSpecialization: boolean;
    confidenceThreshold: number;
  };
  currentContext?: {
    caseId?: string;
    evidenceId?: string;
    reportId?: string;
  };
  conversationHistory: Array<{
    id: string;
    timestamp: Date;
    prompt: string;
    response: string;
    confidence: number;
    context?: unknown;
  }>;
}

const createAIAssistantStore = () => {
  const { subscribe, set, update } = writable<AIAssistantState>({
    isEnabled: false,
    preferences: {
      autoSuggest: true,
      contextAwareness: true,
      legalSpecialization: true,
      confidenceThreshold: 0.7,
    },
    conversationHistory: [],
  });

  return {
    subscribe,

    // Initialize AI assistant for user
    initializeForUser(user: AuthUser) {
      update((state: any) => ({
        ...state,
        isEnabled: true,
        userId: user.id,
        preferences: {
          ...state.preferences,
          legalSpecialization:
            user.role === "prosecutor" || user.role === "detective",
        },
      }));

      // Load user's AI preferences from server
      this.loadPreferences();
    },

    // Load AI preferences from server
    async loadPreferences() {
      try {
        const response = await fetch("/api/ai/preferences", {
          credentials: "include",
        });

        if (response.ok) {
          const preferences = await response.json();
          update((state: any) => ({
            ...state,
            preferences: { ...state.preferences, ...preferences },
          }));
        }
      } catch (error: any) {
        console.error("Failed to load AI preferences:", error);
      }
    },

    // Update AI preferences
    async updatePreferences(updates: Partial<AIAssistantState["preferences"]>) {
      try {
        const response = await fetch("/api/ai/preferences", {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updates),
          credentials: "include",
        });

        if (response.ok) {
          update((state: any) => ({
            ...state,
            preferences: { ...state.preferences, ...updates },
          }));
          return { success: true };
        } else {
          return { success: false, error: "Failed to update preferences" };
        }
      } catch (error: any) {
        return { success: false, error: "Network error" };
      }
    },

    // Set current context for AI assistance
    setContext(context: AIAssistantState["currentContext"]) {
      update((state: any) => ({
        ...state,
        currentContext: context,
      }));
    },

    // Add conversation to history
    addConversation(
      conversation: Omit<
        AIAssistantState["conversationHistory"][0],
        "id" | "timestamp"
      >
    ) {
      const newConversation = {
        ...conversation,
        id: crypto.randomUUID(),
        timestamp: new Date(),
      };

      update((state: any) => ({
        ...state,
        conversationHistory: [
          newConversation,
          ...state.conversationHistory.slice(0, 49),
        ], // Keep last 50
      }));
    },

    // Clear AI assistant state
    clear() {
      set({
        isEnabled: false,
        preferences: {
          autoSuggest: true,
          contextAwareness: true,
          legalSpecialization: true,
          confidenceThreshold: 0.7,
        },
        conversationHistory: [],
      });
    },
  };
};

export const aiAssistantStore = createAIAssistantStore();
// Derived stores for AI assistant (repaired syntax)
export const aiEnabled = derived(aiAssistantStore, ($ai: any) => $ai.isEnabled);
export const aiPreferences = derived(aiAssistantStore, ($ai: any) => $ai.preferences);
export const aiContext = derived(aiAssistantStore, ($ai: any) => $ai.currentContext);
export const recentConversations = derived(aiAssistantStore, ($ai: any) => $ai.conversationHistory.slice(0, 10));

// Note: Auth initialization is now handled server-side via hooks.server.ts
// Client-side initialization only when explicitly needed to prevent redirect loops
