import { writable } from 'svelte/store';

export interface User {
  id: number;
  email: string;
  firstName?: string;
  lastName?: string;
  name?: string;
  role: string;
  department?: string;
  jurisdiction?: string;
  isActive: boolean;
}

export interface SessionStore {
  user: User | null;
  authenticated: boolean;
  loading: boolean;
  error: string | null;
}

const initialState: SessionStore = {
  user: null,
  authenticated: false,
  loading: false,
  error: null
};

function createSessionStore() {
  const { subscribe, set, update } = writable<SessionStore>(initialState);

  return {
    subscribe,
    
    // Login
    login: async (email: string, password: string) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.message || 'Login failed');
        }

        const data = await response.json();
        const user = data.data.user;

        update(state => ({
          ...state,
          user,
          authenticated: true,
          loading: false
        }));

        return user;

      } catch (error: any) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message,
          authenticated: false,
          user: null
        }));
        throw error;
      }
    },

    // Logout
    logout: async (): Promise<any> => {
      update(state => ({ ...state, loading: true }));

      try {
        await fetch('/api/auth/logout', { method: 'POST' });
      } catch (error: any) {
        console.warn('Logout request failed:', error);
      }

      set(initialState);
    },

    // Check authentication status
    checkAuth: async (): Promise<any> => {
      update(state => ({ ...state, loading: true }));

      try {
        const response = await fetch('/api/auth/me');
        
        if (response.ok) {
          const data = await response.json();
          const user = data.data.user;

          update(state => ({
            ...state,
            user,
            authenticated: true,
            loading: false
          }));

          return user;
        } else {
          set(initialState);
          return null;
        }

      } catch (error: any) {
        set(initialState);
        return null;
      }
    },

    // Update user profile
    updateProfile: async (updates: Partial<User>) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/user/profile', {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates)
        });

        if (!response.ok) throw new Error('Failed to update profile');

        const data = await response.json();
        const updatedUser = data.data.user;

        update(state => ({
          ...state,
          user: updatedUser,
          loading: false
        }));

        return updatedUser;

      } catch (error: any) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Set user
    setUser: (user: User | null) => {
      update(state => ({
        ...state,
        user,
        authenticated: !!user
      }));
    },

    // Clear error
    clearError: () => {
      update(state => ({ ...state, error: null }));
    },

    // Reset store
    reset: () => {
      set(initialState);
    }
  };
}

export const sessionStore = createSessionStore();