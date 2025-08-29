import { writable } from 'svelte/store';
import type { Case } from '$lib/types';
import { cache } from '$lib/client/cache';

export interface CaseStore {
  items: Case[];
  currentCase: Case | null;
  loading: boolean;
  error: string | null;
  searchQuery: string;
  filteredItems: Case[];
}

const initialState: CaseStore = {
  items: [],
  currentCase: null,
  loading: false,
  error: null,
  searchQuery: '',
  filteredItems: []
};

function createCaseStore() {
  const { subscribe, set, update } = writable<CaseStore>(initialState);

  return {
    subscribe,
    
    // Load all cases for user
    loadAll: async (userId?: number) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        // Try cache first
        const cached = cache.cases.get(userId);
        if (cached.length > 0) {
          update(state => ({
            ...state,
            items: cached,
            filteredItems: cached,
            loading: false
          }));
        }

        // Fetch from API
        const response = await fetch('/api/cases');
        if (!response.ok) throw new Error('Failed to load cases');
        
        const data = await response.json();
        const cases = data.data || [];

        // Update cache
        cache.cases.set(cases);

        update(state => ({
          ...state,
          items: cases,
          filteredItems: cases,
          loading: false
        }));

      } catch (error: any) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
      }
    },

    // Load specific case
    load: async (caseId: number) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const response = await fetch(`/api/cases/${caseId}`);
        if (!response.ok) throw new Error('Failed to load case');
        
        const data = await response.json();
        const caseData = data.data;

        // Update cache
        cache.cases.update(caseId, caseData);

        update(state => ({
          ...state,
          currentCase: caseData,
          loading: false
        }));

        return caseData;

      } catch (error: any) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Create new case
    create: async (caseData: Partial<Case>) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch('/api/cases', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(caseData)
        });

        if (!response.ok) throw new Error('Failed to create case');

        const data = await response.json();
        const newCase = data.data;

        // Add to cache and store
        cache.cases.add(newCase);
        update(state => {
          const newItems = [newCase, ...state.items];
          return {
            ...state,
            items: newItems,
            filteredItems: state.searchQuery
              ? newItems.filter(c => c.title.toLowerCase().includes(state.searchQuery.toLowerCase()))
              : newItems,
            currentCase: newCase,
            loading: false
          };
        });

        return newCase;

      } catch (error: any) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Update case
    update: async (caseId: number, updates: Partial<Case>) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`/api/cases/${caseId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates)
        });

        if (!response.ok) throw new Error('Failed to update case');

        const data = await response.json();
        const updatedCase = data.data;

        // Update cache
        cache.cases.update(caseId, updatedCase);

        update(state => {
          const idx = state.items.findIndex(c => c.id === caseId);
          const newItems = idx === -1 
            ? state.items
            : [...state.items.slice(0, idx), updatedCase, ...state.items.slice(idx + 1)];
          
          return {
            ...state,
            items: newItems,
            filteredItems: state.searchQuery
              ? newItems.filter(c => c.title.toLowerCase().includes(state.searchQuery.toLowerCase()))
              : newItems,
            currentCase: state.currentCase?.id === caseId ? updatedCase : state.currentCase,
            loading: false
          };
        });

        return updatedCase;

      } catch (error: any) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Set current case
    setCurrent: (caseData: Case | null) => {
      update(state => ({
        ...state,
        currentCase: caseData
      }));
    },

    // Search cases
    search: (query: string) => {
      update(state => {
        const filtered = query
          ? cache.cases.search(query)
          : state.items;
        
        return {
          ...state,
          searchQuery: query,
          filteredItems: filtered
        };
      });
    },

    // Filter by status
    filterByStatus: (status: string) => {
      update(state => {
        const filtered = status === 'all'
          ? state.items
          : state.items.filter(c => c.status === status);
        return {
          ...state,
          filteredItems: filtered
        };
      });
    },

    // Clear filters
    clearFilters: () => {
      update(state => ({
        ...state,
        searchQuery: '',
        filteredItems: state.items
      }));
    },

    // Get case statistics
    getStats: () => {
      let stats = { total: 0, open: 0, closed: 0, pending: 0 };
      
      update(state => {
        stats = {
          total: state.items.length,
          open: state.items.filter(c => c.status === 'open').length,
          closed: state.items.filter(c => c.status === 'closed').length,
          pending: state.items.filter(c => c.status === 'pending').length
        };
        return state;
      });

      return stats;
    },

    // Reset store
    reset: () => {
      set(initialState);
    }
  };
}

export const caseStore = createCaseStore();