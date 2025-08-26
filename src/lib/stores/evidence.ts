import { writable } from 'svelte/store';
import type { Evidence } from '$lib/types';
import { cache } from '$lib/client/cache';

interface EvidenceStore {
  items: Evidence[];
  loading: boolean;
  error: string | null;
  selectedIds: number[];
  searchQuery: string;
  filteredItems: Evidence[];
}

const initialState: EvidenceStore = {
  items: [],
  loading: false,
  error: null,
  selectedIds: [],
  searchQuery: '',
  filteredItems: []
};

function createEvidenceStore() {
  const { subscribe, set, update } = writable<EvidenceStore>(initialState);

  return {
    subscribe,
    
    // Load evidence for a case
    load: async (caseId: number) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        // Try cache first
        const cached = cache.evidence.get(caseId);
        if (cached.length > 0) {
          update(state => ({
            ...state,
            items: cached,
            filteredItems: cached,
            loading: false
          }));
        }

        // Fetch from API
        const response = await fetch(`/api/evidence?caseId=${caseId}`);
        if (!response.ok) throw new Error('Failed to load evidence');
        
        const data = await response.json();
        const evidence = data.data || [];

        // Update cache
        cache.evidence.set(evidence);

        update(state => ({
          ...state,
          items: evidence,
          filteredItems: evidence,
          loading: false
        }));

      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
      }
    },

    // Add new evidence
    add: (item: Evidence) => {
      cache.evidence.add(item);
      update(state => {
        const newItems = [item, ...state.items];
        return {
          ...state,
          items: newItems,
          filteredItems: state.searchQuery 
            ? newItems.filter(i => i.filename.toLowerCase().includes(state.searchQuery.toLowerCase()))
            : newItems
        };
      });
    },

    // Update evidence
    upsert: (item: Evidence) => {
      cache.evidence.update(item.id, item);
      update(state => {
        const idx = state.items.findIndex(x => x.id === item.id);
        const newItems = idx === -1 
          ? [item, ...state.items]
          : [...state.items.slice(0, idx), item, ...state.items.slice(idx + 1)];
        
        return {
          ...state,
          items: newItems,
          filteredItems: state.searchQuery
            ? newItems.filter(i => i.filename.toLowerCase().includes(state.searchQuery.toLowerCase()))
            : newItems
        };
      });
    },

    // Remove evidence
    remove: (id: number) => {
      cache.evidence.remove(id);
      update(state => {
        const newItems = state.items.filter(x => x.id !== id);
        return {
          ...state,
          items: newItems,
          filteredItems: newItems,
          selectedIds: state.selectedIds.filter(selectedId => selectedId !== id)
        };
      });
    },

    // Selection management
    select: (id: number) => {
      update(state => ({
        ...state,
        selectedIds: [...new Set([...state.selectedIds, id])]
      }));
    },

    deselect: (id: number) => {
      update(state => ({
        ...state,
        selectedIds: state.selectedIds.filter(selectedId => selectedId !== id)
      }));
    },

    selectAll: () => {
      update(state => ({
        ...state,
        selectedIds: state.filteredItems.map(item => item.id)
      }));
    },

    clearSelection: () => {
      update(state => ({
        ...state,
        selectedIds: []
      }));
    },

    // Search and filter
    search: (query: string) => {
      update(state => {
        const filtered = query
          ? cache.evidence.search(query)
          : state.items;
        
        return {
          ...state,
          searchQuery: query,
          filteredItems: filtered
        };
      });
    },

    // Filter by tags
    filterByTags: (tags: string[]) => {
      update(state => {
        const filtered = state.items.filter(item =>
          tags.some(tag => item.tags?.includes(tag))
        );
        return {
          ...state,
          filteredItems: filtered
        };
      });
    },

    // Filter by file type
    filterByType: (fileType: string) => {
      update(state => {
        const filtered = fileType === 'all'
          ? state.items
          : state.items.filter(item => item.fileType === fileType);
        return {
          ...state,
          filteredItems: filtered
        };
      });
    },

    // Clear all filters
    clearFilters: () => {
      update(state => ({
        ...state,
        searchQuery: '',
        filteredItems: state.items
      }));
    },

    // Upload evidence
    upload: async (caseId: number, file: File, metadata?: any) => {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('caseId', caseId.toString());
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }

        const response = await fetch('/api/evidence/upload', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        const newEvidence = data.data;

        // Add to store and cache
        cache.evidence.add(newEvidence);
        update(state => {
          const newItems = [newEvidence, ...state.items];
          return {
            ...state,
            items: newItems,
            filteredItems: state.searchQuery
              ? newItems.filter(i => i.filename.toLowerCase().includes(state.searchQuery.toLowerCase()))
              : newItems,
            loading: false
          };
        });

        return newEvidence;

      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error.message
        }));
        throw error;
      }
    },

    // Reset store
    reset: () => {
      set(initialState);
    }
  };
}

export const evidenceStore = createEvidenceStore();