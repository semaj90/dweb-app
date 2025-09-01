
// Global case management store with real data integration
import { writable, derived, get } from "svelte/store";
import { authStore } from "./authStore";
import type { Case, Evidence, Report } from '$lib/server/db/schema';

// Extended case type with relations
export interface CaseWithRelations extends Case {
  evidence?: Evidence[];
  reports?: Report[];
}

// Alias for backward compatibility
export type CaseData = CaseWithRelations;

export interface CaseState {
  cases: CaseWithRelations[];
  activeCaseId: string | null;
  activeCase: CaseWithRelations | null;
  isLoading: boolean;
  error: string | null;
  filters: {
    status?: string;
    caseType?: string;
    priority?: string;
    search?: string;
  };
  pagination: {
    page: number;
    limit: number;
    total: number;
  };
}

const createCaseStore = () => {
  const { subscribe, set, update } = writable<CaseState>({
    cases: [],
    activeCaseId: null,
    activeCase: null,
    isLoading: false,
    error: null,
    filters: {},
    pagination: {
      page: 1,
      limit: 20,
      total: 0,
    },
  });

  return {
    subscribe,

    // Load cases from database
    async loadCases(filters?: Partial<CaseState["filters"]>) {
      update((state) => ({ ...state, isLoading: true, error: null }));

      try {
        const params = new URLSearchParams();
        if (filters?.status) params.append("status", filters.status);
        if (filters?.caseType) params.append("type", filters.caseType);
        if (filters?.priority) params.append("priority", filters.priority);
        if (filters?.search) params.append("search", filters.search);

        const currentState = get({ subscribe });
        params.append("page", currentState.pagination.page.toString());
        params.append("limit", currentState.pagination.limit.toString());

        const response = await fetch(`/api/cases?${params}`, {
          credentials: "include",
        });

        if (response.ok) {
          const data = await response.json();
          update((state) => ({
            ...state,
            cases: data.cases,
            pagination: {
              ...state.pagination,
              total: data.total,
            },
            filters: { ...state.filters, ...filters },
            isLoading: false,
          }));
        } else {
          const error = await response.json();
          update((state) => ({
            ...state,
            error: error.message || "Failed to load cases",
            isLoading: false,
          }));
        }
      } catch (error: any) {
        update((state) => ({
          ...state,
          error: "Network error while loading cases",
          isLoading: false,
        }));
      }
    },

    // Load specific case with relations
    async loadCase(caseId: string) {
      update((state) => ({ ...state, isLoading: true, error: null }));

      try {
        const response = await fetch(`/api/cases/${caseId}`, {
          credentials: "include",
        });

        if (response.ok) {
          const caseData = await response.json();
          update((state) => ({
            ...state,
            activeCase: caseData,
            activeCaseId: caseId,
            isLoading: false,
          }));
          return { success: true, case: caseData };
        } else {
          const error = await response.json();
          update((state) => ({
            ...state,
            error: error.message || "Failed to load case",
            isLoading: false,
          }));
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        update((state) => ({
          ...state,
          error: "Network error while loading case",
          isLoading: false,
        }));
        return { success: false, error: "Network error" };
      }
    },

    // Create new case
    async createCase(caseData: {
      title: string;
      description?: string;
      caseType?: string;
      priority?: string;
    }) {
      update((state) => ({ ...state, isLoading: true, error: null }));

      try {
        const response = await fetch("/api/cases", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(caseData),
          credentials: "include",
        });

        if (response.ok) {
          const newCase = await response.json();
          update((state) => ({
            ...state,
            cases: [newCase, ...state.cases],
            pagination: {
              ...state.pagination,
              total: state.pagination.total + 1,
            },
            isLoading: false,
          }));
          return { success: true, case: newCase };
        } else {
          const error = await response.json();
          update((state) => ({
            ...state,
            error: error.message || "Failed to create case",
            isLoading: false,
          }));
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        update((state) => ({
          ...state,
          error: "Network error while creating case",
          isLoading: false,
        }));
        return { success: false, error: "Network error" };
      }
    },

    // Update case
    async updateCase(caseId: string, updates: Partial<Case>) {
      update((state) => ({ ...state, isLoading: true, error: null }));

      try {
        const response = await fetch(`/api/cases/${caseId}`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updates),
          credentials: "include",
        });

        if (response.ok) {
          const updatedCase = await response.json();
          update((state) => ({
            ...state,
            cases: state.cases.map((c) =>
              c.id === caseId ? { ...c, ...updatedCase } : c
            ),
            activeCase:
              state.activeCase?.id === caseId
                ? { ...state.activeCase, ...updatedCase }
                : state.activeCase,
            isLoading: false,
          }));
          return { success: true, case: updatedCase };
        } else {
          const error = await response.json();
          update((state) => ({
            ...state,
            error: error.message || "Failed to update case",
            isLoading: false,
          }));
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        update((state) => ({
          ...state,
          error: "Network error while updating case",
          isLoading: false,
        }));
        return { success: false, error: "Network error" };
      }
    },

    // Delete case
    async deleteCase(caseId: string) {
      update((state) => ({ ...state, isLoading: true, error: null }));

      try {
        const response = await fetch(`/api/cases/${caseId}`, {
          method: "DELETE",
          credentials: "include",
        });

        if (response.ok) {
          update((state) => ({
            ...state,
            cases: state.cases.filter((c) => c.id !== caseId),
            activeCase:
              state.activeCase?.id === caseId ? null : state.activeCase,
            activeCaseId:
              state.activeCaseId === caseId ? null : state.activeCaseId,
            pagination: {
              ...state.pagination,
              total: Math.max(0, state.pagination.total - 1),
            },
            isLoading: false,
          }));
          return { success: true };
        } else {
          const error = await response.json();
          update((state) => ({
            ...state,
            error: error.message || "Failed to delete case",
            isLoading: false,
          }));
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        update((state) => ({
          ...state,
          error: "Network error while deleting case",
          isLoading: false,
        }));
        return { success: false, error: "Network error" };
      }
    },

    // Generate AI-powered report for case
    async generateReport(
      caseId: string,
      reportType: string,
      customPrompt?: string
    ) {
      update((state) => ({ ...state, isLoading: true, error: null }));

      try {
        const response = await fetch(`/api/cases/${caseId}/generate-report`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            reportType,
            customPrompt,
            useContext7: true, // Enable Context7 MCP for better analysis
          }),
          credentials: "include",
        });

        if (response.ok) {
          const report = await response.json();

          // Update case with new report
          update((state) => ({
            ...state,
            activeCase: state.activeCase
              ? {
                  ...state.activeCase,
                  reports: [...(state.activeCase.reports || []), report],
                }
              : state.activeCase,
            isLoading: false,
          }));

          return { success: true, report };
        } else {
          const error = await response.json();
          update((state) => ({
            ...state,
            error: error.message || "Failed to generate report",
            isLoading: false,
          }));
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        update((state) => ({
          ...state,
          error: "Network error while generating report",
          isLoading: false,
        }));
        return { success: false, error: "Network error" };
      }
    },

    // Set active case
    setActiveCase(caseId: string | null) {
      update((state) => {
        const activeCase = caseId
          ? state.cases.find((c) => c.id === caseId) || null
          : null;
        return {
          ...state,
          activeCaseId: caseId,
          activeCase,
        };
      });
    },

    // Update filters
    setFilters(filters: Partial<CaseState["filters"]>) {
      update((state) => ({
        ...state,
        filters: { ...state.filters, ...filters },
        pagination: { ...state.pagination, page: 1 }, // Reset to first page
      }));

      // Reload cases with new filters
      this.loadCases(get({ subscribe }).filters);
    },

    // Clear filters
    clearFilters() {
      update((state) => ({
        ...state,
        filters: {},
        pagination: { ...state.pagination, page: 1 },
      }));

      this.loadCases();
    },

    // Pagination
    setPage(page: number) {
      update((state) => ({
        ...state,
        pagination: { ...state.pagination, page },
      }));

      this.loadCases(get({ subscribe }).filters);
    },

    // Clear error
    clearError() {
      update((state) => ({ ...state, error: null }));
    },

    // AI-powered case analysis using Context7 MCP
    async analyzeCase(
      caseId: string,
      analysisType: "evidence" | "legal" | "timeline" | "poi"
    ) {
      try {
        const response = await fetch(`/api/cases/${caseId}/analyze`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            analysisType,
            useContext7: true,
            includeMCPMemory: true,
          }),
          credentials: "include",
        });

        if (response.ok) {
          const analysis = await response.json();
          return { success: true, analysis };
        } else {
          const error = await response.json();
          return { success: false, error: error.message };
        }
      } catch (error: any) {
        return { success: false, error: "Network error during analysis" };
      }
    },
  };
};

export const caseStore = createCaseStore();
;
// Derived stores
export const activeCaseId = derived(caseStore, ($cases) => $cases.activeCaseId);
export const activeCase = derived(caseStore, ($cases) => $cases.activeCase);
export const filteredCases = derived(caseStore, ($cases) => $cases.cases);
export const casesLoading = derived(caseStore, ($cases) => $cases.isLoading);
export const casesError = derived(caseStore, ($cases) => $cases.error);
export const casesPagination = derived(caseStore, ($cases) => $cases.pagination);

// Initialize cases when authenticated
if (browser) {
  authStore.subscribe((auth) => {
    if (auth.isAuthenticated && !auth.isLoading) {
      caseStore.loadCases();
    }
  });
}
