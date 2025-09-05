// @ts-nocheck
import { writable, get } from "svelte/store";
import { selectedCase } from "./cases";

// 1. Evidence Interface
export interface Evidence {
  id: string;
  caseId: string;
  title: string;
  type: "document" | "image" | "video" | "note";
  content: string; // URL for files, text for notes
  x: number; // Position on the canvas
  y: number;
  embedding?: number[];
}

// Store State Interface
interface EvidenceStoreState {
  evidence: Evidence[];
  isLoading: boolean;
  error: string | null;
}

// 2. Evidence Store
const createEvidenceStore = () => {
  const { subscribe, set, update } = writable<EvidenceStoreState>({
    evidence: [],
    isLoading: false,
    error: null,
  });

  const fetchEvidence = async (caseId: string | null) => {
    if (!caseId) {
      set({ evidence: [], isLoading: false, error: null }); // Clear evidence if no case is selected
      return;
    }
    update((state) => ({ ...state, isLoading: true, error: null }));
    try {
      const response = await fetch(`/api/evidence/${caseId}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to fetch evidence");
      }
      const evidenceList: Evidence[] = await response.json();
      set({ evidence: evidenceList, isLoading: false, error: null });
    } catch (error: any) {
      console.error("Error fetching evidence:", error);
      set({ evidence: [], isLoading: false, error: error.message });
    }
  };

  // When the selected case changes, automatically fetch new evidence
  selectedCase.subscribe((caseId) => {
    fetchEvidence(caseId);
  });

  return {
    subscribe,
    fetchEvidence,
    // Add a new piece of evidence
    addEvidence: async (
      newEvidenceData: Omit<Evidence, "id" | "x" | "y" | "caseId">,
    ) => {
      update((state) => ({ ...state, isLoading: true }));
      const currentCaseId = get(selectedCase);
      if (!currentCaseId) {
        const err = "No case selected to add evidence to.";
        update((state) => ({ ...state, isLoading: false, error: err }));
        console.error(err);
        return;
      }
      try {
        const response = await fetch("/api/evidence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...newEvidenceData, caseId: currentCaseId }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || "Failed to add evidence");
        }
        const createdEvidence: Evidence = await response.json();
        update((state) => ({
          ...state,
          evidence: [...state.evidence, createdEvidence],
          isLoading: false,
        }));
      } catch (error: any) {
        update((state) => ({
          ...state,
          isLoading: false,
          error: error.message,
        }));
        console.error("Error adding evidence:", error);
      }
    },
    // Update evidence (e.g., position on canvas) with optimistic update
    updateEvidence: async (
      evidenceId: string,
      updates: Partial<Omit<Evidence, "id" | "caseId">>,
    ) => {
      let originalEvidence: Evidence | undefined;

      update((state) => {
        originalEvidence = state.evidence.find(
          (item) => item.id === evidenceId,
        );
        const newEvidence = state.evidence.map((item) =>
          item.id === evidenceId ? { ...item, ...updates } : item,
        );
        return { ...state, evidence: newEvidence };
      });

      try {
        const response = await fetch(`/api/evidence/${evidenceId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updates),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || "Failed to update evidence");
        }
      } catch (error: any) {
        console.error("Error updating evidence:", error);
        // Revert optimistic update on failure
        if (originalEvidence) {
          const original = originalEvidence; // closure capture
          update((state) => ({
            ...state,
            evidence: state.evidence.map((item) =>
              item.id === evidenceId ? original : item,
            ),
            error: error.message,
          }));
        }
      }
    },
    // Delete a piece of evidence with optimistic update
    deleteEvidence: async (evidenceId: string) => {
      let originalList: Evidence[] = [];
      update((state) => {
        originalList = state.evidence;
        const newList = state.evidence.filter((item) => item.id !== evidenceId);
        return { ...state, evidence: newList };
      });

      try {
        const response = await fetch(`/api/evidence/${evidenceId}`, {
          method: "DELETE",
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || "Failed to delete evidence");
        }
      } catch (error: any) {
        console.error("Error deleting evidence:", error);
        // Revert optimistic update on failure
        update((state) => ({
          ...state,
          evidence: originalList,
          error: error.message,
        }));
      }
    },
    clearError: () => {
      update((state) => ({ ...state, error: null }));
    },
  };
};

export const evidence = createEvidenceStore();
export default evidence;
