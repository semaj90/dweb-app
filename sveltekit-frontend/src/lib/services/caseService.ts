import crypto from "crypto";

// Enhanced case service with proper typing and error handling
import type { Writable } from "svelte/store";
import { get, writable } from "svelte/store";

// Types
export interface Report {
  id: string;
  title: string;
  content?: string;
  posX: number;
  posY: number;
  caseId: string;
  createdAt?: string | Date;
  updatedAt?: string | Date;
  // Additional properties used by components
  status?: string;
  tags?: string[];
  isPublic?: boolean;
  createdBy?: string;
  metadata?: Record<string, any>;
  reportType?: string;
}
export interface Evidence {
  id: string;
  title: string;
  fileUrl: string;
  posX: number;
  posY: number;
  caseId: string;
  metadata?: Record<string, any>;
  createdAt?: string;
  updatedAt?: string;
  // Canvas-specific properties
  isSelected?: boolean;
  isDirty?: boolean;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  name?: string;
  url?: string;
}
export interface POI {
  id: string;
  name: string;
  posX: number;
  posY: number;
  relationship?: string;
  caseId: string;
  description?: string;
  createdAt?: string;
  updatedAt?: string;
  // Additional properties expected by POINode component
  aliases?: string[];
  profileImageUrl?: string;
  profileData?: {
    who: string;
    what: string;
    why: string;
    how: string;
  };
  threatLevel?: string;
  status?: string;
  tags?: string[];
  createdBy?: string;
}
export interface CaseData {
  id: string;
  title: string;
  description?: string;
  reports: Report[];
  evidence: Evidence[];
  pois: POI[];
}
// Store creation
function createCaseService() {
  // State stores
  const reports: Writable<Report[]> = writable([]);
  const evidence: Writable<Evidence[]> = writable([]);
  const pois: Writable<POI[]> = writable([]);
  const isLoading: Writable<boolean> = writable(false);
  const error: Writable<string | null> = writable(null);

  // Current case ID
  let currentCaseId: string | null = null;

  // Helper function for API calls
  async function apiCall<T>(
    url: string,
    options: RequestInit = {},
  ): Promise<T> {
    try {
      error.set(null);
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      error.set(message);
      throw err;
    }
  }
  // Load case data
  async function loadCase(caseId: string) {
    if (!caseId) return;

    currentCaseId = caseId;
    isLoading.set(true);

    try {
      const data = await apiCall<CaseData>(`/api/cases/${caseId}`);

      reports.set(data.reports || []);
      evidence.set(data.evidence || []);
      pois.set(data.pois || []);
    } catch (err) {
      console.error("Failed to load case:", err);
      // Reset stores on error
      reports.set([]);
      evidence.set([]);
      pois.set([]);
    } finally {
      isLoading.set(false);
    }
  }
  // Create report
  async function createReport(data: Partial<Report>) {
    if (!currentCaseId) {
      error.set("No case loaded");
      return;
    }
    try {
      const newReport = await apiCall<Report>("/api/reports", {
        method: "POST",
        body: JSON.stringify({
          ...data,
          caseId: currentCaseId,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
        }),
      });

      reports.update((items) => [...items, newReport]);
      return newReport;
    } catch (err) {
      console.error("Failed to create report:", err);
      return null;
    }
  }
  // Create evidence
  async function createEvidence(data: Partial<Evidence>) {
    if (!currentCaseId) {
      error.set("No case loaded");
      return;
    }
    try {
      const newEvidence = await apiCall<Evidence>("/api/evidence", {
        method: "POST",
        body: JSON.stringify({
          ...data,
          caseId: currentCaseId,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
        }),
      });

      evidence.update((items) => [...items, newEvidence]);
      return newEvidence;
    } catch (err) {
      console.error("Failed to create evidence:", err);
      return null;
    }
  }
  // Create POI
  async function createPOI(data: Partial<POI>) {
    if (!currentCaseId) {
      error.set("No case loaded");
      return;
    }
    try {
      const newPOI = await apiCall<POI>("/api/pois", {
        method: "POST",
        body: JSON.stringify({
          ...data,
          caseId: currentCaseId,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
        }),
      });

      pois.update((items) => [...items, newPOI]);
      return newPOI;
    } catch (err) {
      console.error("Failed to create POI:", err);
      return null;
    }
  }
  // Update position
  async function updatePosition(
    type: "report" | "evidence" | "poi",
    id: string,
    position: { x: number; y: number },
  ) {
    try {
      await apiCall(`/api/${type}s/${id}/position`, {
        method: "PATCH",
        body: JSON.stringify({ posX: position.x, posY: position.y }),
      });

      // Update local state
      switch (type) {
        case "report":
          reports.update((items) =>
            items.map((item) =>
              item.id === id
                ? { ...item, posX: position.x, posY: position.y }
                : item,
            ),
          );
          break;
        case "evidence":
          evidence.update((items) =>
            items.map((item) =>
              item.id === id
                ? { ...item, posX: position.x, posY: position.y }
                : item,
            ),
          );
          break;
        case "poi":
          pois.update((items) =>
            items.map((item) =>
              item.id === id
                ? { ...item, posX: position.x, posY: position.y }
                : item,
            ),
          );
          break;
      }
    } catch (err) {
      console.error(`Failed to update ${type} position:`, err);
    }
  }
  // Delete item
  async function deleteItem(type: "report" | "evidence" | "poi", id: string) {
    try {
      await apiCall(`/api/${type}s/${id}`, {
        method: "DELETE",
      });

      // Update local state
      switch (type) {
        case "report":
          reports.update((items) => items.filter((item) => item.id !== id));
          break;
        case "evidence":
          evidence.update((items) => items.filter((item) => item.id !== id));
          break;
        case "poi":
          pois.update((items) => items.filter((item) => item.id !== id));
          break;
      }
    } catch (err) {
      console.error(`Failed to delete ${type}:`, err);
    }
  }
  // Save all changes
  async function saveAll() {
    if (!currentCaseId) {
      error.set("No case loaded");
      return;
    }
    isLoading.set(true);

    try {
      await apiCall(`/api/cases/${currentCaseId}/save-all`, {
        method: "POST",
        body: JSON.stringify({
          reports: get(reports),
          evidence: get(evidence),
          pois: get(pois),
        }),
      });

      error.set(null);
    } catch (err) {
      console.error("Failed to save all:", err);
    } finally {
      isLoading.set(false);
    }
  }
  // Reset service
  function reset() {
    currentCaseId = null;
    reports.set([]);
    evidence.set([]);
    pois.set([]);
    error.set(null);
    isLoading.set(false);
  }
  return {
    // State stores
    reports: { subscribe: reports.subscribe },
    evidence: { subscribe: evidence.subscribe },
    pois: { subscribe: pois.subscribe },
    isLoading: { subscribe: isLoading.subscribe },
    error: { subscribe: error.subscribe },

    // Actions
    loadCase,
    createReport,
    createEvidence,
    createPOI,
    updatePosition,
    deleteItem,
    saveAll,
    reset,
  };
}
// Export singleton instance
export const caseService = createCaseService();
