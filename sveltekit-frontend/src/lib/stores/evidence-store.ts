import crypto from "crypto";

import Fuse from "fuse.js";
import type {     Writable     } from 'svelte/store';
import { derived, writable } from "svelte/store";

// === TYPES ===

export interface Evidence {
  id: string;
  caseId: string;
  criminalId?: string;
  title: string;
  description?: string;
  type: string; // Required property for compatibility
  evidenceType: string;
  fileType?: string;
  subType?: string;
  fileUrl?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  hash?: string;
  tags: string[];
  chainOfCustody: any[];
  collectedAt?: Date;
  collectedBy?: string;
  location?: string;
  labAnalysis: any;
  aiAnalysis: any;
  aiTags: string[];
  aiSummary?: string;
  summary?: string;
  isAdmissible: boolean;
  confidentialityLevel: string;
  canvasPosition: any;
  uploadedBy: string;
  uploadedAt: Date;
  updatedAt: Date;
  // Add missing properties for compatibility
  analysis?: {
    summary: string;
    keyPoints: string[];
    relevance: number;
    admissibility: "admissible" | "questionable" | "inadmissible";
    reasoning: string;
    suggestedTags: string[];
  };
}

export interface UploadFile {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "processing" | "completed" | "error";
  preview?: string;
  extractedText?: string;
  aiAnalysis?: unknown;
  error?: string;
}

export interface EvidenceGridState {
  items: Evidence[];
  searchQuery: string;
  filteredItems: Evidence[];
  sortBy: "uploadedAt" | "title" | "evidenceType" | "fileSize";
  sortOrder: "asc" | "desc";
  selectedItems: Set<string>;
  viewMode: "grid" | "list";
  isLoading: boolean;
  error?: string;
}

export interface UploadModalState {
  isOpen: boolean;
  caseId?: string;
  files: UploadFile[];
  step: "select" | "preview" | "analyze" | "confirm";
  isProcessing: boolean;
  error?: string;
}

// === STORES ===

// Evidence Grid Store
export const evidenceGrid: Writable<EvidenceGridState> = writable({
  items: [],
  searchQuery: "",
  filteredItems: [],
  sortBy: "uploadedAt",
  sortOrder: "desc",
  selectedItems: new Set(),
  viewMode: "grid",
  isLoading: false,
});

// Upload Modal Store
export const uploadModal: Writable<UploadModalState> = writable({
  isOpen: false,
  files: [],
  step: "select",
  isProcessing: false,
});

// === FUSE.JS SEARCH ===

let fuseInstance: Fuse<Evidence> | null = null;
let lastItemsLength = 0;

const fuseOptions = {
  keys: [
    { name: "title", weight: 0.3 },
    { name: "description", weight: 0.2 },
    { name: "tags", weight: 0.2 },
    { name: "evidenceType", weight: 0.1 },
    { name: "fileName", weight: 0.1 },
    { name: "aiSummary", weight: 0.1 },
  ],
  threshold: 0.3,
  includeScore: true,
  includeMatches: true,
};

// Derived store for filtered evidence
export const filteredEvidence = derived(evidenceGrid, ($evidenceGrid) => {
  const { items, searchQuery, sortBy, sortOrder } = $evidenceGrid;

  let filtered = items;

  // Apply search filter
  if (searchQuery.trim()) {
    if (!fuseInstance || lastItemsLength !== items.length) {
      fuseInstance = new Fuse(items, fuseOptions);
      lastItemsLength = items.length;
    }
    const searchResults = fuseInstance.search(searchQuery);
    filtered = searchResults.map((result) => result.item);
  }

  // Apply sorting
  filtered.sort((a, b) => {
    let aVal: any, bVal: any;

    switch (sortBy) {
      case "title":
        aVal = a.title.toLowerCase();
        bVal = b.title.toLowerCase();
        break;
      case "evidenceType":
        aVal = a.evidenceType.toLowerCase();
        bVal = b.evidenceType.toLowerCase();
        break;
      case "fileSize":
        aVal = a.fileSize || 0;
        bVal = b.fileSize || 0;
        break;
      case "uploadedAt":
      default:
        aVal = new Date(a.uploadedAt).getTime();
        bVal = new Date(b.uploadedAt).getTime();
        break;
    }
    if (aVal < bVal) return sortOrder === "asc" ? -1 : 1;
    if (aVal > bVal) return sortOrder === "asc" ? 1 : -1;
    return 0;
  });

  return filtered;
});

// === ACTIONS ===

export const evidenceActions = {
  // Set items directly (useful for SSR initialization)
  setItems(items: Evidence[]) {
    evidenceGrid.update((state) => ({
      ...state,
      items,
      filteredItems: items,
      isLoading: false,
      error: undefined,
    }));
  },

  // Load evidence from API
  async loadEvidence(caseId?: string) {
    evidenceGrid.update((state) => ({
      ...state,
      isLoading: true,
      error: undefined,
    }));

    try {
      const url = caseId ? `/api/evidence?caseId=${caseId}` : "/api/evidence";
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to load evidence: ${response.statusText}`);
      }
      const items: Evidence[] = await response.json();

      evidenceGrid.update((state) => ({
        ...state,
        items,
        filteredItems: items,
        isLoading: false,
      }));
    } catch (error: any) {
      evidenceGrid.update((state) => ({
        ...state,
        isLoading: false,
        error:
          error instanceof Error ? error.message : "Failed to load evidence",
      }));
    }
  },

  // Update search query
  setSearchQuery(query: string) {
    evidenceGrid.update((state) => ({ ...state, searchQuery: query }));
  },

  // Update sort settings
  setSorting(
    sortBy: EvidenceGridState["sortBy"],
    sortOrder: EvidenceGridState["sortOrder"]
  ) {
    evidenceGrid.update((state) => ({ ...state, sortBy, sortOrder }));
  },

  // Toggle item selection
  toggleSelection(itemId: string) {
    evidenceGrid.update((state) => {
      const newSelected = new Set(state.selectedItems);
      if (newSelected.has(itemId)) {
        newSelected.delete(itemId);
      } else {
        newSelected.add(itemId);
      }
      return { ...state, selectedItems: newSelected };
    });
  },

  // Clear all selections
  clearSelection() {
    evidenceGrid.update((state) => ({ ...state, selectedItems: new Set() }));
  },

  // Toggle view mode
  setViewMode(viewMode: EvidenceGridState["viewMode"]) {
    evidenceGrid.update((state) => ({ ...state, viewMode }));
  },

  // Delete evidence
  async deleteEvidence(evidenceId: string) {
    try {
      const response = await fetch(`/api/evidence/${evidenceId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`Failed to delete evidence: ${response.statusText}`);
      }
      evidenceGrid.update((state) => ({
        ...state,
        items: state.items.filter((item) => item.id !== evidenceId),
        selectedItems: new Set(
          [...state.selectedItems].filter((id) => id !== evidenceId)
        ),
      }));
    } catch (error: any) {
      console.error("Delete evidence error:", error);
      throw error;
    }
  },
};

export const uploadActions = {
  // Open upload modal
  openModal(caseId?: string) {
    uploadModal.update((state) => ({
      ...state,
      isOpen: true,
      caseId,
      files: [],
      step: "select",
      isProcessing: false,
      error: undefined,
    }));
  },

  // Close upload modal
  closeModal() {
    uploadModal.update((state) => ({
      ...state,
      isOpen: false,
      files: [],
      step: "select",
      isProcessing: false,
      error: undefined,
    }));
  },

  // Add files for upload
  addFiles(files: FileList | File[]) {
    const fileArray = Array.from(files);
    const uploadFiles: UploadFile[] = fileArray.map((file) => ({
      id: crypto.randomUUID(),
      file,
      progress: 0,
      status: "pending",
    }));

    uploadModal.update((state) => ({
      ...state,
      files: [...state.files, ...uploadFiles],
      step: "preview",
    }));

    // Generate previews for supported file types
    uploadFiles.forEach((uploadFile) => {
      if (uploadFile.file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e: any) => {
          uploadModal.update((state) => ({
            ...state,
            files: state.files.map((f) =>
              f.id === uploadFile.id
                ? { ...f, preview: e.target?.result as string }
                : f
            ),
          }));
        };
        reader.readAsDataURL(uploadFile.file);
      }
    });
  },

  // Remove file from upload queue
  removeFile(fileId: string) {
    uploadModal.update((state) => ({
      ...state,
      files: state.files.filter((f) => f.id !== fileId),
    }));
  },

  // Set upload step
  setStep(step: UploadModalState["step"]) {
    uploadModal.update((state) => ({ ...state, step }));
  },

  // Upload files
  async uploadFiles() {
    uploadModal.update((state) => ({
      ...state,
      isProcessing: true,
      error: undefined,
    }));

    try {
      const state = await new Promise<UploadModalState>((resolve) => {
        uploadModal.subscribe(resolve)();
      });

      for (const uploadFile of state.files) {
        if (uploadFile.status === "completed") continue;

        // Update file status to uploading
        uploadModal.update((modalState) => ({
          ...modalState,
          files: modalState.files.map((f) =>
            f.id === uploadFile.id ? { ...f, status: "uploading" as const } : f
          ),
        }));

        const formData = new FormData();
        formData.append("file", uploadFile.file);
        if (state.caseId) formData.append("caseId", state.caseId);

        const xhr = new XMLHttpRequest();

        // Track upload progress
        xhr.upload.addEventListener("progress", (e: any) => {
          if (e.lengthComputable) {
            const progress = Math.round((e.loaded / e.total) * 100);
            uploadModal.update((modalState) => ({
              ...modalState,
              files: modalState.files.map((f) =>
                f.id === uploadFile.id ? { ...f, progress } : f
              ),
            }));
          }
        });

        // Handle upload completion
        await new Promise<void>((resolve, reject) => {
          xhr.onload = () => {
            if (xhr.status === 200 || xhr.status === 201) {
              const result = JSON.parse(xhr.responseText);
              uploadModal.update((modalState) => ({
                ...modalState,
                files: modalState.files.map((f) =>
                  f.id === uploadFile.id
                    ? {
                        ...f,
                        status: "completed" as const,
                        progress: 100,
                        aiAnalysis: result.aiAnalysis,
                        extractedText: result.extractedText,
                      }
                    : f
                ),
              }));
              resolve();
            } else {
              const error = `Upload failed: ${xhr.statusText}`;
              uploadModal.update((modalState) => ({
                ...modalState,
                files: modalState.files.map((f) =>
                  f.id === uploadFile.id
                    ? { ...f, status: "error" as const, error }
                    : f
                ),
              }));
              reject(new Error(error));
            }
          };

          xhr.onerror = () => {
            const error = "Upload failed: Network error";
            uploadModal.update((modalState) => ({
              ...modalState,
              files: modalState.files.map((f) =>
                f.id === uploadFile.id
                  ? { ...f, status: "error" as const, error }
                  : f
              ),
            }));
            reject(new Error(error));
          };

          xhr.open("POST", "/api/evidence/upload");
          xhr.send(formData);
        });
      }

      // Reload evidence after successful upload
      await evidenceActions.loadEvidence(state.caseId);

      uploadModal.update((modalState) => ({
        ...modalState,
        isProcessing: false,
      }));
    } catch (error: any) {
      uploadModal.update((state) => ({
        ...state,
        isProcessing: false,
        error: error instanceof Error ? error.message : "Upload failed",
      }));
    }
  },
};

export default evidenceActions;
