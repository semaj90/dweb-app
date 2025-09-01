import crypto from "crypto";

import { writable, derived, type Writable } from "svelte/store";
// Orphaned content: import type { Writable as WritableType

// Evidence types
export interface Evidence {
  id: string;
  type: "document" | "image" | "video" | "audio" | "link";
  title: string;
  description?: string;
  url?: string;
  file?: File;
  metadata?: Record<string, any>;
  tags?: string[];
  createdAt: Date;
  updatedAt: Date;
}
// Report structure
export interface Report {
  id: string;
  title: string;
  content: string; // TinyMCE HTML content
  attachedEvidence: Evidence[];
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    version: number;
    status: "draft" | "review" | "final";
    tags: string[];
    classification?: string;
  };
  settings: {
    autoSave: boolean;
    theme: "light" | "dark";
    layout: "single" | "dual" | "masonry";
  };
}
// Default report
const defaultReport: Report = {
  id: crypto.randomUUID(),
  title: "Untitled Report",
  content: "<p>Begin writing your report...</p>",
  attachedEvidence: [],
  metadata: {
    createdAt: new Date(),
    updatedAt: new Date(),
    version: 1,
    status: "draft",
    tags: [],
  },
  settings: {
    autoSave: true,
    theme: "light",
    layout: "single",
  },
};

// Main report store
export const report: Writable<Report> = writable(defaultReport);
// Editor state
export const editorState = writable({
  isEditing: false,
  hasUnsavedChanges: false,
  lastSaved: new Date(),
  wordCount: 0,
  selectedText: "",
  cursorPosition: 0,
});

// UI state for report editing
export const reportUI = writable({
  sidebarOpen: true,
  sidebarWidth: 300,
  evidencePanelOpen: true,
  evidencePanelHeight: 400,
  toolbarCollapsed: false,
  fullscreen: false,
});

// Derived stores
export const reportTitle = derived(report, ($report) => $report.title);
export const reportContent = derived(report, ($report) => $report.content);
export const attachedEvidence = derived(report, ($report) => $report.attachedEvidence);
export const reportMetadata = derived(report, ($report) => $report.metadata);
export const hasUnsavedChanges = derived(editorState, ($state) => $state.hasUnsavedChanges);

// Actions
export const reportActions = {
  updateTitle: (title: string) => {
    report.update((r) => ({
      ...r,
      title,
      metadata: { ...r.metadata, updatedAt: new Date() },
    }));
    editorState.update((s) => ({ ...s, hasUnsavedChanges: true }));
  },

  updateContent: (content: string) => {
    report.update((r) => ({
      ...r,
      content,
      metadata: { ...r.metadata, updatedAt: new Date() },
    }));
    editorState.update((s) => ({
      ...s,
      hasUnsavedChanges: true,
      wordCount: content
        .replace(/<[^>]*>/g, "")
        .trim()
        .split(/\s+/).length,
    }));
  },

  addEvidence: (evidence: Omit<Evidence, "id" | "createdAt" | "updatedAt">) => {
    const newEvidence: Evidence = {
      ...evidence,
      id: crypto.randomUUID(),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    report.update((r) => ({
      ...r,
      attachedEvidence: [...r.attachedEvidence, newEvidence],
      metadata: { ...r.metadata, updatedAt: new Date() },
    }));
    editorState.update((s) => ({ ...s, hasUnsavedChanges: true }));
  },

  removeEvidence: (evidenceId: string) => {
    report.update((r) => ({
      ...r,
      attachedEvidence: r.attachedEvidence.filter((e: any) => e.id !== evidenceId),
      metadata: { ...r.metadata, updatedAt: new Date() },
    }));
    editorState.update((s) => ({ ...s, hasUnsavedChanges: true }));
  },

  reorderEvidence: (newOrder: Evidence[]) => {
    report.update((r) => ({
      ...r,
      attachedEvidence: newOrder,
      metadata: { ...r.metadata, updatedAt: new Date() },
    }));
    editorState.update((s) => ({ ...s, hasUnsavedChanges: true }));
  },

  updateSettings: (settings: Partial<Report["settings"]>) => {
    report.update((r) => ({
      ...r,
      settings: { ...r.settings, ...settings },
      metadata: { ...r.metadata, updatedAt: new Date() },
    }));
  },

  save: async () => {
    // TODO: Implement actual save logic (API call)
    editorState.update((s) => ({
      ...s,
      hasUnsavedChanges: false,
      lastSaved: new Date(),
    }));
  },

  load: (reportData: Report) => {
    report.set(reportData);
    editorState.update((s) => ({
      ...s,
      hasUnsavedChanges: false,
      lastSaved: new Date(),
    }));
  },

  reset: () => {
    report.set(defaultReport);
    editorState.update((s) => ({
      ...s,
      hasUnsavedChanges: false,
      lastSaved: new Date(),
    }));
  },
};

// Auto-save functionality
export const setupAutoSave = (intervalMs: number = 30000) => {
  let autoSaveInterval: NodeJS.Timeout;

  const unsubscribe = report.subscribe(($report) => {
    if ($report.settings.autoSave) {
      clearInterval(autoSaveInterval);
      autoSaveInterval = setInterval(() => {
        editorState.subscribe(($state) => {
          if ($state.hasUnsavedChanges) {
            reportActions.save();
          }
        })();
      }, intervalMs);
    }
  });

  return () => {
    clearInterval(autoSaveInterval);
    unsubscribe();
  };
};
