import crypto from "crypto";

import { writable } from "svelte/store";
// TODO: Fix import - // Orphaned content: import {  import type { Evidence } from '$lib/data/types.js';

let lokiDb: any = null;
let db: any = null;
let evidenceCollection: any = null;
let canvasStateCollection: any = null;
let notesCollection: any = null;

// Initialize Loki database
async function initLoki() {
  if (!browser) return;

  try {
    const Loki = (await import("lokijs")).default;

    lokiDb = new Loki("canvas-cache.db", {
      autoload: true,
      autoloadCallback: databaseInitialize,
      autosave: true,
      autosaveInterval: 5000,
    } as any);
  } catch (error) {
    console.error("Failed to initialize Loki:", error);
  }
}
function databaseInitialize() {
  // Evidence collection
  evidenceCollection = lokiDb.getCollection("evidence");
  if (!evidenceCollection) {
    evidenceCollection = lokiDb.addCollection("evidence", {
      indices: ["id", "caseId", "type", "tags"],
      unique: ["id"],
    });
  }
  // Canvas states collection
  canvasStateCollection = lokiDb.getCollection("canvasStates");
  if (!canvasStateCollection) {
    canvasStateCollection = lokiDb.addCollection("canvasStates", {
      indices: ["reportId"],
      unique: ["reportId"],
    });
  }
  // Notes collection
  notesCollection = lokiDb.getCollection("notes");
  if (!notesCollection) {
    notesCollection = lokiDb.addCollection("notes", {
      indices: ["id", "reportId", "type", "tags"],
      unique: ["id"],
    });
  }
}
// Store for Loki operations
export const lokiStore = writable({
  initialized: false,
  evidence: [] as Evidence[],
  canvasStates: [] as unknown[],
  notes: [] as unknown[],
});

// Loki operations
export const loki = {
  // Initialize the database
  async init() {
    await initLoki();
    lokiStore.update((state) => ({ ...state, initialized: true }));
  },

  // Evidence operations
  evidence: {
    add(evidence: Evidence) {
      if (!evidenceCollection) return;

      const existing = evidenceCollection.findOne({ id: evidence.id });
      if (existing) {
        evidenceCollection.update({ ...existing, ...evidence });
      } else {
        evidenceCollection.insert(evidence);
      }
      this.refreshStore();
    },

    getAll(): Evidence[] {
      if (!evidenceCollection) return [];
      return evidenceCollection.find();
    },

    getByCaseId(caseId: string): Evidence[] {
      if (!evidenceCollection) return [];
      return evidenceCollection.find({ caseId });
    },

    search(query: string): Evidence[] {
      if (!evidenceCollection || !query) return this.getAll();

      return evidenceCollection.find({
        $or: [
          { fileName: { $regex: new RegExp(query, "i") } },
          { description: { $regex: new RegExp(query, "i") } },
          { tags: { $contains: query } },
        ],
      });
    },

    remove(id: string) {
      if (!evidenceCollection) return;
      evidenceCollection.findAndRemove({ id });
      this.refreshStore();
    },

    refreshStore() {
      lokiStore.update((state) => ({
        ...state,
        evidence: this.getAll(),
      }));
    },
  },

  // Canvas state operations
  canvasState: {
    save(reportId: string, canvasData: any) {
      if (!canvasStateCollection) return;

      const existing = canvasStateCollection.findOne({ reportId });
      const stateData = {
        reportId,
        canvasData,
        lastModified: new Date().toISOString(),
      };

      if (existing) {
        canvasStateCollection.update({ ...existing, ...stateData });
      } else {
        canvasStateCollection.insert(stateData);
      }
      this.refreshStore();
    },

    load(reportId: string) {
      if (!canvasStateCollection) return null;
      return canvasStateCollection.findOne({ reportId });
    },

    getAll() {
      if (!canvasStateCollection) return [];
      return canvasStateCollection.find();
    },

    get(id: string) {
      if (!canvasStateCollection) return null;
      return canvasStateCollection.findOne({ id }) || null;
    },

    remove(reportId: string) {
      if (!canvasStateCollection) return;
      canvasStateCollection.findAndRemove({ reportId });
      this.refreshStore();
    },

    delete(id: string) {
      if (!canvasStateCollection) return;
      canvasStateCollection.findAndRemove({ id });
      this.refreshStore();
    },

    refreshStore() {
      lokiStore.update((state) => ({
        ...state,
        canvasStates: this.getAll(),
      }));
    },
  },

  // Convenience methods for API compatibility
  async saveCanvasState(canvasState: any) {
    if (!canvasStateCollection) return;

    const existing = canvasStateCollection.findOne({ id: canvasState.id });
    if (existing) {
      canvasStateCollection.update({ ...existing, ...canvasState });
    } else {
      canvasStateCollection.insert({
        ...canvasState,
        id: canvasState.id || crypto.randomUUID(),
        createdAt: canvasState.createdAt || new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
    }
    this.canvasState.refreshStore();
  },

  async getCanvasState(id: string) {
    return this.canvasState.get(id);
  },

  async getAllCanvasStates() {
    return this.canvasState.getAll();
  },

  async deleteCanvasState(id: string) {
    this.canvasState.delete(id);
  },

  // Notes operations
  notes: {
    add(note: any) {
      if (!notesCollection) return;

      const existing = notesCollection.findOne({ id: note.id });
      if (existing) {
        notesCollection.update({ ...existing, ...note });
      } else {
        notesCollection.insert({
          ...note,
          id: note.id || crypto.randomUUID(),
          createdAt: note.createdAt || new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
      }
      this.refreshStore();
    },

    getByReportId(reportId: string) {
      if (!notesCollection) return [];
      return notesCollection.find({ reportId });
    },

    search(query: string): unknown[] {
      if (!notesCollection || !query) return this.getAll();

      return notesCollection.find({
        $or: [
          { title: { $regex: new RegExp(query, "i") } },
          { content: { $regex: new RegExp(query, "i") } },
          { tags: { $contains: query } },
        ],
      });
    },

    getAll() {
      if (!notesCollection) return [];
      return notesCollection.find();
    },

    remove(id: string) {
      if (!notesCollection) return;
      notesCollection.findAndRemove({ id });
      this.refreshStore();
    },

    refreshStore() {
      lokiStore.update((state) => ({
        ...state,
        notes: this.getAll(),
      }));
    },
  },

  // Clear all data
  clearAll() {
    if (evidenceCollection) evidenceCollection.clear();
    if (canvasStateCollection) canvasStateCollection.clear();
    if (notesCollection) notesCollection.clear();

    lokiStore.update((state) => ({
      ...state,
      evidence: [],
      canvasStates: [],
      notes: [],
    }));
  },
};
