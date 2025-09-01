
import { browser } from "$app/environment";
// Orphaned content: import Fuse from "fuse.js";
import { derived, writable } from "svelte/store";

// Lightweight Fuse fallback if real library not present (prevents build break)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Fuse: any = (globalThis as any).Fuse || class {
  list: any[]; keys: any[]; constructor(list: any[], options: any) { this.list = list; this.keys = options.keys?.map((k: any) => k.name) || []; }
  search(term: string) { const lower = term.toLowerCase(); return this.list.filter(item => this.keys.some(k => String((item as any)[k] ?? '').toLowerCase().includes(lower))).map(i => ({ item: i })); }
};

// Placeholder indexedDB utilities
const idbUtils = {
  del: async (key: string) => localStorage.removeItem(key),
  get: async (key: string) => {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  },
  keys: async () => Object.keys(localStorage).filter(k => k.startsWith('note:')),
  set: async (key: string, value: any) => localStorage.setItem(key, JSON.stringify(value))
};

export interface SavedNote {
  id: string;
  title: string;
  content: string;
  markdown: string;
  html: string;
  contentJson: any;
  noteType: string;
  tags: string[];
  caseId?: string;
  userId: string;
  savedAt: Date;
  metadata?: unknown;
}

export interface NoteFilters {
  search: string;
  noteType: string;
  tags: string[];
  caseId?: string;
}

// Main store for saved notes
export const savedNotes = writable<SavedNote[]>([]);
// Filters store
export const noteFilters = writable<NoteFilters>({
  search: "",
  noteType: "",
  tags: [],
  caseId: undefined,
});

// Derived store for filtered notes with fuzzy search
export const filteredNotes = derived(
  [savedNotes, noteFilters],
  ([$savedNotes, $noteFilters]) => {
    let notes = $savedNotes;

    // Apply basic filters first
    if ($noteFilters.noteType) {
      notes = notes.filter((note) => note.noteType === $noteFilters.noteType);
    }
    if ($noteFilters.caseId) {
      notes = notes.filter((note) => note.caseId === $noteFilters.caseId);
    }
    if ($noteFilters.tags.length > 0) {
      notes = notes.filter((note) =>
        $noteFilters.tags.some((tag) => note.tags.includes(tag))
      );
    }
    // Apply fuzzy search
    if ($noteFilters.search.trim()) {
      const fuse = new Fuse(notes, {
        keys: [
          { name: "title", weight: 0.4 },
          { name: "content", weight: 0.3 },
          { name: "markdown", weight: 0.2 },
          { name: "tags", weight: 0.1 },
        ],
        threshold: 0.4,
        includeScore: true,
      });

      const results = fuse.search($noteFilters.search);
      return results.map((result) => result.item);
    }
    return notes.sort(
      (a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime()
    );
  }
);

// Derived stores for quick stats
export const noteStats = derived(savedNotes, ($savedNotes) => ({
  total: $savedNotes.length,
  byType: $savedNotes.reduce(
    (acc, note) => {
      acc[note.noteType] = (acc[note.noteType] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  ),
  totalTags: Array.from(new Set($savedNotes.flatMap((note) => note.tags)))
    .length,
}));

class NotesManager {
  private static instance: NotesManager;
  private dbPrefix = "warden-note-";

  static getInstance(): NotesManager {
    if (!NotesManager.instance) {
      NotesManager.instance = new NotesManager();
    }
    return NotesManager.instance;
  }

  // Save note to both store and IndexedDB
  async saveNote(note: Omit<SavedNote, "savedAt">): Promise<void> {
    const noteWithTimestamp: SavedNote = {
      ...note,
      savedAt: new Date(),
    };

    // Update store
    savedNotes.update((notes) => {
      const existingIndex = notes.findIndex((n) => n.id === note.id);
      if (existingIndex >= 0) {
        notes[existingIndex] = noteWithTimestamp;
        return notes;
      } else {
        return [noteWithTimestamp, ...notes];
      }
    });

    // Save to IndexedDB for offline access
    if (browser) {
      try {
        await idbUtils.set(`${this.dbPrefix}${note.id}`, noteWithTimestamp);
      } catch (error: any) {
        console.warn("Failed to save note to IndexedDB:", error);
      }
    }
  }

  // Remove note from store and IndexedDB
  async removeNote(noteId: string): Promise<void> {
    savedNotes.update((notes) => notes.filter((n) => n.id !== noteId));

    if (browser) {
      try {
        await idbUtils.del(`${this.dbPrefix}${noteId}`);
      } catch (error: any) {
        console.warn("Failed to remove note from IndexedDB:", error);
      }
    }
  }

  // Load all notes from IndexedDB on app start
  async loadSavedNotes(): Promise<void> {
    if (!browser) return;

    try {
      const allKeys = await idbUtils.keys();
      const noteKeys = allKeys.filter(
        (key) => typeof key === "string" && key.startsWith(this.dbPrefix)
      );

      const notes: SavedNote[] = [];
      for (const key of noteKeys) {
        try {
          const note = await idbUtils.get(key);
          if (note && this.isValidNote(note)) {
            // Ensure savedAt is a Date object
            note.savedAt = new Date(note.savedAt);
            notes.push(note);
          }
        } catch (error: any) {
          console.warn("Failed to load note from IndexedDB:", key, error);
        }
      }
      savedNotes.set(
        notes.sort(
          (a, b) =>
            new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime()
        )
      );
    } catch (error: any) {
      console.warn("Failed to load notes from IndexedDB:", error);
    }
  }

  // Sync with server
  async syncWithServer(apiEndpoint: string = "/api/notes"): Promise<void> {
    try {
      const response = await fetch(`${apiEndpoint}/sync`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const serverNotes: SavedNote[] = await response.json();

        // Merge with local notes, preferring newer timestamps
        savedNotes.update((localNotes) => {
          const merged = new Map<string, SavedNote>();

          // Add local notes
          localNotes.forEach((note) => merged.set(note.id, note));

          // Add or update with server notes
          serverNotes.forEach((serverNote) => {
            const localNote = merged.get(serverNote.id);
            if (
              !localNote ||
              new Date(serverNote.savedAt) > new Date(localNote.savedAt)
            ) {
              merged.set(serverNote.id, serverNote);
            }
          });

          return Array.from(merged.values()).sort(
            (a, b) =>
              new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime()
          );
        });
      }
    } catch (error: any) {
      console.warn("Failed to sync with server:", error);
    }
  }

  // Utility to check if object is a valid note
  private isValidNote(obj: any): obj is SavedNote {
    return (
      obj &&
      typeof obj.id === "string" &&
      typeof obj.title === "string" &&
      typeof obj.content === "string" &&
      Array.isArray(obj.tags)
    );
  }

  // Export notes to file
  async exportNotes(format: "json" | "markdown" = "json"): Promise<void> {
    if (!browser) return;

    const notes = await new Promise<SavedNote[]>((resolve) => {
      const unsubscribe = savedNotes.subscribe((notes) => {
        resolve(notes);
        unsubscribe();
      });
    });

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === "markdown") {
      content = notes
        .map((note) => {
          return `# ${note.title}\n\n${note.markdown || note.content}\n\n---\n\n`;
        })
        .join("");
      filename = `notes-export-${new Date().toISOString().split("T")[0]}.md`;
      mimeType = "text/markdown";
    } else {
      content = JSON.stringify(notes, null, 2);
      filename = `notes-export-${new Date().toISOString().split("T")[0]}.json`;
      mimeType = "application/json";
    }
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

// Export singleton instance
export const notesManager = NotesManager.getInstance();
;
// Convenience functions
export async function saveNoteForLater(note: Omit<SavedNote, "savedAt">): Promise<any> {
  await notesManager.saveNote(note);
}

export async function removeSavedNote(noteId: string): Promise<any> {
  await notesManager.removeNote(noteId);
}

export async function loadSavedNotes(): Promise<any> {
  await notesManager.loadSavedNotes();
}

export function setNoteFilter(filter: Partial<NoteFilters>) {
  noteFilters.update((current) => ({ ...current, ...filter }));
}

export function clearNoteFilters() {
  noteFilters.set({
    search: "",
    noteType: "",
    tags: [],
    caseId: undefined,
  });
}

export default notesManager;
