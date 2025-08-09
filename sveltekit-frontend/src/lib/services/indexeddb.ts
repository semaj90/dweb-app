// @ts-nocheck
import type { User } from "$lib/types/user";

// IndexedDB service for client-side caching and offline RAG support
// Stores embeddings, search results, and user interactions locally
import { browser } from "$app/environment";

interface CachedDocument {
  id: string;
  title: string;
  content: string;
  embedding?: number[];
  metadata: Record<string, any>;
  type: "case" | "evidence" | "document";
  lastUpdated: number;
  syncStatus: "synced" | "pending" | "error";
}
interface SearchResult {
  query: string;
  results: any[];
  timestamp: number;
  executionTime: number;
}
interface UserInteraction {
  id: string;
  type: "search" | "view" | "edit" | "ai_query";
  query?: string;
  documentId?: string;
  timestamp: number;
  metadata: Record<string, any>;
}
class IndexedDBService {
  private db: IDBDatabase | null = null;
  private dbName = "prosecutor_rag_db";
  private version = 1;

  constructor() {
    if (browser) {
      this.initDB();
    }
  }
  private async initDB(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Documents store for caching case/evidence data
        if (!db.objectStoreNames.contains("documents")) {
          const documentsStore = db.createObjectStore("documents", {
            keyPath: "id",
          });
          documentsStore.createIndex("type", "type", { unique: false });
          documentsStore.createIndex("lastUpdated", "lastUpdated", {
            unique: false,
          });
          documentsStore.createIndex("syncStatus", "syncStatus", {
            unique: false,
          });
        }
        // Search results cache
        if (!db.objectStoreNames.contains("searchResults")) {
          const searchStore = db.createObjectStore("searchResults", {
            keyPath: "query",
          });
          searchStore.createIndex("timestamp", "timestamp", { unique: false });
        }
        // User interactions for analytics and personalization
        if (!db.objectStoreNames.contains("userInteractions")) {
          const interactionsStore = db.createObjectStore("userInteractions", {
            keyPath: "id",
          });
          interactionsStore.createIndex("type", "type", { unique: false });
          interactionsStore.createIndex("timestamp", "timestamp", {
            unique: false,
          });
          interactionsStore.createIndex("documentId", "documentId", {
            unique: false,
          });
        }
        // Embeddings cache for offline vector search
        if (!db.objectStoreNames.contains("embeddings")) {
          const embeddingsStore = db.createObjectStore("embeddings", {
            keyPath: "id",
          });
          embeddingsStore.createIndex("documentId", "documentId", {
            unique: false,
          });
          embeddingsStore.createIndex("type", "type", { unique: false });
        }
      };
    });
  }
  // Document operations
  async cacheDocument(doc: CachedDocument): Promise<void> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["documents"], "readwrite");
    const store = transaction.objectStore("documents");

    await new Promise<void>((resolve, reject) => {
      const request = store.put({ ...doc, lastUpdated: Date.now() });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
  async getDocument(id: string): Promise<CachedDocument | null> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["documents"], "readonly");
    const store = transaction.objectStore("documents");

    return new Promise((resolve, reject) => {
      const request = store.get(id);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  }
  async searchDocuments(query: string, limit = 10): Promise<CachedDocument[]> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["documents"], "readonly");
    const store = transaction.objectStore("documents");

    return new Promise((resolve, reject) => {
      const documents: CachedDocument[] = [];
      const request = store.openCursor();

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor && documents.length < limit) {
          const doc = cursor.value as CachedDocument;

          // Simple text matching for offline search
          const searchTerm = query.toLowerCase();
          if (
            doc.title.toLowerCase().includes(searchTerm) ||
            doc.content.toLowerCase().includes(searchTerm)
          ) {
            documents.push(doc);
          }
          cursor.continue();
        } else {
          resolve(documents);
        }
      };

      request.onerror = () => reject(request.error);
    });
  }
  async getDocumentsByType(
    type: "case" | "evidence" | "document",
  ): Promise<CachedDocument[]> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["documents"], "readonly");
    const store = transaction.objectStore("documents");
    const index = store.index("type");

    return new Promise((resolve, reject) => {
      const request = index.getAll(type);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  // Search results caching
  async cacheSearchResults(searchResult: SearchResult): Promise<void> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["searchResults"], "readwrite");
    const store = transaction.objectStore("searchResults");

    await new Promise<void>((resolve, reject) => {
      const request = store.put(searchResult);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
  async getCachedSearchResults(query: string): Promise<SearchResult | null> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["searchResults"], "readonly");
    const store = transaction.objectStore("searchResults");

    return new Promise((resolve, reject) => {
      const request = store.get(query);
      request.onsuccess = () => {
        const result = request.result;
        // Check if cached result is still fresh (< 5 minutes old)
        if (result && Date.now() - result.timestamp < 5 * 60 * 1000) {
          resolve(result);
        } else {
          resolve(null);
        }
      };
      request.onerror = () => reject(request.error);
    });
  }
  // User interactions tracking
  async trackInteraction(
    interaction: Omit<UserInteraction, "id" | "timestamp">,
  ): Promise<void> {
    if (!this.db) await this.initDB();

    const fullInteraction: UserInteraction = {
      ...interaction,
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    const transaction = this.db!.transaction(["userInteractions"], "readwrite");
    const store = transaction.objectStore("userInteractions");

    await new Promise<void>((resolve, reject) => {
      const request = store.put(fullInteraction);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
  async getUserInteractions(
    type?: "search" | "view" | "edit" | "ai_query",
    limit = 50,
  ): Promise<UserInteraction[]> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["userInteractions"], "readonly");
    const store = transaction.objectStore("userInteractions");

    return new Promise((resolve, reject) => {
      const interactions: UserInteraction[] = [];
      let request: IDBRequest;

      if (type) {
        const index = store.index("type");
        request = index.openCursor(type, "prev"); // Most recent first
      } else {
        const index = store.index("timestamp");
        request = index.openCursor(null, "prev"); // Most recent first
      }
      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor && interactions.length < limit) {
          interactions.push(cursor.value);
          cursor.continue();
        } else {
          resolve(interactions);
        }
      };

      request.onerror = () => reject(request.error);
    });
  }
  // Analytics: get user activity patterns
  async getActivitySummary(): Promise<{
    totalInteractions: number;
    searchQueries: string[];
    mostViewedDocuments: Array<{ id: string; views: number }>;
    activityByHour: number[];
  }> {
    const interactions = await this.getUserInteractions(undefined, 1000);

    const searchQueries = interactions
      .filter((i) => i.type === "search" && i.query)
      .map((i) => i.query!)
      .slice(0, 20); // Last 20 searches

    const documentViews = new Map<string, number>();
    const hourlyActivity = new Array(24).fill(0);

    interactions.forEach((interaction) => {
      // Count document views
      if (interaction.documentId) {
        documentViews.set(
          interaction.documentId,
          (documentViews.get(interaction.documentId) || 0) + 1,
        );
      }
      // Activity by hour
      const hour = new Date(interaction.timestamp).getHours();
      hourlyActivity[hour]++;
    });

    const mostViewedDocuments = Array.from(documentViews.entries())
      .map(([id, views]) => ({ id, views }))
      .sort((a, b) => b.views - a.views)
      .slice(0, 10);

    return {
      totalInteractions: interactions.length,
      searchQueries: [...new Set(searchQueries)], // Remove duplicates
      mostViewedDocuments,
      activityByHour: hourlyActivity,
    };
  }
  // Sync status management
  async getPendingSyncDocuments(): Promise<CachedDocument[]> {
    if (!this.db) await this.initDB();

    const transaction = this.db!.transaction(["documents"], "readonly");
    const store = transaction.objectStore("documents");
    const index = store.index("syncStatus");

    return new Promise((resolve, reject) => {
      const request = index.getAll("pending");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  async markDocumentSynced(id: string): Promise<void> {
    const doc = await this.getDocument(id);
    if (doc) {
      doc.syncStatus = "synced";
      await this.cacheDocument(doc);
    }
  }
  // Cleanup old data
  async cleanup(olderThanDays = 30): Promise<void> {
    if (!this.db) await this.initDB();

    const cutoffTime = Date.now() - olderThanDays * 24 * 60 * 60 * 1000;

    // Clean old search results
    const searchTransaction = this.db!.transaction(
      ["searchResults"],
      "readwrite",
    );
    const searchStore = searchTransaction.objectStore("searchResults");
    const searchIndex = searchStore.index("timestamp");

    const searchRequest = searchIndex.openCursor(
      IDBKeyRange.upperBound(cutoffTime),
    );
    searchRequest.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest).result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };

    // Clean old interactions (keep more recent ones)
    const interactionsTransaction = this.db!.transaction(
      ["userInteractions"],
      "readwrite",
    );
    const interactionsStore =
      interactionsTransaction.objectStore("userInteractions");
    const interactionsIndex = interactionsStore.index("timestamp");

    const interactionsRequest = interactionsIndex.openCursor(
      IDBKeyRange.upperBound(cutoffTime),
    );
    interactionsRequest.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest).result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
  }
}
// Export singleton instance
export const indexedDBService = new IndexedDBService();

// Export types for use in components
export type { CachedDocument, SearchResult, UserInteraction };
