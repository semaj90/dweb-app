
/**
 * TypeScript Barrel Store Pattern Implementation
 * Provides centralized, type-safe state management with performance optimizations
 */

import {
  writable,
  derived,
  readable,
  type Writable,
  type Readable,
} from "svelte/store";
// TODO: Fix import - // Orphaned content: import { vscodeCache  // Base interfaces for the barrel pattern
export interface StoreMetadata {
  id: string;
  version: string;
  lastUpdated: number;
  dependencies: string[];
  cacheable: boolean;
  ttl?: number;
}

export interface BarrelStoreEntry<T = any> {
  store: Writable<T> | Readable<T>;
  metadata: StoreMetadata;
  validator?: (value: T) => boolean;
  serializer?: {
    serialize: (value: T) => string;
    deserialize: (value: string) => T;
  };
}

export interface StoreConfig<T> {
  id: string;
  initialValue: T;
  persistent?: boolean;
  cacheable?: boolean;
  ttl?: number;
  validator?: (value: T) => boolean;
  dependencies?: string[];
  computed?: (stores: Record<string, any>) => T;
}

/**
 * Main Barrel Store Manager
 * Implements the barrel pattern for centralized store management
 */
export class BarrelStoreManager {
  private stores = new Map<string, BarrelStoreEntry>();
  private computedStores = new Map<string, Readable<any>>();
  private subscriptions = new Map<string, (() => void)[]>();

  constructor() {
    this.initializePersistentStores();
  }

  /**
   * Create a new store with metadata and caching
   */
  createStore<T>(config: StoreConfig<T>): Writable<T> {
    const {
      id,
      initialValue,
      persistent = false,
      cacheable = false,
      ttl = 1000 * 60 * 15, // 15 minutes
      validator,
      dependencies = [],
    } = config;

    // Check if store already exists
    if (this.stores.has(id)) {
      console.warn(`Store ${id} already exists, returning existing store`);
      return this.stores.get(id)!.store as Writable<T>;
    }

    // Create the store
    const store = writable<T>(initialValue);

    // Set up persistence if enabled
    if (persistent) {
      this.setupPersistence(store, id);
    }

    // Set up caching if enabled
    if (cacheable) {
      this.setupCaching(store, id, ttl);
    }

    // Create metadata
    const metadata: StoreMetadata = {
      id,
      version: "1.0.0",
      lastUpdated: Date.now(),
      dependencies,
      cacheable,
      ttl: cacheable ? ttl : undefined,
    };

    // Store entry
    const entry: BarrelStoreEntry<T> = {
      store,
      metadata,
      validator,
      serializer: {
        serialize: (value: T) => JSON.stringify(value),
        deserialize: (value: string) => JSON.parse(value),
      },
    };

    this.stores.set(id, entry);

    // Set up validation
    if (validator) {
      store.subscribe((value) => {
        if (!validator(value)) {
          console.error(`Validation failed for store ${id}:`, value);
        }
      });
    }

    return store;
  }

  /**
   * Create a computed store that depends on other stores
   */
  createComputed<T>(
    id: string,
    dependencies: string[],
    computeFn: (values: any[]) => T,
    options?: { cacheable?: boolean; ttl?: number }
  ): Readable<T> {
    if (this.computedStores.has(id)) {
      return this.computedStores.get(id)!;
    }

    // Get dependency stores
    const depStores = dependencies.map((depId) => {
      const entry = this.stores.get(depId);
      if (!entry) {
        throw new Error(
          `Dependency store ${depId} not found for computed store ${id}`
        );
      }
      return entry.store;
    });

    // Create derived store
    const computedStore = derived(depStores, (values) => {
      return computeFn(values);
    });

    this.computedStores.set(id, computedStore);

    // Set up caching for computed stores if requested
    if (options?.cacheable) {
      this.setupCaching(computedStore, id, options.ttl);
    }

    return computedStore;
  }

  /**
   * Get store by ID with type safety
   */
  getStore<T>(id: string): Writable<T> | null {
    const entry = this.stores.get(id);
    return entry ? (entry.store as Writable<T>) : null;
  }

  /**
   * Get computed store by ID
   */
  getComputed<T>(id: string): Readable<T> | null {
    return (this.computedStores.get(id) as Readable<T>) || null;
  }

  /**
   * Get all stores (useful for debugging and introspection)
   */
  getAllStores(): Record<string, any> {
    const allStores: Record<string, any> = {};

    for (const [id, entry] of this.stores) {
      allStores[id] = entry.store;
    }

    for (const [id, store] of this.computedStores) {
      allStores[id] = store;
    }

    return allStores;
  }

  /**
   * Export store values for persistence or debugging
   */
  async exportStores(): Promise<Record<string, any>> {
    const exports: Record<string, any> = {};

    for (const [id, entry] of this.stores) {
      const currentValue = await this.getCurrentValue(entry.store);
      exports[id] = {
        value: currentValue,
        metadata: entry.metadata,
      };
    }

    return exports;
  }

  /**
   * Import store values from exported data
   */
  async importStores(data: Record<string, any>): Promise<void> {
    for (const [id, storeData] of Object.entries(data)) {
      const entry = this.stores.get(id);
      if (entry && "set" in entry.store) {
        (entry.store as Writable<any>).set(storeData.value);
      }
    }
  }

  /**
   * Clear all stores
   */
  clearAllStores(): void {
    for (const [id, entry] of this.stores) {
      if ("set" in entry.store) {
        (entry.store as Writable<any>).set(undefined);
      }
    }
  }

  /**
   * Subscribe to store changes with automatic cleanup
   */
  subscribeToStore<T>(
    storeId: string,
    callback: (value: T) => void,
    options?: { immediate?: boolean }
  ): () => void {
    const entry = this.stores.get(storeId);
    if (!entry) {
      throw new Error(`Store ${storeId} not found`);
    }

    const unsubscribe = entry.store.subscribe(callback);

    // Track subscription for cleanup
    if (!this.subscriptions.has(storeId)) {
      this.subscriptions.set(storeId, []);
    }
    this.subscriptions.get(storeId)!.push(unsubscribe);

    return unsubscribe;
  }

  /**
   * Get store metadata
   */
  getStoreMetadata(id: string): StoreMetadata | null {
    const entry = this.stores.get(id);
    return entry ? entry.metadata : null;
  }

  /**
   * Update store metadata
   */
  updateStoreMetadata(id: string, updates: Partial<StoreMetadata>): void {
    const entry = this.stores.get(id);
    if (entry) {
      entry.metadata = {
        ...entry.metadata,
        ...updates,
        lastUpdated: Date.now(),
      };
    }
  }

  /**
   * Cleanup all subscriptions and resources
   */
  dispose(): void {
    // Unsubscribe all subscriptions
    for (const unsubscribes of this.subscriptions.values()) {
      unsubscribes.forEach((unsub) => unsub());
    }

    this.stores.clear();
    this.computedStores.clear();
    this.subscriptions.clear();
  }

  // Private methods

  private async initializePersistentStores(): Promise<void> {
    try {
      const persistedData = await vscodeCache.get("barrel-stores-persistent");
      if (persistedData) {
        await this.importStores(persistedData);
      }
    } catch (error: any) {
      console.warn("Failed to initialize persistent stores:", error);
    }
  }

  private setupPersistence<T>(store: Writable<T>, id: string): void {
    store.subscribe(async (value) => {
      try {
        const allPersistent = await this.exportStores();
        await vscodeCache.set("barrel-stores-persistent", allPersistent);
      } catch (error: any) {
        console.error(`Failed to persist store ${id}:`, error);
      }
    });
  }

  private setupCaching<T>(
    store: Writable<T> | Readable<T>,
    id: string,
    ttl?: number
  ): void {
    store.subscribe(async (value) => {
      try {
        await vscodeCache.set(`store-cache-${id}`, value, { ttl });
      } catch (error: any) {
        console.error(`Failed to cache store ${id}:`, error);
      }
    });
  }

  private async getCurrentValue<T>(
    store: Writable<T> | Readable<T>
  ): Promise<T> {
    return new Promise((resolve) => {
      const unsubscribe = store.subscribe((value) => {
        resolve(value);
        unsubscribe();
      });
    });
  }
}

// Global barrel store manager
export const barrelStore = new BarrelStoreManager();
;
// Legal AI specific stores
export const legalAIStores = {
  // Case management
  currentCase: barrelStore.createStore({
    id: "legal-ai-current-case",
    initialValue: null,
    persistent: true,
    cacheable: true,
  }),

  // Document processing
  documentQueue: barrelStore.createStore({
    id: "legal-ai-document-queue",
    initialValue: [],
    persistent: true,
    cacheable: true,
  }),

  // AI analysis results
  analysisResults: barrelStore.createStore({
    id: "legal-ai-analysis-results",
    initialValue: {},
    persistent: true,
    cacheable: true,
    ttl: 1000 * 60 * 60, // 1 hour
  }),

  // User preferences
  userPreferences: barrelStore.createStore({
    id: "legal-ai-user-preferences",
    initialValue: {
      theme: "dark",
      aiModel: "gemma3-legal",
      autoSave: true,
      notifications: true,
    },
    persistent: true,
    validator: (prefs) => {
      return (
        typeof prefs === "object" &&
        prefs !== null &&
        typeof prefs.theme === "string"
      );
    },
  }),

  // Application state
  appState: barrelStore.createStore({
    id: "legal-ai-app-state",
    initialValue: {
      loading: false,
      error: null,
      currentStep: 0,
      totalSteps: 5,
    },
    cacheable: true,
  }),

  // OCR processing state
  ocrState: barrelStore.createStore({
    id: "legal-ai-ocr-state",
    initialValue: {
      processing: false,
      progress: 0,
      results: [],
      errors: [],
    },
    cacheable: true,
  }),
};

// Computed stores for complex derived state
export const legalAIComputed = {
  // Case completion percentage
  caseProgress: barrelStore.createComputed(
    "legal-ai-case-progress",
    ["legal-ai-current-case", "legal-ai-app-state"],
    ([currentCase, appState]) => {
      if (!currentCase) return 0;
      return (appState.currentStep / appState.totalSteps) * 100;
    },
    { cacheable: true }
  ),

  // Document processing status
  documentStatus: barrelStore.createComputed(
    "legal-ai-document-status",
    ["legal-ai-document-queue", "legal-ai-ocr-state"],
    ([queue, ocrState]) => {
      const total = queue.length;
      const processed = queue.filter((doc: any) => doc.processed).length;
      const processing = ocrState.processing;

      return {
        total,
        processed,
        processing,
        remaining: total - processed,
        progress: total > 0 ? (processed / total) * 100 : 0,
      };
    }
  ),

  // Application readiness
  appReady: barrelStore.createComputed(
    "legal-ai-app-ready",
    ["legal-ai-user-preferences", "legal-ai-app-state"],
    ([preferences, appState]) => {
      return !appState.loading && !appState.error && preferences !== null;
    }
  ),
};

// Utility functions for store management
export const storeUtils = {
  /**
   * Reset all legal AI stores to initial state
   */
  resetAll(): void {
    Object.values(legalAIStores).forEach((store) => {
      if ("set" in store) {
        store.set(undefined);
      }
    });
  },

  /**
   * Export all store values for backup
   */
  async backup(): Promise<string> {
    const data = await barrelStore.exportStores();
    return JSON.stringify(data, null, 2);
  },

  /**
   * Import store values from backup
   */
  async restore(backupData: string): Promise<void> {
    try {
      const data = JSON.parse(backupData);
      await barrelStore.importStores(data);
    } catch (error: any) {
      console.error("Failed to restore from backup:", error);
      throw error;
    }
  },

  /**
   * Get all store values for debugging
   */
  async debug(): Promise<Record<string, any>> {
    return await barrelStore.exportStores();
  },
};

// Export barrel store instance for advanced usage
export { barrelStore as default };
