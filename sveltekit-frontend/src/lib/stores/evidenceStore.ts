import crypto from "crypto";

// Real-time evidence store with WebSocket/SSE integration and local undo
import { writable, derived, get } from "svelte/store";

export interface Evidence {
  id: string;
  title: string;
  description: string;
  type: string;
  caseId: string;
  fileUrl?: string;
  metadata?: Record<string, any>;
  tags?: string[];
  location?: {
    latitude: number;
    longitude: number;
    address?: string;
  };
  classification?: {
    category: string;
    relevance: number;
    confidence: number;
  };
  timeline?: {
    createdAt: string;
    updatedAt: string;
    collectedAt?: string;
  };
  analysis?: {
    summary: string;
    keyPoints: string[];
    relevance: number;
    admissibility: "admissible" | "questionable" | "inadmissible";
    reasoning: string;
    suggestedTags: string[];
    // Additional legacy properties for compatibility
    aiSummary?: string;
    vectorSimilarity?: number;
    relatedEvidence?: string[];
  };
}
export interface EvidenceOperation {
  id: string;
  type: "CREATE" | "UPDATE" | "DELETE";
  timestamp: string;
  userId?: string;
  evidenceId: string;
  previousState?: Evidence | null;
  newState?: Evidence | null;
  changes?: Partial<Evidence>;
}
class RealTimeEvidenceStore {
  // Core stores
  public evidence = writable<Evidence[]>([]);
  public isLoading = writable(false);
  public error = writable<string | null>(null);
  public isConnected = writable(false);

  // Undo/Redo functionality
  private operationHistory = writable<EvidenceOperation[]>([]);
  private currentHistoryIndex = writable(-1);

  // Connection management
  private websocket: WebSocket | null = null;
  private eventSource: EventSource | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  // Local cache and sync
  private localCache: Map<string, Evidence> = new Map();
  private pendingOperations: EvidenceOperation[] = [];

  constructor() {
    if (browser) {
      this.initializeConnection();
      this.loadFromLocalStorage();
    }
  }
  // Connection Management
  private async initializeConnection() {
    try {
      // Try WebSocket first
      await this.connectWebSocket();
    } catch (error: any) {
      console.warn("WebSocket failed, falling back to SSE:", error);
      this.connectSSE();
    }
  }
  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.hostname}:3030`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
          console.log("✅ WebSocket connected");
          this.isConnected.set(true);
          this.reconnectAttempts = 0;

          // Subscribe to evidence updates
          this.websocket?.send(
            JSON.stringify({
              type: "subscribe",
              channels: ["evidence_update", "case_update"],
            }),
          );

          resolve();
        };

        this.websocket.onmessage = (event: any) => {
          try {
            const message = JSON.parse(event.data);
            this.handleRealtimeUpdate(message);
          } catch (error: any) {
            console.error("WebSocket message parse error:", error);
          }
        };

        this.websocket.onclose = () => {
          console.log("WebSocket disconnected");
          this.isConnected.set(false);
          this.websocket = null;

          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            setTimeout(() => {
              this.reconnectAttempts++;
              this.connectWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
          } else {
            console.warn(
              "Max WebSocket reconnect attempts reached, falling back to SSE",
            );
            this.connectSSE();
          }
        };

        this.websocket.onerror = (error) => {
          console.error("WebSocket error:", error);
          reject(error);
        };
      } catch (error: any) {
        reject(error);
      }
    });
  }
  private connectSSE() {
    try {
      const userId = this.getCurrentUserId();
      const sseUrl = `/api/updates?userId=${userId}&subscriptions=evidence_update,case_update`;

      this.eventSource = new EventSource(sseUrl);

      this.eventSource.onopen = () => {
        console.log("✅ SSE connected");
        this.isConnected.set(true);
        this.reconnectAttempts = 0;
      };

      this.eventSource.onmessage = (event: any) => {
        try {
          const message = JSON.parse(event.data);
          this.handleRealtimeUpdate(message);
        } catch (error: any) {
          console.error("SSE message parse error:", error);
        }
      };

      this.eventSource.onerror = () => {
        console.log("SSE disconnected");
        this.isConnected.set(false);

        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => {
            this.reconnectAttempts++;
            this.connectSSE();
          }, this.reconnectDelay * this.reconnectAttempts);
        }
      };
    } catch (error: any) {
      console.error("SSE connection failed:", error);
    }
  }
  // Real-time update handling
  private handleRealtimeUpdate(message: any) {
    if (message.channel === "evidence_update") {
      const { type, evidenceId, data, changes, userId } = message.data;

      switch (type) {
        case "EVIDENCE_CREATED":
          this.handleEvidenceCreated(data, userId);
          break;
        case "EVIDENCE_UPDATED":
          this.handleEvidenceUpdated(evidenceId, changes, userId);
          break;
        case "EVIDENCE_DELETED":
          this.handleEvidenceDeleted(evidenceId, userId);
          break;
      }
    }
  }
  private handleEvidenceCreated(evidenceData: Evidence, userId?: string) {
    this.evidence.update((items) => {
      // Check if evidence already exists (avoid duplicates)
      if (items.find((item) => item.id === evidenceData.id)) {
        return items;
      }
      // Add to local cache
      this.localCache.set(evidenceData.id, evidenceData);

      // Add operation to history
      this.addToHistory({
        id: crypto.randomUUID(),
        type: "CREATE",
        timestamp: new Date().toISOString(),
        userId,
        evidenceId: evidenceData.id,
        previousState: null,
        newState: evidenceData,
      });

      return [...items, evidenceData];
    });

    this.saveToLocalStorage();
  }
  private handleEvidenceUpdated(
    evidenceId: string,
    changes: Partial<Evidence>,
    userId?: string,
  ) {
    this.evidence.update((items) => {
      const index = items.findIndex((item) => item.id === evidenceId);
      if (index === -1) return items;

      const previousState = { ...items[index] };
      const newState = { ...items[index], ...changes };

      // Update local cache
      this.localCache.set(evidenceId, newState);

      // Add operation to history
      this.addToHistory({
        id: crypto.randomUUID(),
        type: "UPDATE",
        timestamp: new Date().toISOString(),
        userId,
        evidenceId,
        previousState,
        newState,
        changes,
      });

      items[index] = newState;
      return [...items];
    });

    this.saveToLocalStorage();
  }
  private handleEvidenceDeleted(evidenceId: string, userId?: string) {
    this.evidence.update((items) => {
      const index = items.findIndex((item) => item.id === evidenceId);
      if (index === -1) return items;

      const previousState = items[index];

      // Remove from local cache
      this.localCache.delete(evidenceId);

      // Add operation to history
      this.addToHistory({
        id: crypto.randomUUID(),
        type: "DELETE",
        timestamp: new Date().toISOString(),
        userId,
        evidenceId,
        previousState,
        newState: null,
      });

      return items.filter((item) => item.id !== evidenceId);
    });

    this.saveToLocalStorage();
  }
  // CRUD Operations with optimistic updates
  public async createEvidence(
    evidenceData: Omit<Evidence, "id">,
  ): Promise<string> {
    const evidenceId = crypto.randomUUID();
    const newEvidence: Evidence = {
      ...evidenceData,
      id: evidenceId,
      timeline: {
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        ...evidenceData.timeline,
      },
    };

    // Optimistic update
    this.handleEvidenceCreated(newEvidence, this.getCurrentUserId());

    try {
      // Send to server
      const response = await fetch("/api/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newEvidence),
      });

      if (!response.ok) {
        throw new Error(`Failed to create evidence: ${response.statusText}`);
      }
      const result = await response.json();
      return result.id || evidenceId;
    } catch (error: any) {
      // Revert optimistic update on error
      this.handleEvidenceDeleted(evidenceId);
      throw error;
    }
  }
  public async updateEvidence(
    evidenceId: string,
    changes: Partial<Evidence>,
  ): Promise<void> {
    const currentEvidence = get(this.evidence).find(
      (item) => item.id === evidenceId,
    );
    if (!currentEvidence) {
      throw new Error(`Evidence ${evidenceId} not found`);
    }
    // Optimistic update
    this.handleEvidenceUpdated(evidenceId, changes, this.getCurrentUserId());

    try {
      // Send to server
      const response = await fetch(`/api/evidence/${evidenceId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });

      if (!response.ok) {
        throw new Error(`Failed to update evidence: ${response.statusText}`);
      }
    } catch (error: any) {
      // Revert optimistic update on error
      this.handleEvidenceUpdated(
        evidenceId,
        currentEvidence,
        this.getCurrentUserId(),
      );
      throw error;
    }
  }
  public async deleteEvidence(evidenceId: string): Promise<void> {
    const currentEvidence = get(this.evidence).find(
      (item) => item.id === evidenceId,
    );
    if (!currentEvidence) {
      throw new Error(`Evidence ${evidenceId} not found`);
    }
    // Optimistic update
    this.handleEvidenceDeleted(evidenceId, this.getCurrentUserId());

    try {
      // Send to server
      const response = await fetch(`/api/evidence/${evidenceId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`Failed to delete evidence: ${response.statusText}`);
      }
    } catch (error: any) {
      // Revert optimistic update on error
      this.handleEvidenceCreated(currentEvidence, this.getCurrentUserId());
      throw error;
    }
  }
  // Undo/Redo functionality
  private addToHistory(operation: EvidenceOperation) {
    this.operationHistory.update((history) => {
      const currentIndex = get(this.currentHistoryIndex);

      // Remove any operations after current index (when undoing then doing new operation)
      if (currentIndex < history.length - 1) {
        history = history.slice(0, currentIndex + 1);
      }
      history.push(operation);

      // Limit history size (keep last 100 operations)
      if (history.length > 100) {
        history = history.slice(-100);
      }
      return history;
    });

    this.currentHistoryIndex.update(
      (index) => get(this.operationHistory).length - 1,
    );
  }
  public canUndo(): boolean {
    return get(this.currentHistoryIndex) >= 0;
  }
  public canRedo(): boolean {
    const history = get(this.operationHistory);
    const currentIndex = get(this.currentHistoryIndex);
    return currentIndex < history.length - 1;
  }
  public undo(): boolean {
    if (!this.canUndo()) return false;

    const history = get(this.operationHistory);
    const currentIndex = get(this.currentHistoryIndex);
    const operation = history[currentIndex];

    // Reverse the operation
    switch (operation.type) {
      case "CREATE":
        if (operation.newState) {
          this.evidence.update((items) =>
            items.filter((item) => item.id !== operation.evidenceId),
          );
        }
        break;
      case "UPDATE":
        if (operation.previousState) {
          this.evidence.update((items) => {
            const index = items.findIndex(
              (item) => item.id === operation.evidenceId,
            );
            if (index !== -1) {
              items[index] = operation.previousState!;
            }
            return [...items];
          });
        }
        break;
      case "DELETE":
        if (operation.previousState) {
          this.evidence.update((items) => [...items, operation.previousState!]);
        }
        break;
    }
    this.currentHistoryIndex.update((index) => index - 1);
    this.saveToLocalStorage();
    return true;
  }
  public redo(): boolean {
    if (!this.canRedo()) return false;

    const history = get(this.operationHistory);
    const currentIndex = get(this.currentHistoryIndex);
    const operation = history[currentIndex + 1];

    // Replay the operation
    switch (operation.type) {
      case "CREATE":
        if (operation.newState) {
          this.evidence.update((items) => [...items, operation.newState!]);
        }
        break;
      case "UPDATE":
        if (operation.newState) {
          this.evidence.update((items) => {
            const index = items.findIndex(
              (item) => item.id === operation.evidenceId,
            );
            if (index !== -1) {
              items[index] = operation.newState!;
            }
            return [...items];
          });
        }
        break;
      case "DELETE":
        this.evidence.update((items) =>
          items.filter((item) => item.id !== operation.evidenceId),
        );
        break;
    }
    this.currentHistoryIndex.update((index) => index + 1);
    this.saveToLocalStorage();
    return true;
  }
  // Local storage persistence
  private saveToLocalStorage() {
    if (!browser) return;

    try {
      const data = {
        evidence: get(this.evidence),
        operationHistory: get(this.operationHistory),
        currentHistoryIndex: get(this.currentHistoryIndex),
        lastUpdated: new Date().toISOString(),
      };

      localStorage.setItem("evidenceStore", JSON.stringify(data));
    } catch (error: any) {
      console.error("Failed to save to localStorage:", error);
    }
  }
  private loadFromLocalStorage() {
    if (!browser) return;

    try {
      const stored = localStorage.getItem("evidenceStore");
      if (stored) {
        const data = JSON.parse(stored);

        // Check if data is not too old (24 hours)
        const lastUpdated = new Date(data.lastUpdated);
        const now = new Date();
        const hoursDiff =
          (now.getTime() - lastUpdated.getTime()) / (1000 * 60 * 60);

        if (hoursDiff < 24) {
          this.evidence.set(data.evidence || []);
          this.operationHistory.set(data.operationHistory || []);
          this.currentHistoryIndex.set(data.currentHistoryIndex || -1);

          // Rebuild local cache
          data.evidence?.forEach((item: Evidence) => {
            this.localCache.set(item.id, item);
          });
        }
      }
    } catch (error: any) {
      console.error("Failed to load from localStorage:", error);
    }
  }
  // Utility methods
  private getCurrentUserId(): string {
    // In a real app, get from auth store or session
    return "current-user-id";
  }
  public disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.isConnected.set(false);
  }
  // Derived stores for convenience
  public evidenceById = derived(this.evidence, (items) => {
    const map = new Map<string, Evidence>();
    items.forEach((item) => map.set(item.id, item));
    return map;
  });

  public evidenceByCase = derived(this.evidence, (items) => {
    const map = new Map<string, Evidence[]>();
    items.forEach((item) => {
      if (!map.has(item.caseId)) {
        map.set(item.caseId, []);
      }
      map.get(item.caseId)!.push(item);
    });
    return map;
  });
}
// Export singleton instance
export const evidenceStore = new RealTimeEvidenceStore();
;