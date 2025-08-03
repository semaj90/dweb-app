import { writable, derived } from "svelte/store";
import type { Database } from "$lib/types";
import type { ChatMessage } from "$lib/types/api";

// ======================================================================
// ENHANCED AI AGENT STORE WITH PRODUCTION FEATURES
// Integrates with local LLMs, vector search, and real-time capabilities
// ======================================================================

interface AIAgentState {
  // Connection & Health
  isConnected: boolean;
  isProcessing: boolean;
  systemHealth: "healthy" | "degraded" | "critical";
  lastHeartbeat: Date | null;

  // Chat & Conversation
  currentConversation: ChatMessage[];
  conversationHistory: ChatMessage[][];
  activeSessionId: string | null;

  // AI Processing
  currentModel: string;
  availableModels: string[];
  processingQueue: ProcessingJob[];
  completedJobs: ProcessingJob[];

  // RAG & Knowledge
  vectorStore: {
    isIndexed: boolean;
    documentCount: number;
    lastIndexUpdate: Date | null;
  };
  similarDocuments: SimilarDocument[];
  citationSources: CitationSource[];

  // Real-time Features
  streamingResponse: string;
  isStreaming: boolean;
  typingIndicator: boolean;

  // Error Handling
  errors: AIError[];
  retryQueue: string[];

  // Performance Metrics
  responseTimeMs: number;
  averageResponseTime: number;
  totalRequests: number;
  successRate: number;
}

interface ProcessingJob {
  id: string;
  type: "chat" | "summarize" | "analyze" | "embed" | "search";
  status: "pending" | "processing" | "completed" | "failed";
  input: any;
  output?: any;
  startTime: Date;
  endTime?: Date;
  error?: string;
  retryCount: number;
}

interface SimilarDocument {
  id: string;
  title: string;
  content: string;
  similarity: number;
  metadata: Record<string, any>;
}

interface CitationSource {
  id: string;
  title: string;
  url?: string;
  content: string;
  relevance: number;
  type: "document" | "case" | "statute" | "evidence";
}

interface AIError {
  id: string;
  type: "connection" | "processing" | "timeout" | "model" | "rate_limit";
  message: string;
  timestamp: Date;
  context?: any;
  resolved: boolean;
  retryable: boolean;
}

// Main AI Agent Store
const createAIAgentStore = () => {
  const { subscribe, set, update } = writable<AIAgentState>({
    isConnected: false,
    isProcessing: false,
    systemHealth: "healthy",
    lastHeartbeat: null,
    currentConversation: [],
    conversationHistory: [],
    activeSessionId: null,
    currentModel: "gemma3-legal",
    availableModels: ["gemma3-legal", "mistral-7b", "llama3.1-8b"],
    processingQueue: [],
    completedJobs: [],
    vectorStore: {
      isIndexed: false,
      documentCount: 0,
      lastIndexUpdate: null,
    },
    similarDocuments: [],
    citationSources: [],
    streamingResponse: "",
    isStreaming: false,
    typingIndicator: false,
    errors: [],
    retryQueue: [],
    responseTimeMs: 0,
    averageResponseTime: 0,
    totalRequests: 0,
    successRate: 100,
  });

  return {
    subscribe,

    // Connection Management
    async connect(modelName?: string) {
      update((state) => ({ ...state, isProcessing: true }));

      try {
        const response = await fetch("/api/ai/connect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: modelName || "gemma3-legal" }),
        });

        if (!response.ok) throw new Error("Connection failed");

        const data = await response.json();

        update((state) => ({
          ...state,
          isConnected: true,
          isProcessing: false,
          currentModel: data.model,
          availableModels: data.availableModels || state.availableModels,
          lastHeartbeat: new Date(),
          systemHealth: "healthy",
        }));

        // Start heartbeat
        this.startHeartbeat();
      } catch (error) {
        this.addError({
          type: "connection",
          message: (error as Error).message,
          retryable: true,
        });

        update((state) => ({
          ...state,
          isConnected: false,
          isProcessing: false,
          systemHealth: "critical",
        }));
      }
    },

    disconnect() {
      update((state) => ({
        ...state,
        isConnected: false,
        systemHealth: "degraded",
        currentConversation: [],
        activeSessionId: null,
      }));
    },

    // Chat Functions
    async sendMessage(message: string, context?: any) {
      const startTime = Date.now();
      const jobId = crypto.randomUUID();
      const sessionId = crypto.randomUUID();

      // Add user message
      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        content: message,
        role: "user",
        timestamp: new Date(),
      };

      update((state) => ({
        ...state,
        currentConversation: [...state.currentConversation, userMessage],
        activeSessionId: sessionId,
        isProcessing: true,
        typingIndicator: true,
      }));

      // Add processing job
      const job: ProcessingJob = {
        id: jobId,
        type: "chat",
        status: "pending",
        input: { message, context, sessionId },
        startTime: new Date(),
        retryCount: 0,
      };

      update((state) => ({
        ...state,
        processingQueue: [...state.processingQueue, job],
      }));

      try {
        // Check if streaming is supported
        const response = await fetch("/api/ai/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            sessionId,
            context: {
              conversationHistory: [],
              ...context,
            },
            stream: true,
          }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        // Handle streaming response
        if (response.body) {
          await this.handleStreamingResponse(response.body, jobId);
        } else {
          const data = await response.json();
          this.completeJob(jobId, data);
        }

        const responseTime = Date.now() - startTime;

        update((state) => ({
          ...state,
          responseTimeMs: responseTime,
          averageResponseTime:
            (state.averageResponseTime * state.totalRequests + responseTime) /
            (state.totalRequests + 1),
          totalRequests: state.totalRequests + 1,
          isProcessing: false,
          typingIndicator: false,
        }));
      } catch (error) {
        this.addError({
          type: "processing",
          message: (error as Error).message,
          context: { jobId, message: message.substring(0, 100) },
          retryable: true,
        });

        update((state) => ({
          ...state,
          isProcessing: false,
          typingIndicator: false,
          successRate:
            state.totalRequests > 0
              ? (((state.totalRequests * state.successRate) / 100 - 1) /
                  state.totalRequests) *
                100
              : 0,
        }));

        this.failJob(jobId, (error as Error).message);
      }
    },

    // Streaming Response Handler
    async handleStreamingResponse(stream: ReadableStream, jobId: string) {
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = "";

      update((state) => ({
        ...state,
        isStreaming: true,
        streamingResponse: "",
      }));

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.content) {
                  assistantMessage += data.content;

                  update((state) => ({
                    ...state,
                    streamingResponse: assistantMessage,
                  }));
                }

                if (data.done) {
                  this.completeStreamingResponse(assistantMessage, data, jobId);
                  return;
                }
              } catch (e) {
                console.warn("Failed to parse streaming data:", line);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
        update((state) => ({
          ...state,
          isStreaming: false,
          streamingResponse: "",
        }));
      }
    },

    completeStreamingResponse(content: string, data: any, jobId: string) {
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        content,
        role: "assistant",
        timestamp: new Date(),
        sources: data.sources || [],
        metadata: {
          model: data.model,
          confidence: data.confidence,
          executionTime: data.executionTime,
          fromCache: data.fromCache || false,
        },
      };

      update((state) => ({
        ...state,
        currentConversation: [...state.currentConversation, assistantMessage],
        similarDocuments: data.sources || [],
        citationSources: data.citations || [],
      }));

      this.completeJob(jobId, { message: assistantMessage });
    },

    // RAG Functions
    async searchSimilarDocuments(query: string, limit = 5) {
      try {
        const response = await fetch("/api/rag/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, limit, type: "semantic" }),
        });

        if (!response.ok) throw new Error("Search failed");

        const results = await response.json();

        update((state) => ({
          ...state,
          similarDocuments: results.documents || [],
        }));

        return results.documents;
      } catch (error) {
        this.addError({
          type: "processing",
          message: `Search failed: ${(error as Error).message}`,
          retryable: true,
        });
        return [];
      }
    },

    async indexDocument(document: {
      title: string;
      content: string;
      metadata?: any;
    }) {
      try {
        const response = await fetch("/api/rag/index", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(document),
        });

        if (!response.ok) throw new Error("Indexing failed");

        const result = await response.json();

        update((state) => ({
          ...state,
          vectorStore: {
            ...state.vectorStore,
            documentCount:
              result.totalDocuments || state.vectorStore.documentCount + 1,
            lastIndexUpdate: new Date(),
            isIndexed: true,
          },
        }));

        return result;
      } catch (error) {
        this.addError({
          type: "processing",
          message: `Indexing failed: ${(error as Error).message}`,
          retryable: true,
        });
        throw error;
      }
    },

    // Model Management
    async switchModel(modelName: string) {
      update((state) => ({ ...state, isProcessing: true }));

      try {
        const response = await fetch("/api/ai/model/switch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: modelName }),
        });

        if (!response.ok) throw new Error("Model switch failed");

        update((state) => ({
          ...state,
          currentModel: modelName,
          isProcessing: false,
          currentConversation: [], // Clear conversation on model switch
        }));
      } catch (error) {
        this.addError({
          type: "model",
          message: (error as Error).message,
          retryable: true,
        });

        update((state) => ({ ...state, isProcessing: false }));
      }
    },

    // Conversation Management
    clearConversation() {
      update((state) => ({
        ...state,
        currentConversation: [],
        activeSessionId: null,
        streamingResponse: "",
        similarDocuments: [],
        citationSources: [],
      }));
    },

    saveConversation() {
      update((state) => ({
        ...state,
        conversationHistory: [
          ...state.conversationHistory,
          [...state.currentConversation],
        ],
        currentConversation: [],
        activeSessionId: null,
      }));
    },

    loadConversation(index: number) {
      update((state) => {
        if (index >= 0 && index < state.conversationHistory.length) {
          return {
            ...state,
            currentConversation: [...state.conversationHistory[index]],
            activeSessionId: crypto.randomUUID(),
          };
        }
        return state;
      });
    },

    // Error Handling
    addError(error: Omit<AIError, "id" | "timestamp" | "resolved">) {
      const newError: AIError = {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        resolved: false,
        ...error,
      };

      update((state) => ({
        ...state,
        errors: [...state.errors, newError],
        systemHealth:
          state.systemHealth === "healthy" ? "degraded" : state.systemHealth,
      }));
    },

    resolveError(errorId: string) {
      update((state) => ({
        ...state,
        errors: state.errors.map((error) =>
          error.id === errorId ? { ...error, resolved: true } : error
        ),
      }));
    },

    clearErrors() {
      update((state) => ({
        ...state,
        errors: [],
        systemHealth: state.isConnected ? "healthy" : "degraded",
      }));
    },

    // Job Management
    completeJob(jobId: string, result: any) {
      update((state) => {
        const job = state.processingQueue.find((j) => j.id === jobId);
        if (!job) return state;

        const completedJob: ProcessingJob = {
          ...job,
          status: "completed",
          output: result,
          endTime: new Date(),
        };

        return {
          ...state,
          processingQueue: state.processingQueue.filter((j) => j.id !== jobId),
          completedJobs: [...state.completedJobs, completedJob],
        };
      });
    },

    failJob(jobId: string, error: string) {
      update((state) => {
        const job = state.processingQueue.find((j) => j.id === jobId);
        if (!job) return state;

        const failedJob: ProcessingJob = {
          ...job,
          status: "failed",
          error,
          endTime: new Date(),
        };

        return {
          ...state,
          processingQueue: state.processingQueue.filter((j) => j.id !== jobId),
          completedJobs: [...state.completedJobs, failedJob],
        };
      });
    },

    // Health Monitoring
    startHeartbeat() {
      const interval = setInterval(async () => {
        try {
          const response = await fetch("/api/ai/health");
          const health = await response.json();

          update((state) => ({
            ...state,
            lastHeartbeat: new Date(),
            systemHealth: health.status,
            availableModels: health.models || state.availableModels,
          }));
        } catch (error) {
          update((state) => ({
            ...state,
            systemHealth: "critical",
            isConnected: false,
          }));
        }
      }, 30000); // Every 30 seconds

      // Store interval ID for cleanup
      return interval;
    },
  };
};

// Export the store
export const aiAgentStore = createAIAgentStore();

// Derived stores for easy component access
export const isAIConnected = derived(
  aiAgentStore,
  (state) => state.isConnected
);
export const currentConversation = derived(
  aiAgentStore,
  (state) => state.currentConversation
);
export const isProcessing = derived(
  aiAgentStore,
  (state) => state.isProcessing
);
export const streamingResponse = derived(
  aiAgentStore,
  (state) => state.streamingResponse
);
export const similarDocuments = derived(
  aiAgentStore,
  (state) => state.similarDocuments
);
export const aiErrors = derived(aiAgentStore, (state) =>
  state.errors.filter((e) => !e.resolved)
);
export const systemHealth = derived(
  aiAgentStore,
  (state) => state.systemHealth
);
export const performanceMetrics = derived(aiAgentStore, (state) => ({
  responseTime: state.responseTimeMs,
  averageResponseTime: state.averageResponseTime,
  totalRequests: state.totalRequests,
  successRate: state.successRate,
}));

// Auto-connect on store initialization
if (typeof window !== "undefined") {
  aiAgentStore.connect().catch(console.error);
}
