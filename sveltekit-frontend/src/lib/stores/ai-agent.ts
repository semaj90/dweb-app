import type { ChatMessage } from "$lib/types/api";
import crypto from "crypto";
import stream from "stream";

import { writable, derived } from "svelte/store";
import { realAIService } from "$lib/services/real-ai-service";

// ======================================================================
// ENHANCED AI AGENT STORE WITH PRODUCTION FEATURES
// Integrates with local LLMs, vector search, and real-time capabilities
// ======================================================================

export interface AIAgentState {
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

export interface ProcessingJob {
  id: string;
  type: "chat" | "summarize" | "analyze" | "embed" | "search";
  status: "pending" | "processing" | "completed" | "failed";
  input: any;
  output?: unknown;
  startTime: Date;
  endTime?: Date;
  error?: string;
  retryCount: number;
}

export interface SimilarDocument {
  id: string;
  title: string;
  content: string;
  similarity: number;
  metadata: Record<string, any>;
}

export interface CitationSource {
  id: string;
  title: string;
  url?: string;
  content: string;
  relevance: number;
  type: "document" | "case" | "statute" | "evidence";
}

export interface AIError {
  id: string;
  type: "connection" | "processing" | "timeout" | "model" | "rate_limit";
  message: string;
  timestamp: Date;
  context?: unknown;
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
        // Use real AI service for connection
        const connectionResult = await realAIService.connect(modelName || "gemma3-legal");

        if (!connectionResult.success) {
          throw new Error(connectionResult.error || "Connection failed");
        }

        update((state) => ({
          ...state,
          isConnected: true,
          isProcessing: false,
          currentModel: connectionResult.model,
          availableModels: connectionResult.availableModels,
          lastHeartbeat: new Date(),
          systemHealth: "healthy",
        }));

        // Start heartbeat
        this.startHeartbeat();
      } catch (error: any) {
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
    async sendMessage(message: string, context?: unknown) {
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
        // Use real AI service for chat
        const chatResponse = await realAIService.sendMessage({
          message,
          sessionId,
          context: {
            conversationHistory: [],
            ...context,
          },
          options: {
            stream: true,
            useRAG: true
          }
        });

        // Complete the chat response
        this.completeStreamingResponse(chatResponse.response, chatResponse, jobId);

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
      } catch (error: any) {
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
              } catch (e: any) {
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
        // Use real AI service for document search
        const documents = await realAIService.searchSimilarDocuments(query, {
          limit,
          threshold: 0.7
        });

        update((state) => ({
          ...state,
          similarDocuments: documents,
        }));

        return documents;
      } catch (error: any) {
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
      metadata?: unknown;
    }) {
      try {
        // Use real AI service for document indexing
        const result = await realAIService.indexDocument({
          title: document.title,
          content: document.content,
          metadata: document.metadata as Record<string, any>
        });

        if (!result.success) {
          throw new Error(result.error || "Indexing failed");
        }

        update((state) => ({
          ...state,
          vectorStore: {
            ...state.vectorStore,
            documentCount: state.vectorStore.documentCount + 1,
            lastIndexUpdate: new Date(),
            isIndexed: true,
          },
        }));

        return { success: true };
      } catch (error: any) {
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
        // Use real AI service for model switching
        const result = await realAIService.switchModel(modelName);

        if (!result.success) {
          throw new Error(result.error || "Model switch failed");
        }

        update((state) => ({
          ...state,
          currentModel: modelName,
          isProcessing: false,
          currentConversation: [], // Clear conversation on model switch
        }));
      } catch (error: any) {
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
          // Use real AI service for health checks
          const health = await realAIService.healthCheck();

          update((state) => ({
            ...state,
            lastHeartbeat: new Date(),
            systemHealth: health.overall ? "healthy" : "critical",
            availableModels: health.models.map(m => m.name),
            isConnected: health.overall,
          }));
        } catch (error: any) {
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
;
// Derived stores for easy component access (fixed corruption)
export const isAIConnected = derived(aiAgentStore, (state) => state.isConnected);
export const currentConversation = derived(aiAgentStore, (state) => state.currentConversation);
export const isProcessing = derived(aiAgentStore, (state) => state.isProcessing);
export const streamingResponse = derived(aiAgentStore, (state) => state.streamingResponse);
export const similarDocuments = derived(aiAgentStore, (state) => state.similarDocuments);
export const aiErrors = derived(aiAgentStore, (state) => state.errors.filter((e: any) => !e.resolved));
export const systemHealth = derived(aiAgentStore, (state) => state.systemHealth);
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
