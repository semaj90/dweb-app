/**
 * Svelte 5 Compatible Chat Store with XState-like Interface
 * Enhanced for Gemma3 Legal AI Integration
 * Optimized for legal document analysis and precedent search
 */

import { writable, derived, readonly } from "svelte/store";
import crypto from "crypto";

// === TYPE DEFINITIONS ===
export interface ChatMessage {
  id: string;
  content: string;
  role: "user" | "assistant" | "system";
  timestamp: Date;
  conversationId?: string;
  metadata?: {
    model?: string;
    tokensUsed?: number;
    references?: string[];
    confidence?: number;
    legalContext?: unknown;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  created: Date;
  updated: Date;
  metadata?: {
    caseType?: string;
    jurisdiction?: string;
    precedents?: string[];
  };
}

export interface ChatSettings {
  model: string;
  temperature: number;
  maxTokens: number;
  streaming: boolean;
  contextWindow: number;
  proactiveMode: boolean;
  emotionalMode: boolean;
  legalMode?: boolean;
  citationMode?: boolean;
}

export interface ServiceStatus {
  ollama: "unknown" | "loading" | "connected" | "error";
  qdrant: "unknown" | "loading" | "connected" | "error";
  database: "unknown" | "loading" | "connected" | "error";
  gemma3: "unknown" | "loading" | "ready" | "error";
}

// === CHAT STATE INTERFACE ===
export interface ChatContext {
  messages: ChatMessage[];
  conversations: Conversation[];
  currentConversation: Conversation | null;
  error: Error | null;
  settings: ChatSettings;
  isLoading: boolean;
  isTyping: boolean;
  isStreaming: boolean;
  modelStatus: "unknown" | "loading" | "ready" | "error";
  contextInjection: {
    enabled: boolean;
    documents: string[];
    vectorResults: any[];
    precedents?: string[];
    caseContext?: unknown;
  };
}

// === INITIAL STATE ===
const initialState: ChatContext = {
  messages: [],
  conversations: [],
  currentConversation: null,
  error: null,
  isLoading: false,
  isTyping: false,
  isStreaming: false,
  modelStatus: "unknown",
  settings: {
    model: "gemma3-legal",
    temperature: 0.1,
    maxTokens: 1024,
    streaming: true,
    contextWindow: 8192,
    proactiveMode: true,
    emotionalMode: false,
    legalMode: true,
    citationMode: true,
  },
  contextInjection: {
    enabled: false,
    documents: [],
    vectorResults: [],
    precedents: [],
    caseContext: null,
  },
};

// === MAIN STORE ===
export const chatStore = writable<ChatContext>(initialState);
;
// === SERVICE STATUS ===
export const serviceStatus = writable<ServiceStatus>({
  ollama: "unknown",
  qdrant: "unknown",
  database: "unknown",
  gemma3: "unknown",
});

// === DERIVED STORES ===
export const messages = derived(chatStore, ($store) => $store.messages);
export const currentConversation = derived(chatStore, ($store) => $store.currentConversation);
export const conversations = derived(chatStore, ($store) => $store.conversations);
export const isLoading = derived(chatStore, ($store) => $store.isLoading);
export const isStreaming = derived(chatStore, ($store) => $store.isStreaming);
export const isTyping = derived(chatStore, ($store) => $store.isTyping);
export const error = derived(chatStore, ($store) => $store.error);
export const settings = derived(chatStore, ($store) => $store.settings);
export const modelStatus = derived(chatStore, ($store) => $store.modelStatus);
export const contextInjection = derived(chatStore, ($store) => $store.contextInjection);
export const conversationsList = derived(conversations, ($conversations) =>
  [...$conversations].sort((a, b) => b.updated.getTime() - a.updated.getTime())
);
export const isActiveChat = derived(currentConversation, ($conversation) => !!$conversation);

// === ACTIONS ===
export const chatActions = {
  // Create new conversation
  newConversation: (title?: string, caseType?: string) => {
    const conversation: Conversation = {
      id: crypto.randomUUID(),
      title: title || "New Legal Consultation",
      messages: [],
      created: new Date(),
      updated: new Date(),
      metadata: {
        caseType: caseType || "general",
        jurisdiction: "federal",
        precedents: [],
      },
    };

    chatStore.update((state) => ({
      ...state,
      currentConversation: conversation,
      conversations: [conversation, ...state.conversations],
      messages: [],
    }));

    return conversation.id;
  },

  // Load conversation
  loadConversation: (conversationId: string) => {
    chatStore.update((state) => {
      const conversation = state.conversations.find(
        (c) => c.id === conversationId,
      );
      return {
        ...state,
        currentConversation: conversation || null,
        messages: conversation?.messages || [],
      };
    });
  },

  // Add message
  addMessage: (content: string, role: "user" | "assistant" | "system", metadata?: unknown) => {
    chatStore.update((state) => {
      if (!state.currentConversation) {
        // Create new conversation if none exists
        const conversation: Conversation = {
          id: crypto.randomUUID(),
          title: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
          messages: [],
          created: new Date(),
          updated: new Date(),
          metadata: {
            caseType: "general",
            jurisdiction: "federal",
            precedents: [],
          },
        };
        state.currentConversation = conversation;
        state.conversations = [conversation, ...state.conversations];
      }

      const message: ChatMessage = {
        id: crypto.randomUUID(),
        content,
        role,
        timestamp: new Date(),
        conversationId: state.currentConversation.id,
        metadata,
      };

      const updatedMessages = [...state.messages, message];
      state.currentConversation.messages = updatedMessages;
      state.currentConversation.updated = new Date();

      // Update title if it's the first user message
      if (
        role === "user" &&
        updatedMessages.filter((m) => m.role === "user").length === 1
      ) {
        state.currentConversation.title =
          content.slice(0, 50) + (content.length > 50 ? "..." : "");
      }

      return {
        ...state,
        messages: updatedMessages,
      };
    });
  },

  // Send message with streaming support
  sendMessage: async (content: string) => {
    chatActions.addMessage(content, "user");

    chatStore.update((state) => ({
      ...state,
      isLoading: true,
      isTyping: true,
      error: null,
    }));

    try {
      const response = await fetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          conversationId: getCurrentConversationId(),
          settings: getSettings(),
          contextInjection: getContextInjection(),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Handle streaming vs non-streaming responses
      if (response.headers.get("content-type")?.includes("text/stream")) {
        await handleStreamingResponse(response);
      } else {
        const data = await response.json();
        chatActions.addMessage(data.response, "assistant", {
          model: data.model,
          tokensUsed: data.tokensUsed,
          references: data.references,
          confidence: data.confidence,
          legalContext: data.legalContext,
        });
      }
    } catch (error: any) {
      console.error("Error sending message:", error);
      chatStore.update((state) => ({
        ...state,
        error:
          error instanceof Error ? error : new Error("Failed to send message"),
      }));
    } finally {
      chatStore.update((state) => ({
        ...state,
        isLoading: false,
        isTyping: false,
        isStreaming: false,
      }));
    }
  },

  // Delete conversation
  deleteConversation: (conversationId: string) => {
    chatStore.update((state) => {
      const conversations = state.conversations.filter(
        (c) => c.id !== conversationId,
      );
      const currentConversation =
        state.currentConversation?.id === conversationId
          ? null
          : state.currentConversation;

      return {
        ...state,
        conversations,
        currentConversation,
        messages: currentConversation?.messages || [],
      };
    });
  },

  // Update settings
  updateSettings: (newSettings: Partial<ChatSettings>) => {
    chatStore.update((state) => ({
      ...state,
      settings: { ...state.settings, ...newSettings },
    }));
  },

  // Legal-specific context injection
  injectLegalContext: (documents: string[], precedents?: string[], caseContext?: unknown) => {
    chatStore.update((state) => ({
      ...state,
      contextInjection: {
        ...state.contextInjection,
        enabled: true,
        documents,
        precedents: precedents || [],
        caseContext: caseContext || null,
      },
    }));
  },

  // Context injection
  injectContext: (documents: string[]) => {
    chatStore.update((state) => ({
      ...state,
      contextInjection: {
        ...state.contextInjection,
        enabled: true,
        documents,
      },
    }));
  },

  clearContext: () => {
    chatStore.update((state) => ({
      ...state,
      contextInjection: {
        enabled: false,
        documents: [],
        vectorResults: [],
        precedents: [],
        caseContext: null,
      },
    }));
  },

  // Model status
  checkModelStatus: async () => {
    chatStore.update((state) => ({ ...state, modelStatus: "loading" }));

    try {
      const response = await fetch("/api/ai/model-status");
      if (response.ok) {
        const data = await response.json();
        chatStore.update((state) => ({
          ...state,
          modelStatus: data.status || "ready",
        }));
      } else {
        chatStore.update((state) => ({ ...state, modelStatus: "error" }));
      }
    } catch (error: any) {
      chatStore.update((state) => ({ ...state, modelStatus: "error" }));
    }
  },

  // Error handling
  clearError: () => {
    chatStore.update((state) => ({ ...state, error: null }));
  },

  // Reset chat
  resetChat: () => {
    chatStore.update((state) => ({
      ...state,
      currentConversation: null,
      messages: [],
      error: null,
      isLoading: false,
      isTyping: false,
      isStreaming: false,
    }));
  },

  // Loading states
  setLoading: (loading: boolean) => {
    chatStore.update((state) => ({ ...state, isLoading: loading }));
  },

  setTyping: (typing: boolean) => {
    chatStore.update((state) => ({ ...state, isTyping: typing }));
  },

  setStreaming: (streaming: boolean) => {
    chatStore.update((state) => ({ ...state, isStreaming: streaming }));
  },
};

// === SERVICE ACTIONS ===
export const serviceActions = {
  updateStatus: (
    service: keyof ServiceStatus,
    status: ServiceStatus[keyof ServiceStatus],
  ) => {
    serviceStatus.update((current) => ({ ...current, [service]: status }));
  },

  checkAllServices: async () => {
    // Check Ollama
    try {
      const ollamaResponse = await fetch("/api/ai/test-ollama");
      serviceActions.updateStatus(
        "ollama",
        ollamaResponse.ok ? "connected" : "error",
      );
    } catch {
      serviceActions.updateStatus("ollama", "error");
    }

    // Check Gemma3 model
    try {
      const modelResponse = await fetch("/api/ai/model-status");
      if (modelResponse.ok) {
        const data = await modelResponse.json();
        serviceActions.updateStatus("gemma3", data.status || "ready");
      } else {
        serviceActions.updateStatus("gemma3", "error");
      }
    } catch {
      serviceActions.updateStatus("gemma3", "error");
    }

    // Check database
    try {
      const dbResponse = await fetch("/api/health/database");
      serviceActions.updateStatus(
        "database",
        dbResponse.ok ? "connected" : "error",
      );
    } catch {
      serviceActions.updateStatus("database", "error");
    }

    // Check Qdrant
    try {
      const qdrantResponse = await fetch("/api/health/qdrant");
      serviceActions.updateStatus(
        "qdrant",
        qdrantResponse.ok ? "connected" : "error",
      );
    } catch {
      serviceActions.updateStatus("qdrant", "error");
    }
  },
};

// === HELPER FUNCTIONS ===
function getCurrentConversationId(): string | undefined {
  let currentId: string | undefined;
  const unsubscribe = chatStore.subscribe((state) => {
    currentId = state.currentConversation?.id;
  });
  unsubscribe();
  return currentId;
}

function getSettings(): ChatSettings {
  let settings: ChatSettings;
  const unsubscribe = chatStore.subscribe((state) => {
    settings = state.settings;
  });
  unsubscribe();
  return settings!;
}

function getContextInjection() {
  let context: any;
  const unsubscribe = chatStore.subscribe((state) => {
    context = state.contextInjection;
  });
  unsubscribe();
  return context;
}

// Handle streaming responses
async function handleStreamingResponse(response: Response): Promise<any> {
  if (!response.body) return;

  chatStore.update((state) => ({ ...state, isStreaming: true }));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let assistantMessage = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      assistantMessage += chunk;

      // Update the last message or create new one
      chatStore.update((state) => {
        const messages = [...state.messages];
        const lastMessage = messages[messages.length - 1];

        if (lastMessage && lastMessage.role === "assistant") {
          messages[messages.length - 1] = {
            ...lastMessage,
            content: assistantMessage,
          };
        } else {
          messages.push({
            id: crypto.randomUUID(),
            content: assistantMessage,
            role: "assistant",
            timestamp: new Date(),
            conversationId: state.currentConversation?.id,
          });
        }

        return { ...state, messages };
      });
    }
  } catch (error: any) {
    console.error("Streaming error:", error);
    chatStore.update((state) => ({
      ...state,
      error: error instanceof Error ? error : new Error("Streaming failed"),
    }));
  }
}

// === XSTATE-LIKE COMPATIBILITY ===
export interface XStateCompatibleState {
  context: ChatContext;
  matches: (state: string) => boolean;
}

export const xstateCompatibleStore = derived(chatStore, ($chatStore): XStateCompatibleState => ({
  context: $chatStore,
  matches: (state: string) => {
    switch (state) {
      case "loading":
        return $chatStore.isLoading;
      case "streaming":
        return $chatStore.isStreaming;
      case "error":
        return !!$chatStore.error;
      case "idle":
        return !$chatStore.isLoading && !$chatStore.error;
      default:
        return false;
    }
  },
}));

export function useChatActor() {
  return {
    state: readonly(xstateCompatibleStore),
  };
}

// === LEGAL AI INTEGRATION ===
export const legalActions = {
  // Search legal precedents
  searchPrecedents: async (query: string, jurisdiction?: string) => {
    try {
      const response = await fetch("/api/legal/precedents/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, jurisdiction }),
      });

      if (response.ok) {
        const precedents = await response.json();
        chatStore.update((state) => ({
          ...state,
          contextInjection: {
            ...state.contextInjection,
            precedents: precedents.results || [],
          },
        }));
        return precedents;
      }
    } catch (error: any) {
      console.error("Error searching precedents:", error);
    }
    return null;
  },

  // Analyze legal document
  analyzeDocument: async (documentText: string) => {
    try {
      const response = await fetch("/api/legal/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: documentText }),
      });

      if (response.ok) {
        const analysis = await response.json();
        return analysis;
      }
    } catch (error: any) {
      console.error("Error analyzing document:", error);
    }
    return null;
  },

  // Extract legal entities
  extractEntities: async (text: string) => {
    try {
      const response = await fetch("/api/legal/entities", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (error: any) {
      console.error("Error extracting entities:", error);
    }
    return null;
  },
};

// === PERSISTENCE ===
export const persistenceHelpers = {
  saveToStorage: () => {
    if (typeof window === "undefined") return;

    const unsubscribe = chatStore.subscribe((state) => {
      try {
        localStorage.setItem(
          "chat-conversations",
          JSON.stringify(state.conversations),
        );
        localStorage.setItem("chat-settings", JSON.stringify(state.settings));
      } catch (error: any) {
        console.warn("Failed to save chat data to localStorage:", error);
      }
    });

    return unsubscribe;
  },

  loadFromStorage: () => {
    if (typeof window === "undefined") return;

    try {
      const conversations = localStorage.getItem("chat-conversations");
      const settings = localStorage.getItem("chat-settings");

      chatStore.update((state) => ({
        ...state,
        conversations: conversations
          ? JSON.parse(conversations).map((c: any) => ({
              ...c,
              created: new Date(c.created),
              updated: new Date(c.updated),
              messages: c.messages.map((m: any) => ({
                ...m,
                timestamp: new Date(m.timestamp),
              })),
            }))
          : state.conversations,
        settings: settings
          ? { ...state.settings, ...JSON.parse(settings) }
          : state.settings,
      }));
    } catch (error: any) {
      console.warn("Failed to load chat data from localStorage:", error);
    }
  },
};

// Initialize persistence on client-side
if (typeof window !== "undefined") {
  persistenceHelpers.loadFromStorage();
  persistenceHelpers.saveToStorage();
}

// Export default store for convenience
export default chatStore;