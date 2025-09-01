
// SSR-safe AI store for Gemma3 Q4_K_M GGUF integration
// Manages LLM state, conversation history, and settings with proper hydration
import { writable, derived, get } from "svelte/store";
import { browser } from "$app/environment";
import type { AIResponse, LocalModel } from "$lib/data/types";
import type { ConversationHistory } from "$lib/types";

// Define Gemma3Config interface directly
export interface Gemma3Config {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number;
  repeatPenalty: number;
  systemPrompt: string;
  useSystemPrompt: boolean;
  streamOutput: boolean;
}

// SSR-safe storage utilities
const SSR_SAFE_STORAGE = {
  getItem: (key: string): string | null => {
    if (!browser) return null;
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  setItem: (key: string, value: string): void => {
    if (!browser) return;
    try {
      localStorage.setItem(key, value);
    } catch {
      // Silently fail in SSR or if storage is unavailable
    }
  },
  removeItem: (key: string): void => {
    if (!browser) return;
    try {
      localStorage.removeItem(key);
    } catch {
      // Silently fail
    }
  },
};

// AI conversation state interface
export interface AIConversationState {
  id: string;
  messages: Array<{
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: Date;
    sources?: Array<{
      id: string;
      title: string;
      content: string;
      score: number;
      type: string;
    }>;
    metadata?: {
      provider: "local" | "cloud" | "hybrid";
      model: string;
      confidence: number;
      executionTime: number;
      fromCache: boolean;
    };
  }>;
  isActive: boolean;
  lastUpdated: number;
}

// AI settings state interface
export interface AISettingsState {
  preferredProvider: "local" | "cloud" | "auto";
  gemma3Config: Gemma3Config;
  enableStreaming: boolean;
  maxHistoryLength: number;
  autoSave: boolean;
  uiTheme: "dark" | "light" | "auto";
}

// AI status state interface
export interface AIStatusState {
  isLoading: boolean;
  isInitializing: boolean;
  localModelAvailable: boolean;
  cloudModelAvailable: boolean;
  currentProvider: "local" | "cloud" | "hybrid" | null;
  currentModel: string | null;
  error: string | null;
  lastHealthCheck: number | null;
}

// Default states with SSR safety
const DEFAULT_CONVERSATION: AIConversationState = {
  id: "",
  messages: [],
  isActive: false,
  lastUpdated: 0,
};

const DEFAULT_SETTINGS: AISettingsState = {
  preferredProvider: "auto",
  gemma3Config: {
    model: "gemma2:2b",
    temperature: 0.7,
    maxTokens: 512,
    topP: 0.9,
    topK: 40,
    repeatPenalty: 1.1,
    systemPrompt:
      "You are a specialized legal AI assistant. Provide accurate, professional legal analysis based on the provided context.",
    useSystemPrompt: true,
    streamOutput: true,
  },
  enableStreaming: true,
  maxHistoryLength: 50,
  autoSave: true,
  uiTheme: "auto",
};

const DEFAULT_STATUS: AIStatusState = {
  isLoading: false,
  isInitializing: false,
  localModelAvailable: false,
  cloudModelAvailable: false,
  currentProvider: null,
  currentModel: null,
  error: null,
  lastHealthCheck: null,
};

// Create SSR-safe stores with persistence
function createPersistedStore<T>(key: string, defaultValue: T) {
  // Initialize with default value (SSR-safe)
  const { subscribe, set, update } = writable<T>(defaultValue);

  // Load from localStorage on hydration (browser only)
  if (browser) {
    const stored = SSR_SAFE_STORAGE.getItem(key);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        set(parsed);
      } catch (error: any) {
        console.warn(`Failed to parse stored ${key}:`, error);
        // Keep default value on parse error
      }
    }
  }
  return {
    subscribe,
    set: (value: T) => {
      set(value);
      if (browser) {
        SSR_SAFE_STORAGE.setItem(key, JSON.stringify(value));
      }
    },
    update: (updater: (value: T) => T) => {
      update((currentValue) => {
        const newValue = updater(currentValue);
        if (browser) {
          SSR_SAFE_STORAGE.setItem(key, JSON.stringify(newValue));
        }
        return newValue;
      });
    },
  };
}

// Main AI stores (restored from backup - removing corruption)
export const aiConversation = createPersistedStore<AIConversationState>(
  "ai_conversation",
  DEFAULT_CONVERSATION
);
export const aiSettings = createPersistedStore<AISettingsState>(
  "ai_settings",
  DEFAULT_SETTINGS
);
export const aiStatus = writable<AIStatusState>(DEFAULT_STATUS);

// Conversation history store (separate for performance)
export const conversationHistory = createPersistedStore<ConversationHistory[]>(
  "ai_conversation_history",
  []
);

// Derived stores for computed values
export const isAIReady = derived(
  [aiStatus],
  ([$aiStatus]) => $aiStatus.localModelAvailable || $aiStatus.cloudModelAvailable
);

export const currentModelInfo = derived(
  [aiStatus],
  ([$aiStatus]) => ({
    provider: $aiStatus.currentProvider,
    model: $aiStatus.currentModel,
    available: $aiStatus.localModelAvailable || $aiStatus.cloudModelAvailable,
  })
);

// AI store actions and utilities
export const aiStore = {
  // Initialize AI system
  async initialize(): Promise<void> {
    aiStatus.update((state) => ({
      ...state,
      isInitializing: true,
      error: null,
    }));

    try {
      // Check local model availability
      const localHealthCheck = await fetch("/api/ai/health/local", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      const localHealth = await localHealthCheck.json();

      // Check cloud model availability
      const cloudHealthCheck = await fetch("/api/ai/health/cloud", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      const cloudHealth = await cloudHealthCheck.json();

      aiStatus.update((state) => ({
        ...state,
        isInitializing: false,
        localModelAvailable: localHealth.success && localHealth.available,
        cloudModelAvailable: cloudHealth.success && cloudHealth.available,
        currentProvider:
          localHealth.success && localHealth.available
            ? "local"
            : cloudHealth.success && cloudHealth.available
              ? "cloud"
              : null,
        currentModel: localHealth.success
          ? localHealth.model
          : cloudHealth.success
            ? cloudHealth.model
            : null,
        lastHealthCheck: Date.now(),
      }));
    } catch (error: any) {
      console.error("AI initialization failed:", error);
      aiStatus.update((state) => ({
        ...state,
        isInitializing: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to initialize AI system",
      }));
    }
  },

  // Send message to AI
  async sendMessage(
    content: string,
    options: {
      includeHistory?: boolean;
      maxSources?: number;
      searchThreshold?: number;
      useCache?: boolean;
    } = {}
  ): Promise<AIResponse | null> {
    aiStatus.update((state) => ({ ...state, isLoading: true, error: null }));

    try {
      const response = await fetch("/api/ai/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: content,
          context: [],
          includeHistory: options.includeHistory ?? true,
          maxSources: options.maxSources ?? 5,
          searchThreshold: options.searchThreshold ?? 0.7,
          useCache: options.useCache ?? true,
        }),
      });

      if (!response.ok) {
        throw new Error(`AI request failed: ${response.statusText}`);
      }
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || "AI request failed");
      }
      const aiResponse = result.data as AIResponse;

      // Add to conversation
      aiConversation.update((conversation) => {
        const messageId = `msg_${Date.now()}`;
        const userMessage = {
          id: `${messageId}_user`,
          role: "user" as const,
          content,
          timestamp: new Date(),
        };
        const assistantMessage = {
          id: `${messageId}_assistant`,
          role: "assistant" as const,
          content: aiResponse.content,
          timestamp: new Date(),
          sources: aiResponse.sources,
          metadata: {
            provider: aiResponse.metadata.provider,
            model: aiResponse.metadata.model,
            confidence: aiResponse.metadata.confidence,
            executionTime: aiResponse.metadata.executionTime,
            fromCache: aiResponse.metadata.fromCache,
          },
        };

        return {
          id: conversation.id || `conv_${Date.now()}`,
          messages: [...conversation.messages, userMessage, assistantMessage],
          isActive: true,
          lastUpdated: Date.now(),
        } as AIConversationState;
      });

      aiStatus.update((state) => ({ ...state, isLoading: false }));
      return aiResponse;
    } catch (error: any) {
      console.error("AI message failed:", error);
      aiStatus.update((state) => ({
        ...state,
        isLoading: false,
        error:
          error instanceof Error ? error.message : "Failed to send message",
      }));
      return null;
    }
  },

  // Clear conversation
  clearConversation(): void {
    aiConversation.set(DEFAULT_CONVERSATION);
  },

  // Update settings
  updateSettings(newSettings: Partial<AISettingsState>): void {
    aiSettings.update((settings) => ({ ...settings, ...newSettings }));
  },

  // Reset to defaults
  reset(): void {
    aiConversation.set(DEFAULT_CONVERSATION);
    aiSettings.set(DEFAULT_SETTINGS);
    aiStatus.set(DEFAULT_STATUS);
    if (browser) {
      SSR_SAFE_STORAGE.removeItem("ai_conversation");
      SSR_SAFE_STORAGE.removeItem("ai_settings");
      SSR_SAFE_STORAGE.removeItem("ai_conversation_history");
    }
  },

  // Save conversation to history
  saveConversationToHistory(): void {
    const conversation = get(aiConversation);
    if (conversation.messages.length === 0) return;

    conversationHistory.update((history) => {
      const newConversation: ConversationHistory = {
        id: conversation.id || `conv_${Date.now()}`,
        title:
          conversation.messages[0]?.content.substring(0, 50) + "..." ||
          "Untitled Conversation",
        messages: conversation.messages.map((msg: any) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          sources: msg.sources
            ? msg.sources.map((source: any) => ({
                id: source.id,
                title: source.title,
                content: source.content,
                score: source.score,
                type: (source.type || "document") as
                  | "case"
                  | "evidence"
                  | "statute"
                  | "document",
              }))
            : undefined,
          metadata: msg.metadata,
        })),
        timestamp: Date.now(),
        metadata: {
          messageCount: conversation.messages.length,
          lastModel:
            conversation.messages[conversation.messages.length - 1]?.metadata
              ?.model || "unknown",
        },
      };

      const newHistory = [newConversation, ...history];

      // Limit history length
      const maxLength = get(aiSettings).maxHistoryLength;
      return newHistory.slice(0, maxLength);
    });
  },

  // Load conversation from history
  loadConversationFromHistory(historyId: string): void {
    const history = get(conversationHistory);
    const historyItem = history.find((item) => item.id === historyId);

    if (historyItem) {
      aiConversation.set({
        id: historyItem.id,
        messages: historyItem.messages as any,
        isActive: true,
        lastUpdated: Date.now(),
      });
    }
  },
};

// Auto-initialize on browser mount
if (browser) {
  // Initialize with a small delay to ensure proper hydration
  setTimeout(() => {
    aiStore.initialize().catch(console.error);
  }, 100);
}

// Export store subscriptions for reactive UI
export {
  aiConversation as conversation,
  aiSettings as settings,
  aiStatus as status,
};

export default aiStore;
