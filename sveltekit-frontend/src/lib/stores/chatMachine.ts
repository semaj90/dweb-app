/**
 * XState Chat Machine for Svelte 5 + Gemma3 Integration
 * Enhanced with streaming, context injection, and model selection
 */

import { createMachine, assign } from "xstate";
import type { ChatMessage, Conversation, ChatSettings } from "./types";

export interface ChatContext {
  messages: ChatMessage[];
  conversations: Conversation[];
  currentConversation: Conversation | null;
  error: Error | null;
  settings: ChatSettings;
  stream: ReadableStream<string> | null;
  modelStatus: "unknown" | "loading" | "ready" | "error";
  contextInjection: {
    enabled: boolean;
    documents: string[];
    vectorResults: any[];
  };
}

export type ChatEvent =
  | { type: "SEND_MESSAGE"; message: string }
  | { type: "RECEIVE_MESSAGE"; message: string; metadata?: any }
  | { type: "START_STREAMING" }
  | { type: "STREAM_CHUNK"; chunk: string }
  | { type: "STREAM_COMPLETE" }
  | { type: "STREAM_ERROR"; error: Error }
  | { type: "NEW_CONVERSATION"; title?: string }
  | { type: "LOAD_CONVERSATION"; conversationId: string }
  | { type: "DELETE_CONVERSATION"; conversationId: string }
  | { type: "UPDATE_SETTINGS"; settings: Partial<ChatSettings> }
  | { type: "INJECT_CONTEXT"; documents: string[] }
  | { type: "CLEAR_CONTEXT" }
  | { type: "CHECK_MODEL_STATUS" }
  | { type: "MODEL_READY" }
  | { type: "MODEL_ERROR"; error: Error }
  | { type: "CLEAR_ERROR" }
  | { type: "RESET_CHAT" };

const initialContext: ChatContext = {
  messages: [],
  conversations: [],
  currentConversation: null,
  error: null,
  stream: null,
  modelStatus: "unknown",
  settings: {
    model: "gemma3-legal",
    temperature: 0.1,
    maxTokens: 1024,
    streaming: true,
    contextWindow: 8192,
    proactiveMode: true,
    emotionalMode: false,
  },
  contextInjection: {
    enabled: false,
    documents: [],
    vectorResults: [],
  },
};

export const chatMachine = createMachine(
  {
    id: "chat",
    initial: "idle",
    context: initialContext,
    states: {
      idle: {
        on: {
          SEND_MESSAGE: "sendingMessage",
          NEW_CONVERSATION: {
            actions: "createNewConversation",
          },
          LOAD_CONVERSATION: {
            actions: "loadConversation",
          },
          DELETE_CONVERSATION: {
            actions: "deleteConversation",
          },
          UPDATE_SETTINGS: {
            actions: "updateSettings",
          },
          INJECT_CONTEXT: {
            actions: "injectContext",
          },
          CLEAR_CONTEXT: {
            actions: "clearContext",
          },
          CHECK_MODEL_STATUS: "checkingModel",
          RESET_CHAT: {
            actions: "resetChat",
          },
        },
      },

      sendingMessage: {
        entry: "addUserMessage",
        invoke: {
          id: "sendMessage",
          src: "sendMessageService",
          onDone: {
            target: "idle",
            actions: "handleMessageResponse",
          },
          onError: {
            target: "error",
            actions: "setError",
          },
        },
        on: {
          START_STREAMING: "streaming",
        },
      },

      streaming: {
        entry: "initializeStream",
        on: {
          STREAM_CHUNK: {
            actions: "appendStreamChunk",
          },
          STREAM_COMPLETE: {
            target: "idle",
            actions: "finalizeStream",
          },
          STREAM_ERROR: {
            target: "error",
            actions: "setStreamError",
          },
        },
      },

      checkingModel: {
        invoke: {
          id: "checkModel",
          src: "checkModelService",
          onDone: {
            target: "idle",
            actions: "setModelReady",
          },
          onError: {
            target: "idle",
            actions: "setModelError",
          },
        },
      },

      error: {
        on: {
          CLEAR_ERROR: {
            target: "idle",
            actions: "clearError",
          },
          SEND_MESSAGE: "sendingMessage",
        },
      },
    },
  },
  {
    actions: {
      addUserMessage: assign({
        messages: ({ context, event }) => {
          if (event.type !== "SEND_MESSAGE") return context.messages;

          const message: ChatMessage = {
            id: crypto.randomUUID(),
            content: event.message,
            role: "user",
            timestamp: new Date(),
            conversationId: context.currentConversation?.id,
          };

          return [...context.messages, message];
        },
        currentConversation: ({ context, event }) => {
          if (event.type !== "SEND_MESSAGE") return context.currentConversation;

          if (!context.currentConversation) {
            const conversation: Conversation = {
              id: crypto.randomUUID(),
              title:
                event.message.slice(0, 50) +
                (event.message.length > 50 ? "..." : ""),
              messages: [],
              created: new Date(),
              updated: new Date(),
            };
            return conversation;
          }

          return {
            ...context.currentConversation,
            updated: new Date(),
          };
        },
      }),

      handleMessageResponse: assign({
        messages: ({ context, event }) => {
          const response: ChatMessage = {
            id: crypto.randomUUID(),
            content: event.output.response,
            role: "assistant",
            timestamp: new Date(),
            conversationId: context.currentConversation?.id,
            metadata: event.output.metadata,
          };

          return [...context.messages, response];
        },
      }),

      createNewConversation: assign({
        currentConversation: ({ event }) => {
          const title =
            event.type === "NEW_CONVERSATION"
              ? event.title || "New Conversation"
              : "New Conversation";
          return {
            id: crypto.randomUUID(),
            title,
            messages: [],
            created: new Date(),
            updated: new Date(),
          };
        },
        messages: () => [],
      }),

      loadConversation: assign({
        currentConversation: ({ context, event }) => {
          if (event.type !== "LOAD_CONVERSATION")
            return context.currentConversation;
          return (
            context.conversations.find((c) => c.id === event.conversationId) ||
            null
          );
        },
        messages: ({ context, event }) => {
          if (event.type !== "LOAD_CONVERSATION") return context.messages;
          const conversation = context.conversations.find(
            (c) => c.id === event.conversationId
          );
          return conversation?.messages || [];
        },
      }),

      deleteConversation: assign({
        conversations: ({ context, event }) => {
          if (event.type !== "DELETE_CONVERSATION")
            return context.conversations;
          return context.conversations.filter(
            (c) => c.id !== event.conversationId
          );
        },
        currentConversation: ({ context, event }) => {
          if (event.type !== "DELETE_CONVERSATION")
            return context.currentConversation;
          return context.currentConversation?.id === event.conversationId
            ? null
            : context.currentConversation;
        },
      }),

      updateSettings: assign({
        settings: ({ context, event }) => {
          if (event.type !== "UPDATE_SETTINGS") return context.settings;
          return { ...context.settings, ...event.settings };
        },
      }),

      injectContext: assign({
        contextInjection: ({ context, event }) => {
          if (event.type !== "INJECT_CONTEXT") return context.contextInjection;
          return {
            ...context.contextInjection,
            enabled: true,
            documents: event.documents,
          };
        },
      }),

      clearContext: assign({
        contextInjection: ({ context }) => ({
          ...context.contextInjection,
          enabled: false,
          documents: [],
          vectorResults: [],
        }),
      }),

      initializeStream: assign({
        stream: () => null, // Will be set by streaming service
      }),

      appendStreamChunk: assign({
        messages: ({ context, event }) => {
          if (event.type !== "STREAM_CHUNK") return context.messages;

          const lastMessage = context.messages[context.messages.length - 1];
          if (lastMessage && lastMessage.role === "assistant") {
            return [
              ...context.messages.slice(0, -1),
              { ...lastMessage, content: lastMessage.content + event.chunk },
            ];
          }

          // Create new assistant message if none exists
          const newMessage: ChatMessage = {
            id: crypto.randomUUID(),
            content: event.chunk,
            role: "assistant",
            timestamp: new Date(),
            conversationId: context.currentConversation?.id,
          };

          return [...context.messages, newMessage];
        },
      }),

      finalizeStream: assign({
        stream: () => null,
      }),

      setModelReady: assign({
        modelStatus: () => "ready",
      }),

      setModelError: assign({
        modelStatus: () => "error",
        error: ({ event }) =>
          event.type === "MODEL_ERROR" ? event.error : null,
      }),

      setError: assign({
        error: ({ event }) =>
          new Error(event.type === "error" ? event.data : "Unknown error"),
      }),

      setStreamError: assign({
        error: ({ event }) =>
          event.type === "STREAM_ERROR" ? event.error : null,
        stream: () => null,
      }),

      clearError: assign({
        error: () => null,
      }),

      resetChat: assign({
        currentConversation: () => null,
        messages: () => [],
        error: () => null,
        stream: () => null,
      }),
    },

    services: {
      sendMessageService: async ({ context }) => {
        const response = await fetch("/api/ai/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: context.messages[context.messages.length - 1]?.content,
            conversationId: context.currentConversation?.id,
            settings: context.settings,
            contextInjection: context.contextInjection.enabled
              ? {
                  documents: context.contextInjection.documents,
                }
              : undefined,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return data;
      },

      checkModelService: async () => {
        const response = await fetch("/api/ai/model-status");

        if (!response.ok) {
          throw new Error("Model not ready");
        }

        return await response.json();
      },
    },
  }
);
