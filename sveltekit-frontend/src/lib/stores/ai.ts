// lib/stores/ai.ts
// Global AI Summary Store using XState, with memoization and streaming support
import { createMachine, assign } from "xstate";
import { useMachine } from "@xstate/svelte";

// Memoization cache (in-memory, can be replaced with Redis for persistence)
const summaryCache = new Map<string, string>();

export const aiGlobalMachine = createMachine(
  {
    id: "aiGlobalSummary",
    initial: "idle",
    context: {
      summary: "",
      error: "",
      loading: false,
      caseId: "",
      evidence: [],
      userId: "",
      stream: "",
      cacheKey: "",
      sources: [] as any[], // Top evidence sources
    },
    states: {
      idle: {
        on: { SUMMARIZE: { target: "summarizing", actions: "setContext" } },
      },
      summarizing: {
        entry: assign({ loading: (_) => true, error: (_) => "", stream: "" }),
        invoke: {
          src: "summarizeEvidence",
          onDone: {
            target: "success",
            actions: assign((ctx, e) => {
              summaryCache.set(ctx.cacheKey, e.data.summary);
              return {
                summary: e.data.summary,
                sources: e.data.sources || [],
                loading: false,
                stream: "",
              };
            }),
          },
          onError: {
            target: "failure",
            actions: assign({
              error: (_, e) => e.data || "Error generating summary.",
              loading: (_) => false,
            }),
          },
        },
      },
      success: { on: { SUMMARIZE: "summarizing" } },
      failure: { on: { SUMMARIZE: "summarizing" } },
    },
  },
  {
    actions: {
      setContext: assign((ctx, evt) => {
        const cacheKey = evt.caseId + ":" + hashEvidence(evt.evidence);
        return {
          caseId: evt.caseId,
          evidence: evt.evidence,
          userId: evt.userId,
          cacheKey,
        };
      }),
    },
    services: {
      summarizeEvidence: async (ctx) => {
        // Memoization: check cache first
        if (summaryCache.has(ctx.cacheKey)) {
          return { summary: summaryCache.get(ctx.cacheKey), sources: [] };
        }
        // Streaming support (if API supports ReadableStream)
        const res = await fetch("/api/ai-summary", {
          method: "POST",
          body: JSON.stringify({
            caseId: ctx.caseId,
            evidence: ctx.evidence,
            userId: ctx.userId,
          }),
          headers: { "Content-Type": "application/json" },
        });
        // If streaming, process stream here (see AiAssistant.svelte for UI)
        const data = await res.json();
        if (!data.summary) throw new Error("No summary returned");
        return { summary: data.summary, sources: data.sources || [] };
      },
    },
  }
);

// Utility: hash evidence array for cache key
function hashEvidence(evidence: any[]): string {
  // Simple hash, replace with a better hash for production
  return btoa(JSON.stringify(evidence)).slice(0, 32);
}

export const useAIGlobalStore = () => useMachine(aiGlobalMachine);
