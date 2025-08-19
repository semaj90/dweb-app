// @ts-nocheck
// lib/stores/ai.ts
// Global AI Summary Store using XState v5, with memoization and streaming support
import { setup, createActor, assign, fromPromise } from "xstate";
import { writable, , // Memoization cache (in-memory, can be replaced with Redis for persistence), const summaryCache = new Map<string, string>();, , // Define context and events interfaces, interface AIContext {,   summary: string;,   error: string;,   loading: boolean;,   caseId: string;,   evidence: any[];,   userId: string;,   stream: string;,   cacheKey: string;,   sources: any[]; } from

type AIEvent =
  | { type: "SUMMARIZE"; caseId: string; evidence: any[]; userId: string }
  | { type: "RETRY" }
  | { type: "RESET" };

export const aiGlobalMachine = setup({
  types: {
    context: {} as AIContext,
    events: {} as AIEvent,
  },
  actions: {
    setContext: assign(({ context, event }) => {
      if (event.type !== "SUMMARIZE") return {};
      const cacheKey = event.caseId + ":" + hashEvidence(event.evidence);
      return {
        caseId: event.caseId,
        evidence: event.evidence,
        userId: event.userId,
        cacheKey,
        loading: true,
        error: "",
        stream: "",
      };
    }),
    setSuccess: assign(({ event }) => {
      if ((event as any).type === "xstate.done.actor.summarizeEvidence") {
        const data = (event as any).output;
        return {
          summary: data?.summary || "",
          sources: data?.sources || [],
          loading: false,
          stream: "",
          error: "",
        };
      }
      return {};
    }),
    setError: assign(({ event }) => {
      if ((event as any).type === "xstate.error.actor.summarizeEvidence") {
        return {
          error: ((event as any).error as Error)?.message || "Error generating summary.",
          loading: false,
        };
      }
      return {};
    }),
  },
  actors: {
    summarizeEvidence: fromPromise(async ({ input }: { input: AIContext }) => {
      // Memoization: check cache first
      if (summaryCache.has(input.cacheKey)) {
        return {
          summary: summaryCache.get(input.cacheKey)!,
          sources: [],
        };
      }

      // Call AI summary API
      const res = await fetch("/api/ai-summary", {
        method: "POST",
        body: JSON.stringify({
          caseId: input.caseId,
          evidence: input.evidence,
          userId: input.userId,
        }),
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        throw new Error(`API request failed: ${res.statusText}`);
      }

      const data = await res.json();
      if (!data.summary) throw new Error("No summary returned");

      // Cache the result
      summaryCache.set(input.cacheKey, data.summary);

      return {
        summary: data.summary,
        sources: data.sources || [],
      };
    }),
  },
}).createMachine({
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
    sources: [],
  },
  states: {
    idle: {
      on: {
        SUMMARIZE: {
          target: "summarizing",
          actions: "setContext",
        },
      },
    },
    summarizing: {
      invoke: {
        src: "summarizeEvidence",
        input: ({ context }) => context,
        onDone: {
          target: "success",
          actions: "setSuccess",
        },
        onError: {
          target: "failure",
          actions: "setError",
        },
      },
    },
    success: {
      on: {
        SUMMARIZE: "summarizing",
        RESET: "idle",
      },
    },
    failure: {
      on: {
        SUMMARIZE: "summarizing",
        RETRY: "summarizing",
        RESET: "idle",
      },
    },
  },
});

// Utility: hash evidence array for cache key
function hashEvidence(evidence: any[]): string {
  // Simple hash, replace with a better hash for production
  if (typeof btoa !== "undefined") {
    return btoa(JSON.stringify(evidence)).slice(0, 32);
  }
  // Fallback for Node.js environment
  return Buffer.from(JSON.stringify(evidence)).toString("base64").slice(0, 32);
}

// Create and export the actor
export const aiGlobalActor = createActor(aiGlobalMachine);

// Svelte store wrapper for reactivity
export const aiGlobalStore = writable({
  state: "idle",
  context: aiGlobalMachine.config.context || {},
});

// Subscribe to actor state changes
aiGlobalActor.subscribe((state) => {
  aiGlobalStore.set({
    state: state.value as string,
    context: state.context,
  });
});

// Start the actor
aiGlobalActor.start();

// Export convenience functions
export const aiGlobalActions = {
  summarize: (caseId: string, evidence: any[], userId: string) => {
    aiGlobalActor.send({
      type: "SUMMARIZE",
      caseId,
      evidence,
      userId,
    });
  },
  retry: () => {
    aiGlobalActor.send({ type: "RETRY" });
  },
  reset: () => {
    aiGlobalActor.send({ type: "RESET" });
  },
};

export default aiGlobalStore;
