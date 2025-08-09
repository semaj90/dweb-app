// @ts-nocheck
// XState Machine for AI Agent Shell
import { createMachine, assign } from "xstate";

// Define context and event types
interface AgentShellContext {
  input: string;
  response: string;
  jobId?: string;
  rating?: number;
  searchQuery?: string;
  searchResults?: any;
}

type AgentShellEvent =
  | { type: "PROMPT"; input: string }
  | { type: "xstate.done.actor.callAgent"; data: string }
  | { type: "ACCEPT_PATCH"; jobId: string }
  | { type: "RATE_SUGGESTION"; jobId: string; rating: number }
  | { type: "SEMANTIC_SEARCH"; query: string };

export const agentShellMachine = createMachine({
  id: "agentShell",
  initial: "idle",
  context: { input: "", response: "" },
  types: {} as {
    context: AgentShellContext;
    events: AgentShellEvent;
  },
  states: {
    idle: {
      on: {
        PROMPT: {
          target: "processing",
          actions: assign({
            input: ({ event }) => (event as any).input || "",
          }),
        },
      },
    },
    processing: {
      invoke: {
        src: "callAgent",
        onDone: {
          target: "idle",
          actions: assign({
            response: (_, e) => (e && "data" in e ? (e as any).data : ""),
          }),
        },
        onError: "idle",
      },
      // Next step: handle agent patch acceptance, rating, and semantic search
      on: {
        ACCEPT_PATCH: {
          actions: "acceptPatchAction",
        },
        RATE_SUGGESTION: {
          actions: "rateSuggestionAction",
        },
        SEMANTIC_SEARCH: {
          actions: "semanticSearchAction",
        },
      },
    },
  },
});
