
/**
 * RAG State Machine - XState Implementation
 * Manages complex RAG system states and transitions
 */

import { createMachine, assign } from "xstate";

export interface RAGContext {
  query: string;
  results: any[];
  error: string | null;
  retryCount: number;
  searchStartTime: number;
  cacheStatus: "miss" | "hit" | "partial";
  optimizationLevel: "basic" | "enhanced" | "neural";
}

export type RAGEvent =
  | { type: "SEARCH_START"; query: string }
  | { type: "SEARCH_SUCCESS"; results: any[] }
  | { type: "SEARCH_ERROR"; error: string }
  | { type: "RETRY" }
  | { type: "OPTIMIZE" }
  | { type: "CACHE_HIT"; results: any[] }
  | { type: "RESET" };

export const ragStateMachine = createMachine({
  id: "ragSystem",
  initial: "idle",
  context: {
    query: "",
    results: [],
    error: null,
    retryCount: 0,
    searchStartTime: 0,
    cacheStatus: "miss",
    optimizationLevel: "basic",
  },
  states: {
    idle: {
      on: {
        SEARCH_START: {
          target: "searching",
          actions: assign({
            query: ({ event }) => event.query,
            searchStartTime: () => Date.now(),
            retryCount: 0,
            error: null,
          }),
        },
      },
    },
    searching: {
      on: {
        SEARCH_SUCCESS: {
          target: "success",
          actions: assign({
            results: ({ event }) => event.results,
            cacheStatus: "miss",
          }),
        },
        SEARCH_ERROR: {
          target: "error",
          actions: assign({
            error: ({ event }) => event.error,
          }),
        },
        CACHE_HIT: {
          target: "success",
          actions: assign({
            results: ({ event }) => event.results,
            cacheStatus: "hit",
          }),
        },
      },
    },
    success: {
      on: {
        SEARCH_START: {
          target: "searching",
          actions: assign({
            query: ({ event }) => event.query,
            searchStartTime: () => Date.now(),
            retryCount: 0,
            error: null,
          }),
        },
        OPTIMIZE: {
          target: "optimizing",
        },
        RESET: {
          target: "idle",
          actions: assign({
            query: "",
            results: [],
            error: null,
            retryCount: 0,
          }),
        },
      },
    },
    error: {
      on: {
        RETRY: {
          target: "searching",
          guard: ({ context }) => context.retryCount < 3,
          actions: assign({
            retryCount: ({ context }) => context.retryCount + 1,
            error: null,
          }),
        },
        SEARCH_START: {
          target: "searching",
          actions: assign({
            query: ({ event }) => event.query,
            searchStartTime: () => Date.now(),
            retryCount: 0,
            error: null,
          }),
        },
        RESET: {
          target: "idle",
          actions: assign({
            query: "",
            results: [],
            error: null,
            retryCount: 0,
          }),
        },
      },
    },
    optimizing: {
      after: {
        2000: {
          target: "success",
          actions: assign({
            optimizationLevel: ({ context }) =>
              context.optimizationLevel === "basic"
                ? "enhanced"
                : context.optimizationLevel === "enhanced"
                  ? "neural"
                  : "neural",
          }),
        },
      },
      on: {
        SEARCH_START: {
          target: "searching",
          actions: assign({
            query: ({ event }) => event.query,
            searchStartTime: () => Date.now(),
            retryCount: 0,
            error: null,
          }),
        },
      },
    },
  },
});
