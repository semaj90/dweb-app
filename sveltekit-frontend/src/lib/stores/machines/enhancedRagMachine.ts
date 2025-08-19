// @ts-nocheck
import { createMachine, assign  } from "xstate";
// Orphaned content: import {

export const enhancedRagMachine = createMachine(
  {
    id: 'enhancedRag',
    initial: 'idle',
    context: {
      query: '',
      results: [],
      error: null,
      loading: false,
    },
    states: {
      idle: {
        on: { EXECUTE: { target: 'retrieving', actions: 'setQuery' } },
      },
      retrieving: {
        entry: assign({ loading: true, error: null }),
        invoke: {
          src: (ctx) =>
            fetch('/api/rag/enhanced', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ query: ctx.query, k: 8 }),
            }).then((r) => r.json()),
          onDone: {
            target: 'ready',
            actions: assign({ results: (_, e) => e.data.results, loading: false }),
          },
          onError: {
            target: 'failure',
            actions: assign({ error: (_, e) => e.data?.message || 'RAG failed', loading: false }),
          },
        },
      },
      ready: {
        on: { EXECUTE: 'retrieving', RESET: { target: 'idle', actions: 'reset' } },
      },
      failure: {
        on: { RETRY: 'retrieving', RESET: { target: 'idle', actions: 'reset' } },
      },
    },
  },
  {
    actions: {
      setQuery: assign({ query: (_, e: any) => e.query }),
      reset: assign({ query: '', results: [], error: null, loading: false }),
    },
  }
);

export const enhancedRagStore = writable({
  state: 'idle',
  results: [],
  loading: false,
  error: null,
});

