import { createMachine, assign, interpret } from 'xstate';

export const legalProcessingMachine = createMachine({
  id: 'legalProcessing',
  initial: 'idle',
  context: {
    query: '',
    results: [],
    recommendations: [],
    didYouMean: [],
    jobId: null,
    progress: 0,
    useGPU: true,
    llmProvider: 'ollama',
    userId: null,
    sessionId: null,
    synthesizedAnswer: '',
    errors: []
  },
  states: {
    idle: {
      on: {
        SUBMIT_QUERY: { target: 'preprocessing', actions: ['setQuery'] },
        LOAD_HISTORY: { target: 'loadingHistory' }
      }
    },
    preprocessing: {
      invoke: {
        src: 'preprocessQuery',
        onDone: { target: 'processing', actions: ['setDidYouMean'] },
        onError: { target: 'error', actions: ['setError'] }
      }
    },
    processing: {
      invoke: {
        src: 'processQuery',
        onDone: { target: 'completed', actions: ['setResults'] },
        onError: { target: 'error', actions: ['setError'] }
      }
    },
    completed: {
      entry: ['recordActivity'],
      on: {
        NEW_QUERY: { target: 'idle', actions: ['resetContext'] },
        REFINE: { target: 'preprocessing', actions: ['updateQuery'] }
      }
    },
    error: {
      on: {
        RETRY: { target: 'preprocessing', actions: ['clearErrors'] },
        NEW_QUERY: { target: 'idle', actions: ['resetContext'] }
      }
    },
    loadingHistory: {
      invoke: {
        src: 'loadHistory',
        onDone: { target: 'idle', actions: ['setHistory'] },
        onError: { target: 'error', actions: ['setError'] }
      }
    }
  }
}, {
  actions: {
    setQuery: assign({ query: (_, event) => event.query, userId: (_, event) => event.userId }),
    setDidYouMean: assign({ didYouMean: (_, event) => event.data.suggestions || [] }),
    setResults: assign({ 
      results: (_, event) => event.data.results || [],
      synthesizedAnswer: (_, event) => event.data.synthesizedAnswer || '',
      recommendations: (_, event) => event.data.recommendations || []
    }),
    setError: assign({ errors: (ctx, event) => [...ctx.errors, event.data] }),
    clearErrors: assign({ errors: [] }),
    resetContext: assign({
      query: '', results: [], recommendations: [], didYouMean: [], 
      errors: [], jobId: null, progress: 0, synthesizedAnswer: ''
    }),
    updateQuery: assign({ query: (_, event) => event.query }),
    recordActivity: (context) => {
      if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
        navigator.serviceWorker.ready.then(reg => 
          reg.active?.postMessage({
            type: 'RECORD_ACTIVITY',
            data: { userId: context.userId, query: context.query, timestamp: Date.now() }
          })
        );
      }
    }
  },
  services: {
    preprocessQuery: async (context) => {
      const response = await fetch('/api/legal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          endpoint: 'did-you-mean',
          query: context.query,
          threshold: 0.7
        })
      });
      if (!response.ok) throw new Error('Preprocessing failed');
      return response.json();
    },
    processQuery: async (context) => {
      const response = await fetch('/api/legal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          endpoint: 'rag-enhanced',
          query: context.query,
          userId: context.userId,
          useGPU: context.useGPU,
          llmProvider: context.llmProvider,
          topK: 20
        })
      });
      if (!response.ok) throw new Error('Processing failed');
      return response.json();
    },
    loadHistory: async (context) => {
      const response = await fetch(`/api/legal?endpoint=activity/${context.userId}`);
      if (!response.ok) throw new Error('History load failed');
      return response.json();
    }
  }
});

export function createLegalProcessingService() {
  return interpret(legalProcessingMachine);
}