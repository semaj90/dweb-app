// @ts-nocheck
import { createMachine, assign, interpret, type InterpreterFrom } from 'xstate';

// Define event types
export interface SubmitQueryEvent {
  type: 'SUBMIT_QUERY';
  query: string;
  userId?: string;
}

export interface LoadHistoryEvent {
  type: 'LOAD_HISTORY';
}

export interface NewQueryEvent {
  type: 'NEW_QUERY';
}

export interface RefineEvent {
  type: 'REFINE';
  query: string;
}

export interface RetryEvent {
  type: 'RETRY';
}

// Union type for all events
export type LegalProcessingEvent = 
  | SubmitQueryEvent
  | LoadHistoryEvent
  | NewQueryEvent
  | RefineEvent
  | RetryEvent;

// Define context interface
export interface LegalProcessingContext {
  query: string;
  results: Array<any>;
  recommendations: Array<any>;
  didYouMean: Array<string>;
  jobId: string | null;
  progress: number;
  useGPU: boolean;
  llmProvider: string;
  userId: string | null;
  sessionId: string | null;
  synthesizedAnswer: string;
  errors: Array<any>;
}

// Define service response interfaces
interface PreprocessResponse {
  suggestions: Array<string>;
}

interface ProcessResponse {
  results: Array<any>;
  synthesizedAnswer: string;
  recommendations: Array<any>;
}

interface HistoryResponse {
  history: Array<any>;
}

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
  } as LegalProcessingContext,
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
    setQuery: assign({
      query: (_, event: any) => {
        if (event?.type === 'SUBMIT_QUERY') {
          return event.query;
        }
        return '';
      },
      userId: (_, event: any) => {
        if (event?.type === 'SUBMIT_QUERY') {
          return event.userId || null;
        }
        return null;
      }
    }),
    setDidYouMean: assign({
      didYouMean: (_, event: any) => {
        const data = event.data as PreprocessResponse;
        return data.suggestions || [];
      }
    }),
    setResults: assign({
      results: (_, event: any) => {
        const data = event.data as ProcessResponse;
        return data.results || [];
      },
      synthesizedAnswer: (_, event: any) => {
        const data = event.data as ProcessResponse;
        return data.synthesizedAnswer || '';
      },
      recommendations: (_, event: any) => {
        const data = event.data as ProcessResponse;
        return data.recommendations || [];
      }
    }),
    setError: assign({
      errors: (ctx: any, event: any) => [...(ctx.errors || []), event.data]
    }),
    clearErrors: assign({ errors: [] }),
    resetContext: assign({
      query: '',
      results: [],
      recommendations: [],
      didYouMean: [],
      errors: [],
      jobId: null,
      progress: 0,
      synthesizedAnswer: ''
    }),
    updateQuery: assign({
      query: (_, event: any) => {
        if (event?.type === 'REFINE') {
          return event.query;
        }
        return '';
      }
    }),
    setHistory: assign({
      // Handle history data if needed
    }),
    recordActivity: (args: any) => {
      const context = args.context as LegalProcessingContext;
      if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
        navigator.serviceWorker.ready.then(reg =>
          reg.active?.postMessage({
            type: 'RECORD_ACTIVITY',
            data: { userId: context.userId, query: context.query, timestamp: Date.now() }
          })
        );
      }
    }
  }
});

export default legalProcessingMachine;
