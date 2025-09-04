
// Legal AI Form State Machines with XState
// Advanced state management for legal document processing workflows

import { createMachine, assign, fromPromise, type ActorRefFrom } from "xstate";
import { z } from "zod";

// ============================================================================
// ZOD VALIDATION SCHEMAS
// ============================================================================

export const DocumentUploadSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255, 'Title too long'),
  description: z.string().optional(),
  documentType: z.enum(['contract', 'motion', 'brief', 'evidence', 'correspondence', 'statute', 'regulation', 'case_law', 'other']),
  jurisdiction: z.enum(['federal', 'state', 'local', 'international']).optional(),
  tags: z.array(z.string()).default([]),
  file: z.any().refine((file) => file instanceof File, 'File is required'),
  aiProcessing: z.object({
    generateSummary: z.boolean().default(true),
    extractEntities: z.boolean().default(true),
    riskAssessment: z.boolean().default(true),
    generateRecommendations: z.boolean().default(false)
  }).default({})
});

export const CaseCreationSchema = z.object({
  title: z.string().min(1, 'Case title is required').max(255, 'Title too long'),
  description: z.string().min(10, 'Description must be at least 10 characters').max(5000, 'Description too long'),
  caseNumber: z.string().optional(),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).default('medium'),
  status: z.enum(['active', 'pending', 'closed', 'archived']).default('active'),
  assignedTo: z.string().uuid().optional(),
  jurisdiction: z.enum(['federal', 'state', 'local', 'international']).optional(),
  tags: z.array(z.string()).default([]),
  estimatedDuration: z.number().min(1).max(365).optional(), // days
  budget: z.number().min(0).optional()
});

export const SearchQuerySchema = z.object({
  query: z.string().min(1, 'Search query is required').max(500, 'Query too long'),
  filters: z.object({
    documentTypes: z.array(z.string()).default([]),
    jurisdictions: z.array(z.string()).default([]),
    dateRange: z.object({
      from: z.date().optional(),
      to: z.date().optional()
    }).optional(),
    tags: z.array(z.string()).default([]),
    similarityThreshold: z.number().min(0).max(1).default(0.7),
    maxResults: z.number().min(1).max(100).default(20)
  }).default({}),
  useAI: z.boolean().default(true),
  cacheResults: z.boolean().default(true)
});

export const AIAnalysisSchema = z.object({
  documentId: z.string().uuid(),
  analysisType: z.enum(['summary', 'entities', 'risk', 'recommendations', 'precedents', 'compliance']),
  options: z.object({
    model: z.string().default('llama3.2'),
    temperature: z.number().min(0).max(2).default(0.7),
    maxTokens: z.number().min(100).max(8000).default(2000),
    includeConfidence: z.boolean().default(true),
    generateCitations: z.boolean().default(true)
  }).default({})
});

// ============================================================================
// STATE MACHINE CONTEXTS
// ============================================================================

export interface DocumentUploadContext {
  formData: z.infer<typeof DocumentUploadSchema> | null;
  validationErrors: Record<string, string[]>;
  uploadProgress: number;
  uploadedFile: any | null;
  processingProgress: number;
  aiResults: any | null;
  error: string | null;
  retryCount: number;
  maxRetries: number;
}

export interface CaseCreationContext {
  formData: z.infer<typeof CaseCreationSchema> | null;
  validationErrors: Record<string, string[]>;
  createdCase: any | null;
  relatedDocuments: any[];
  error: string | null;
  isAutoSaving: boolean;
  lastSaved: Date | null;
}

export interface SearchContext {
  query: z.infer<typeof SearchQuerySchema> | null;
  results: any[];
  validationErrors: Record<string, string[]>;
  isSearching: boolean;
  searchHistory: string[];
  filters: any;
  pagination: {
    page: number;
    pageSize: number;
    total: number;
  };
  analytics: {
    searchTime: number;
    resultCount: number;
    cacheHit: boolean;
  } | null;
  error: string | null;
}

export interface AIAnalysisContext {
  analysisData: z.infer<typeof AIAnalysisSchema> | null;
  validationErrors: Record<string, string[]>;
  analysisResults: any | null;
  confidence: number;
  processingTime: number;
  tokensUsed: number;
  model: string;
  error: string | null;
  isStreaming: boolean;
  streamedContent: string;
}

// ============================================================================
// DOCUMENT UPLOAD STATE MACHINE
// ============================================================================

export const documentUploadMachine = createMachine({
  /** @xstate-layout N4IgpgJg5mDOIC5QAoC2BDAxgCwJYDswBKAOhwRABtEJMAPVAQzAEsKwBiD3VAIwEcKAGwCuPAB4J4yHnT5cK8VBmwAlOvUQQAJBAAOEgBYJhIAB6JYugJYA2AEwAOSuYDsAGne0LLpWsun6gSCYoGDh4RCQUNAxEURBMFAAzMABrZFQpOXtZAGtUZCwsXDBwOXjMtOJyBBtEMzVLCyd9fXNzKe9fEECkELCIqJoFJEoKOOIU9MksHLzEotKyipq6hqam1vbOpOJOxBUXQdMlHWdpgx8-QOCQxEiqRFKKOKo1ZnXCwonc4rXYGOr6JQaKaqAy0cxOKb-YBBY4dCotZjHFZ3IhPTYvUQfXJ5JJ-f6A4F4sGeEG1OT+NqzZrwAImIA */
  id: 'documentUpload',
  initial: 'idle',
  context: {
    formData: null,
    validationErrors: {},
    uploadProgress: 0,
    uploadedFile: null,
    processingProgress: 0,
    aiResults: null,
    error: null,
    retryCount: 0,
    maxRetries: 3
  } as DocumentUploadContext,
  states: {
    idle: {
      on: {
        VALIDATE_FORM: {
          target: 'validating',
          actions: assign({
            formData: ({ event }) => event.data
          })
        }
      }
    },
    validating: {
      invoke: {
        id: 'validateDocumentForm',
        src: 'validateDocumentForm',
        input: ({ context }) => context.formData,
        onDone: {
          target: 'valid',
          actions: assign({
            validationErrors: () => ({})
          })
        },
        onError: {
          target: 'invalid',
          actions: assign({
            validationErrors: ({ event }) => ((event as any)?.error ?? {})
          })
        }
      }
    },
    invalid: {
      on: {
        VALIDATE_FORM: {
          target: 'validating',
          actions: assign({
            formData: ({ event }) => event.data
          })
        },
        RESET: 'idle'
      }
    },
    valid: {
      on: {
        UPLOAD: 'uploading',
        VALIDATE_FORM: {
          target: 'validating',
          actions: assign({
            formData: ({ event }) => event.data
          })
        }
      }
    },
    uploading: {
      invoke: {
        id: 'uploadDocument',
        src: 'uploadDocument',
        input: ({ context }) => context.formData,
        onDone: {
          target: 'uploaded',
          actions: assign({
            uploadedFile: ({ event }) => (event.output as any) ?? null,
            uploadProgress: () => 100
          })
        },
        onError: {
          target: 'uploadError',
          actions: assign({
            error: ({ event }) => ((event as any)?.error?.message ?? String((event as any)?.error))
          })
        }
      },
      on: {
        UPLOAD_PROGRESS: {
          actions: assign({
            uploadProgress: ({ event }) => event.progress
          })
        }
      }
    },
    uploaded: {
      always: [
        {
          target: 'processing',
          guard: ({ context }) => context.formData?.aiProcessing.generateSummary ||
            context.formData?.aiProcessing.extractEntities ||
                                  context.formData?.aiProcessing.riskAssessment
        },
        { target: 'completed' }
      ]
    },
    processing: {
      invoke: {
        id: 'processDocument',
        src: 'processDocument',
        input: ({ context }) => ({
          documentId: context.uploadedFile?.id,
          options: context.formData?.aiProcessing
        }),
        onDone: {
          target: 'completed',
          actions: assign({
            aiResults: ({ event }) => (event.output as any) ?? null,
            processingProgress: () => 100
          })
        },
        onError: {
          target: 'processingError',
          actions: assign({
            error: ({ event }) => ((event as any)?.error?.message ?? String((event as any)?.error))
          })
        }
      },
      on: {
        PROCESSING_PROGRESS: {
          actions: assign({
            processingProgress: ({ event }) => event.progress
          })
        }
      }
    },
    uploadError: {
      on: {
        RETRY: [
          {
            target: 'uploading',
            guard: ({ context }) => context.retryCount < context.maxRetries,
            actions: assign({
              retryCount: ({ context }) => context.retryCount + 1,
              error: () => null
            })
          },
          { target: 'failed' }
        ],
        RESET: 'idle'
      }
    },
    processingError: {
      on: {
        RETRY: [
          {
            target: 'processing',
            guard: ({ context }) => context.retryCount < context.maxRetries,
            actions: assign({
              retryCount: ({ context }) => context.retryCount + 1,
              error: () => null
            })
          },
          { target: 'failed' }
        ],
        SKIP_PROCESSING: 'completed',
        RESET: 'idle'
      }
    },
    completed: {
      on: {
        RESET: 'idle',
        NEW_UPLOAD: 'idle'
      }
    },
    failed: {
      on: {
        RESET: 'idle'
      }
    }
  }
}, {
  actors: {
    validateDocumentForm: fromPromise(async ({ input }) => {
      try {
        DocumentUploadSchema.parse(input);
        return true;
      } catch (error: any) {
        if (error instanceof z.ZodError) {
          throw error.flatten().fieldErrors;
        }
        throw error;
      }
    }),
    uploadDocument: fromPromise(async ({ input }) => {
      // Mock upload implementation
      const formData = new FormData();
      Object.entries(input || {}).forEach(([key, value]) => {
        if (key === 'file') {
          formData.append('file', value);
        } else if (typeof value === 'object') {
          formData.append(key, JSON.stringify(value));
        } else {
          formData.append(key, String(value));
        }
      });

      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      return await response.json();
    }),
    processDocument: fromPromise(async ({ input }) => {
      const response = await fetch('/api/ai/process-document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });

      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }

      return await response.json();
    })
  }
});

// ============================================================================
// CASE CREATION STATE MACHINE
// ============================================================================

export const caseCreationMachine = createMachine({
  id: 'caseCreation',
  initial: 'idle',
  context: {
    formData: null,
    validationErrors: {},
    createdCase: null,
    relatedDocuments: [],
    error: null,
    isAutoSaving: false,
    lastSaved: null
  } as CaseCreationContext,
  states: {
    idle: {
      on: {
        START_CREATION: 'creating',
        LOAD_DRAFT: 'loadingDraft'
      }
    },
    loadingDraft: {
      invoke: {
        id: 'loadDraft',
        src: 'loadDraft',
        onDone: {
          target: 'editing',
          actions: assign({
            formData: ({ event }) => event.output
          })
        },
        onError: 'creating'
      }
    },
    creating: {
      on: {
        UPDATE_FORM: {
          target: 'editing',
          actions: assign({
            formData: ({ event }) => event.data
          })
        }
      }
    },
    editing: {
      on: {
        UPDATE_FORM: {
          actions: assign({
            formData: ({ event }) => event.data
          })
        },
        AUTO_SAVE: 'autoSaving',
        VALIDATE: 'validating',
        SUBMIT: 'validating'
      },
      after: {
        5000: 'autoSaving' // Auto-save every 5 seconds
      }
    },
    autoSaving: {
      invoke: {
        id: 'autoSave',
        src: 'autoSave',
        input: ({ context }) => context.formData,
        onDone: {
          target: 'editing',
          actions: assign({
            lastSaved: () => new Date(),
            isAutoSaving: () => false
          })
        },
        onError: {
          target: 'editing',
          actions: assign({
            isAutoSaving: () => false
          })
        }
      },
      entry: assign({
        isAutoSaving: () => true
      })
    },
    validating: {
      invoke: {
        id: 'validateCase',
        src: 'validateCase',
        input: ({ context }) => context.formData,
        onDone: 'submitting',
        onError: {
          target: 'editing',
          actions: assign({
            validationErrors: ({ event }) => event.error
          })
        }
      }
    },
    submitting: {
      invoke: {
        id: 'createCase',
        src: 'createCase',
        input: ({ context }) => context.formData,
        onDone: {
          target: 'completed',
          actions: assign({
            createdCase: ({ event }) => event.output
          })
        },
        onError: {
          target: 'editing',
          actions: assign({
            error: ({ event }) => event.error.message
          })
        }
      }
    },
    completed: {
      on: {
        NEW_CASE: 'idle',
        EDIT_CASE: 'editing'
      }
    }
  }
}, {
  actors: {
    loadDraft: fromPromise(async () => {
      // Load draft from localStorage or API
      const draft = localStorage.getItem('case-draft');
      return draft ? JSON.parse(draft) : null;
    }),
    autoSave: fromPromise(async ({ input }) => {
      // Auto-save to localStorage
      localStorage.setItem('case-draft', JSON.stringify(input));
      return true;
    }),
    validateCase: fromPromise(async ({ input }) => {
      try {
        CaseCreationSchema.parse(input);
        return true;
      } catch (error: any) {
        if (error instanceof z.ZodError) {
          throw error.flatten().fieldErrors;
        }
        throw error;
      }
    }),
    createCase: fromPromise(async ({ input }) => {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });

      if (!response.ok) {
        throw new Error(`Case creation failed: ${response.statusText}`);
      }

      return await response.json();
    })
  }
});

// ============================================================================
// SEARCH STATE MACHINE
// ============================================================================

export const searchMachine = createMachine({
  id: 'search',
  initial: 'idle',
  context: {
    query: null,
    results: [],
    validationErrors: {},
    isSearching: false,
    searchHistory: [],
    filters: {},
    pagination: {
      page: 1,
      pageSize: 20,
      total: 0
    },
    analytics: null,
    error: null
  } as SearchContext,
  states: {
    idle: {
      on: {
        SEARCH: 'validating',
        LOAD_HISTORY: 'loadingHistory'
      }
    },
    loadingHistory: {
      invoke: {
        id: 'loadSearchHistory',
        src: 'loadSearchHistory',
        onDone: {
          target: 'idle',
          actions: assign({
            searchHistory: ({ event }) => event.output
          })
        },
        onError: 'idle'
      }
    },
    validating: {
      invoke: {
        id: 'validateSearch',
        src: 'validateSearch',
        input: ({ context }) => context.query,
        onDone: 'searching',
        onError: {
          target: 'idle',
          actions: assign({
            validationErrors: ({ event }) => event.error
          })
        }
      }
    },
    searching: {
      invoke: {
        id: 'performSearch',
        src: 'performSearch',
        input: ({ context }) => context.query,
        onDone: {
          target: 'results',
          actions: assign({
            results: ({ event }) => event.output.results,
            analytics: ({ event }) => event.output.analytics,
            pagination: ({ event }) => event.output.pagination,
            searchHistory: ({ context, event }) => [
              event.output.query,
              ...context.searchHistory.filter((q: any) => q !== event.output.query)
            ].slice(0, 10)
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => event.error.message
          })
        }
      },
      entry: assign({
        isSearching: () => true,
        results: () => []
      }),
      exit: assign({
        isSearching: () => false
      })
    },
    results: {
      on: {
        SEARCH: {
          target: 'validating',
          actions: assign({
            query: ({ event }) => event.data
          })
        },
        REFINE_SEARCH: 'validating',
        CLEAR_RESULTS: 'idle',
        LOAD_MORE: 'loadingMore'
      }
    },
    loadingMore: {
      invoke: {
        id: 'loadMoreResults',
        src: 'loadMoreResults',
        input: ({ context }) => ({
          query: context.query,
          page: context.pagination.page + 1
        }),
        onDone: {
          target: 'results',
          actions: assign({
            results: ({ context, event }) => [...context.results, ...event.output.results],
            pagination: ({ event }) => event.output.pagination
          })
        },
        onError: 'results'
      }
    },
    error: {
      on: {
        RETRY: 'searching',
        NEW_SEARCH: 'idle'
      }
    }
  }
}, {
  actors: {
    loadSearchHistory: fromPromise(async () => {
      const history = localStorage.getItem('search-history');
      return history ? JSON.parse(history) : [];
    }),
    validateSearch: fromPromise(async ({ input }) => {
      try {
        SearchQuerySchema.parse(input);
        return true;
      } catch (error: any) {
        if (error instanceof z.ZodError) {
          throw error.flatten().fieldErrors;
        }
        throw error;
      }
    }),
    performSearch: fromPromise(async ({ input }) => {
      const response = await fetch('/api/search/vector', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();

      // Save to history
      const history = JSON.parse(localStorage.getItem('search-history') || '[]');
      const updatedHistory = [input.query, ...history.filter((q: string) => q !== input.query)].slice(0, 10);
      localStorage.setItem('search-history', JSON.stringify(updatedHistory));

      return data;
    }),
    loadMoreResults: fromPromise(async ({ input }) => {
      const response = await fetch('/api/search/vector', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...input.query,
          pagination: { page: input.page }
        })
      });

      if (!response.ok) {
        throw new Error(`Load more failed: ${response.statusText}`);
      }

      return await response.json();
    })
  }
});

// ============================================================================
// AI ANALYSIS STATE MACHINE
// ============================================================================

export const aiAnalysisMachine = createMachine({
  id: 'aiAnalysis',
  initial: 'idle',
  context: {
    analysisData: null,
    validationErrors: {},
    analysisResults: null,
    confidence: 0,
    processingTime: 0,
    tokensUsed: 0,
    model: 'llama3.2',
    error: null,
    isStreaming: false,
    streamedContent: ''
  } as AIAnalysisContext,
  states: {
    idle: {
      on: {
        START_ANALYSIS: 'validating'
      }
    },
    validating: {
      invoke: {
        id: 'validateAnalysis',
        src: 'validateAnalysis',
        input: ({ context }) => context.analysisData,
        onDone: 'analyzing',
        onError: {
          target: 'idle',
          actions: assign({
            validationErrors: ({ event }) => event.error
          })
        }
      }
    },
    analyzing: {
      invoke: {
        id: 'performAnalysis',
        src: 'performAnalysis',
        input: ({ context }) => context.analysisData,
        onDone: {
          target: 'completed',
          actions: assign({
            analysisResults: ({ event }) => event.output.results,
            confidence: ({ event }) => event.output.confidence,
            processingTime: ({ event }) => event.output.processingTime,
            tokensUsed: ({ event }) => event.output.tokensUsed
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => event.error.message
          })
        }
      },
      on: {
        STREAM_CONTENT: {
          actions: assign({
            streamedContent: ({ context, event }) => context.streamedContent + event.content,
            isStreaming: () => true
          })
        }
      }
    },
    completed: {
      on: {
        NEW_ANALYSIS: 'idle',
        RETRY_ANALYSIS: 'analyzing'
      }
    },
    error: {
      on: {
        RETRY: 'analyzing',
        NEW_ANALYSIS: 'idle'
      }
    }
  }
}, {
  actors: {
    validateAnalysis: fromPromise(async ({ input }) => {
      try {
        AIAnalysisSchema.parse(input);
        return true;
      } catch (error: any) {
        if (error instanceof z.ZodError) {
          throw error.flatten().fieldErrors;
        }
        throw error;
      }
    }),
    performAnalysis: fromPromise(async ({ input }) => {
      const startTime = Date.now();

      const response = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        ...data,
        processingTime: Date.now() - startTime
      };
    })
  }
});

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type DocumentUploadMachine = typeof documentUploadMachine;
export type CaseCreationMachine = typeof caseCreationMachine;
export type SearchMachine = typeof searchMachine;
export type AIAnalysisMachine = typeof aiAnalysisMachine;

export type DocumentUploadActor = ActorRefFrom<DocumentUploadMachine>;
export type CaseCreationActor = ActorRefFrom<CaseCreationMachine>;
export type SearchActor = ActorRefFrom<SearchMachine>;
export type AIAnalysisActor = ActorRefFrom<AIAnalysisMachine>;

// Schema exports
export {
  DocumentUploadSchema,
  CaseCreationSchema,
  SearchQuerySchema,
  AIAnalysisSchema
};