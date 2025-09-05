/**
 * Legal Case Management State Machine
 * Comprehensive XState machine for managing legal case workflows
 */
import { createMachine, assign } from 'xstate';
import type { Case, Evidence, NewCase, NewEvidence } from '../server/db/schema-types';
import { aiSummarizationService } from '../services/ai-summarization-service';
import { vectorSearchService } from '../services/vector-search-service';

// Context types
export interface LegalCaseContext {
  // Case data
  case: Case | null;
  caseId: string | null;

  // Evidence management
  evidence: Evidence[];
  selectedEvidence: Evidence | null;
  uploadQueue: File[];

  // AI processing
  aiAnalysisProgress: number;
  aiSummary: string | null;
  similarCases: Array<{ id: string; title: string; similarity: number }>;

  // Search and filtering
  searchQuery: string;
  searchResults: any[];
  filters: {
    evidenceType?: string;
    dateRange?: { start: Date; end: Date };
    tags?: string[];
    isAdmissible?: boolean;
  };

  // UI state
  activeTab: 'overview' | 'evidence' | 'analysis' | 'search';
  isLoading: boolean;
  error: string | null;

  // Form data
  formData: {
    caseForm: Partial<NewCase>;
    evidenceForm: Partial<NewEvidence>;
  };

  // Workflow state
  workflowStage: 'investigation' | 'analysis' | 'preparation' | 'review' | 'closed';
  nextActions: string[];

  // Collaboration
  collaborators: Array<{ id: string; name: string; role: string }>;
  notifications: Array<{ id: string; message: string; type: 'info' | 'warning' | 'error' }>;

  // Performance tracking
  stats: {
    totalEvidence: number;
    processedEvidence: number;
    averageConfidence: number;
    processingTime: number;
  };
}

// Event types
export type LegalCaseEvents =
  | { type: 'LOAD_CASE'; caseId: string }
  | { type: 'CREATE_CASE'; caseData: NewCase }
  | { type: 'UPDATE_CASE'; updates: Partial<Case> }
  | { type: 'DELETE_CASE' }

  // Evidence events
  | { type: 'ADD_EVIDENCE'; files: File[] }
  | { type: 'SELECT_EVIDENCE'; evidence: Evidence }
  | { type: 'DELETE_EVIDENCE'; evidenceId: string }
  | { type: 'PROCESS_EVIDENCE'; evidenceId: string }

  // AI events
  | { type: 'START_AI_ANALYSIS' }
  | { type: 'AI_ANALYSIS_PROGRESS'; progress: number }
  | { type: 'AI_ANALYSIS_COMPLETE'; summary: string }
  | { type: 'FIND_SIMILAR_CASES' }
  | { type: 'GENERATE_RECOMMENDATIONS' }

  // Search events
  | { type: 'SEARCH'; query: string }
  | { type: 'APPLY_FILTERS'; filters: LegalCaseContext['filters'] }
  | { type: 'CLEAR_SEARCH' }

  // Navigation events
  | { type: 'SWITCH_TAB'; tab: LegalCaseContext['activeTab'] }
  | { type: 'SET_WORKFLOW_STAGE'; stage: LegalCaseContext['workflowStage'] }

  // Form events
  | { type: 'UPDATE_CASE_FORM'; data: Partial<NewCase> }
  | { type: 'UPDATE_EVIDENCE_FORM'; data: Partial<NewEvidence> }
  | { type: 'SUBMIT_CASE_FORM' }
  | { type: 'SUBMIT_EVIDENCE_FORM' }
  | { type: 'RESET_FORMS' }

  // Error handling
  | { type: 'RETRY' }
  | { type: 'DISMISS_ERROR' }

  // Generic events
  | { type: 'REFRESH' }
  | { type: 'RESET' };

// === Services (async operations) ===
// XState expects functions of the form (context, event) => Promise<any>
// below we expose functions that return promises; when invoked by the machine we pass them directly

const loadCaseService = async (_context: LegalCaseContext, event: any): Promise<any> => {
  const caseId = event?.caseId;
  if (!caseId) throw new Error('Missing caseId');
  const response = await fetch(`/api/cases/${caseId}`);
  if (!response.ok) throw new Error('Failed to load case');
  return await response.json();
};

const createCaseService = async (context: LegalCaseContext): Promise<any> => {
  const response = await fetch('/api/cases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(context.formData.caseForm)
  });
  if (!response.ok) throw new Error('Failed to create case');
  return await response.json();
};

const loadEvidenceService = async (_context: LegalCaseContext, event: any): Promise<any> => {
  const caseId = event?.caseId ?? _context.caseId;
  if (!caseId) throw new Error('Missing caseId for evidence load');
  const response = await fetch(`/api/cases/${caseId}/evidence`);
  if (!response.ok) throw new Error('Failed to load evidence');
  return await response.json();
};

const processEvidenceService = async (_context: LegalCaseContext, event: any): Promise<any> => {
  const evidenceId = event?.evidenceId ?? _context?.selectedEvidence?.id;
  if (!evidenceId) throw new Error('Missing evidenceId for processing');
  const result = await aiSummarizationService.summarizeEvidence(evidenceId);
  return result;
};

const findSimilarCasesService = async (_context: LegalCaseContext, event: any): Promise<any> => {
  const caseId = event?.caseId ?? _context.caseId;
  if (!caseId) throw new Error('Missing caseId for similarity search');
  const similarDocs = await vectorSearchService.findSimilarDocuments(caseId, {
    limit: 5,
    threshold: 0.7
  });
  return similarDocs;
};

const searchService = async (_context: LegalCaseContext, event: any): Promise<any> => {
  const query = event?.query ?? '';
  const results = await vectorSearchService.search({
    query,
    filters: (_context as any).filters,
    options: { limit: 20 }
  });
  return results;
};

// === Guards ===
const isValidCaseData = ({ context }: { context: LegalCaseContext }) => {
  const { caseForm } = context.formData;
  return !!(caseForm && (caseForm as any).title && (caseForm as any).description && (caseForm as any).caseNumber);
};

const hasEvidence = ({ context }: { context: LegalCaseContext }) => {
  return Array.isArray(context.evidence) && context.evidence.length > 0;
};

const hasAIAnalysis = ({ context }: { context: LegalCaseContext }) => {
  return !!context.aiSummary;
};

// === Actions (assign helpers) ===
// note: event.data is used in onDone handlers

const assignCaseData = assign({
  case: (_ctx, event: any) => (event?.data ?? null),
  caseId: (_ctx, event: any) => (event?.data?.id ?? null),
  isLoading: () => false,
  error: () => null
});

const assignEvidence = assign({
  evidence: (_ctx, event: any) => (event?.data ?? []),
  stats: (context: LegalCaseContext, event: any) => ({
    ...context.stats,
    totalEvidence: Array.isArray(event?.data) ? event.data.length : context.stats.totalEvidence,
    processedEvidence: Array.isArray(event?.data)
      ? event.data.filter((e: any) => !!e.aiSummary).length
      : context.stats.processedEvidence
  })
});

const assignSearchResults = assign({
  searchResults: (_ctx, event: any) => ((event?.data?.results ?? []) as any[]),
  searchQuery: (_ctx, event: any) => ((event?.query ?? '') as string)
});

const assignError = assign({
  error: (_ctx, event: any) => ((event?.data?.message ?? String(event?.data ?? event?.error ?? 'An error occurred')) as string),
  isLoading: () => false
});

const setLoading = assign({
  isLoading: () => true,
  error: () => null
});

const clearError = assign({
  error: () => null
});

const updateFormData = assign({
  formData: (context: LegalCaseContext, event: any) => ({
    ...context.formData,
    caseForm: { ...context.formData.caseForm, ...(event.data ?? {}) }
  })
});

const switchTab = assign({
  activeTab: (_ctx, event: any) => event.tab
});

const updateWorkflowStage = assign({
  workflowStage: (_ctx, event: any) => event.stage,
  nextActions: (_ctx, event: any) => {
    const nextActionsMap: Record<string, string[]> = {
      investigation: ['Collect evidence', 'Interview witnesses', 'Review documents'],
      analysis: ['Analyze evidence', 'Generate AI summary', 'Find precedents'],
      preparation: ['Prepare legal briefs', 'Organize evidence', 'Plan strategy'],
      review: ['Final review', 'Quality check', 'Prepare for court'],
      closed: ['Archive case', 'Generate reports', 'Post-case analysis']
    };
    return (nextActionsMap as any)[event.stage] || [];
  }
});

const assignAIProgress = assign({
  aiAnalysisProgress: (_ctx, event: any) => (event?.progress ?? 0)
});

const assignAISummary = assign({
  aiSummary: (_ctx, event: any) => ((event?.data?.summary ?? null) as string | null),
  aiAnalysisProgress: () => 100,
  stats: (context: LegalCaseContext, event: any) => ({
    ...context.stats,
    averageConfidence: (event?.data?.confidence ?? context.stats.averageConfidence),
    processingTime: (event?.data?.processingTime ?? context.stats.processingTime)
  })
});

const assignSimilarCases = assign({
  similarCases: (_ctx, event: any) => (event?.data ?? [])
});

// === Main state machine ===
export const legalCaseMachine = createMachine<LegalCaseContext, LegalCaseEvents>({
  id: 'legalCase',
  context: {
    case: null,
    caseId: null,
    evidence: [],
    selectedEvidence: null,
    uploadQueue: [],
    aiAnalysisProgress: 0,
    aiSummary: null,
    similarCases: [],
    searchQuery: '',
    searchResults: [],
    filters: {},
    activeTab: 'overview',
    isLoading: false,
    error: null,
    formData: {
      caseForm: {},
      evidenceForm: {}
    },
    workflowStage: 'investigation',
    nextActions: ['Collect evidence', 'Interview witnesses', 'Review documents'],
    collaborators: [],
    notifications: [],
    stats: {
      totalEvidence: 0,
      processedEvidence: 0,
      averageConfidence: 0,
      processingTime: 0
    }
  },
  initial: 'idle',
  states: {
    idle: {
      on: {
        LOAD_CASE: {
          target: 'loadingCase',
          actions: 'setLoading'
        },
        CREATE_CASE: {
          target: 'creatingCase',
          cond: 'isValidCaseData',
          actions: 'setLoading'
        },
        SEARCH: {
          target: 'searching',
          actions: 'setLoading'
        },
        SWITCH_TAB: {
          actions: 'switchTab'
        }
      }
    },

    loadingCase: {
      invoke: {
        id: 'loadCase',
        src: loadCaseService,
        // onDone gets event.data
        onDone: {
          target: 'caseLoaded',
          actions: 'assignCaseData'
        },
        onError: {
          target: 'error',
          actions: 'assignError'
        }
      }
    },

    creatingCase: {
      invoke: {
        id: 'createCase',
        src: createCaseService,
        onDone: {
          target: 'caseLoaded',
          actions: [
            'assignCaseData',
            assign({ formData: () => ({ caseForm: {}, evidenceForm: {} }) })
          ]
        },
        onError: {
          target: 'error',
          actions: 'assignError'
        }
      }
    },

    caseLoaded: {
      initial: 'loadingEvidence',
      entry: [
        assign({ isLoading: () => false })
      ],
      states: {
        loadingEvidence: {
          invoke: {
            id: 'loadEvidence',
            src: loadEvidenceService,
            onDone: {
              target: 'ready',
              actions: 'assignEvidence'
            },
            onError: {
              target: 'ready',
              actions: 'assignError'
            }
          }
        },

        ready: {
          on: {
            ADD_EVIDENCE: {
              target: 'uploadingEvidence',
              actions: [
                'setLoading',
                assign({
                  uploadQueue: (_ctx, event: any) => (event.files ?? [])
                })
              ]
            },
            PROCESS_EVIDENCE: {
              target: 'processingEvidence',
              actions: 'setLoading'
            },
            START_AI_ANALYSIS: {
              target: 'aiAnalysis',
              actions: 'setLoading',
              cond: 'hasEvidence'
            },
            FIND_SIMILAR_CASES: {
              target: 'findingSimilarCases',
              actions: 'setLoading'
            },
            UPDATE_CASE: {
              target: 'updatingCase',
              actions: 'setLoading'
            },
            DELETE_CASE: {
              target: 'deletingCase',
              actions: 'setLoading'
            }
          }
        },

        uploadingEvidence: {
          invoke: {
            id: 'uploadEvidence',
            src: async (context: LegalCaseContext) => {
              const formData = new FormData();
              (context.uploadQueue || []).forEach((file: any) => formData.append('files', file));
              formData.append('caseId', context.caseId ?? '');

              const response = await fetch('/api/evidence/upload', {
                method: 'POST',
                body: formData
              });

              if (!response.ok) throw new Error('Upload failed');
              return await response.json();
            },
            onDone: {
              target: 'loadingEvidence',
              actions: [
                assign({ uploadQueue: () => [] }),
                assign({
                  notifications: (ctx) => [
                    ...ctx.notifications,
                    {
                      id: Date.now().toString(),
                      message: 'Evidence uploaded successfully',
                      type: 'info' as const
                    }
                  ]
                })
              ]
            },
            onError: {
              target: 'ready',
              actions: 'assignError'
            }
          }
        },

        processingEvidence: {
          invoke: {
            id: 'processEvidence',
            src: processEvidenceService,
            onDone: {
              target: 'ready',
              actions: [
                assign({ isLoading: () => false }),
                assign({
                  evidence: (ctx, event: any) =>
                    ctx.evidence.map((e: any) =>
                      e.id === (ctx.selectedEvidence?.id ?? event?.data?.id)
                        ? { ...e, aiSummary: event?.data?.summary ?? e.aiSummary }
                        : e
                    )
                })
              ]
            },
            onError: {
              target: 'ready',
              actions: 'assignError'
            }
          }
        },

        aiAnalysis: {
          initial: 'analyzing',
          states: {
            analyzing: {
              invoke: {
                id: 'aiSummarizeCase',
                src: async (context: LegalCaseContext) => {
                  if (!context.caseId) throw new Error('Missing caseId for AI analysis');
                  return await aiSummarizationService.summarizeCase(context.caseId);
                },
                onDone: {
                  target: 'complete',
                  actions: 'assignAISummary'
                },
                onError: {
                  target: '#legalCase.caseLoaded.ready',
                  actions: 'assignError'
                }
              },
              on: {
                AI_ANALYSIS_PROGRESS: {
                  actions: 'assignAIProgress'
                }
              }
            },

            complete: {
              entry: [
                assign({ isLoading: () => false }),
                assign({
                  notifications: (ctx) => [
                    ...ctx.notifications,
                    {
                      id: Date.now().toString(),
                      message: 'AI analysis completed',
                      type: 'info' as const
                    }
                  ]
                })
              ],
              after: {
                1000: {
                  target: '#legalCase.caseLoaded.ready'
                }
              }
            }
          }
        },

        findingSimilarCases: {
          invoke: {
            id: 'findSimilarCases',
            src: findSimilarCasesService,
            onDone: {
              target: 'ready',
              actions: [
                assign({ isLoading: () => false }),
                'assignSimilarCases'
              ]
            },
            onError: {
              target: 'ready',
              actions: 'assignError'
            }
          }
        },

        updatingCase: {
          invoke: {
            id: 'updateCase',
            src: async (context: LegalCaseContext, event: any) => {
              const caseId = context.caseId;
              if (!caseId) throw new Error('Missing caseId for update');
              const response = await fetch(`/api/cases/${caseId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(event.updates)
              });
              if (!response.ok) throw new Error('Update failed');
              return await response.json();
            },
            onDone: {
              target: 'ready',
              actions: [
                assign({ isLoading: () => false }),
                'assignCaseData'
              ]
            },
            onError: {
              target: 'ready',
              actions: 'assignError'
            }
          }
        },

        deletingCase: {
          invoke: {
            id: 'deleteCase',
            src: async (context: LegalCaseContext) => {
              if (!context.caseId) throw new Error('Missing caseId for delete');
              const response = await fetch(`/api/cases/${context.caseId}`, {
                method: 'DELETE'
              });
              if (!response.ok) throw new Error('Delete failed');
              return true;
            },
            onDone: {
              target: '#legalCase.idle',
              actions: assign({
                case: () => null,
                caseId: () => null,
                evidence: () => [],
                selectedEvidence: () => null,
                aiSummary: () => null,
                similarCases: () => [],
                isLoading: () => false
              })
            },
            onError: {
              target: 'ready',
              actions: 'assignError'
            }
          }
        }
      },

      on: {
        SWITCH_TAB: {
          actions: 'switchTab'
        },
        SET_WORKFLOW_STAGE: {
          actions: 'updateWorkflowStage'
        },
        UPDATE_CASE_FORM: {
          actions: 'updateFormData'
        },
        SELECT_EVIDENCE: {
          actions: assign({
            selectedEvidence: (_ctx, event: any) => event.evidence
          })
        },
        APPLY_FILTERS: {
          actions: assign({
            filters: (_ctx, event: any) => event.filters
          })
        },
        REFRESH: {
          target: '.loadingEvidence'
        }
      }
    },

    searching: {
      invoke: {
        id: 'search',
        src: searchService,
        onDone: {
          target: 'idle',
          actions: [
            assign({ isLoading: () => false }),
            'assignSearchResults'
          ]
        },
        onError: {
          target: 'error',
          actions: 'assignError'
        }
      }
    },

    error: {
      entry: assign({ isLoading: () => false }),
      on: {
        RETRY: {
          target: 'idle',
          actions: 'clearError'
        },
        DISMISS_ERROR: {
          actions: 'clearError'
        }
      }
    }
  },

  on: {
    RESET: {
      target: 'idle',
      actions: assign(() => ({
        case: null,
        caseId: null,
        evidence: [],
        selectedEvidence: null,
        uploadQueue: [],
        aiAnalysisProgress: 0,
        aiSummary: null,
        similarCases: [],
        searchQuery: '',
        searchResults: [],
        filters: {},
        activeTab: 'overview',
        isLoading: false,
        error: null,
        formData: {
          caseForm: {},
          evidenceForm: {}
        },
        workflowStage: 'investigation',
        nextActions: ['Collect evidence', 'Interview witnesses', 'Review documents'],
        notifications: [],
        stats: {
          totalEvidence: 0,
          processedEvidence: 0,
          averageConfidence: 0,
          processingTime: 0
        }
      }))
    }
  }
},
{
  // machine implementations: guards and action mappings
  guards: {
    isValidCaseData,
    hasEvidence,
    hasAIAnalysis
  },
  actions: {
    assignCaseData,
    assignEvidence,
    assignSearchResults,
    assignError,
    setLoading,
    clearError,
    updateFormData,
    switchTab,
    updateWorkflowStage,
    assignAIProgress,
    assignAISummary,
    assignSimilarCases
  }
});

// Selector functions for accessing state
export const legalCaseSelectors = {
  isLoading: (state: any) => state.context.isLoading,
  hasError: (state: any) => !!state.context.error,
  getCurrentCase: (state: any) => state.context.case,
  getEvidence: (state: any) => state.context.evidence,
  getAISummary: (state: any) => state.context.aiSummary,
  getSimilarCases: (state: any) => state.context.similarCases,
  getSearchResults: (state: any) => state.context.searchResults,
  getActiveTab: (state: any) => state.context.activeTab,
  getWorkflowStage: (state: any) => state.context.workflowStage,
  getNextActions: (state: any) => state.context.nextActions,
  getStats: (state: any) => state.context.stats,
  canStartAIAnalysis: (state: any) => hasEvidence({ context: state.context }),
  isInState: (stateName: string) => (state: any) => state.matches(stateName)
};