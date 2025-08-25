
/**
 * Legal Case Management State Machine
 * Comprehensive XState machine for managing legal case workflows
 */
import { createMachine, assign, fromPromise } from 'xstate';
import type { Case, Evidence, NewCase, NewEvidence } from '../server/db/schema-types.js';
import { aiSummarizationService } from '../services/ai-summarization-service.js';
import { vectorSearchService } from '../services/vector-search-service.js';

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

// Services (async operations)
const loadCaseService = fromPromise(async ({ input }: { input: { caseId: string } }) => {
  // Implementation would load case from database
  const response = await fetch(`/api/cases/${input.caseId}`);
  if (!response.ok) throw new Error('Failed to load case');
  return await response.json();
});

const createCaseService = fromPromise(async ({ input }: { input: { caseData: NewCase } }) => {
  const response = await fetch('/api/cases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input.caseData)
  });
  if (!response.ok) throw new Error('Failed to create case');
  return await response.json();
});

const loadEvidenceService = fromPromise(async ({ input }: { input: { caseId: string } }) => {
  const response = await fetch(`/api/cases/${input.caseId}/evidence`);
  if (!response.ok) throw new Error('Failed to load evidence');
  return await response.json();
});

const processEvidenceService = fromPromise(async ({ input }: { input: { evidenceId: string } }) => {
  const result = await aiSummarizationService.summarizeEvidence(input.evidenceId);
  return result;
});

const findSimilarCasesService = fromPromise(async ({ input }: { input: { caseId: string } }) => {
  const similarDocs = await vectorSearchService.findSimilarDocuments(input.caseId, {
    limit: 5,
    threshold: 0.7
  });
  return similarDocs;
});

const searchService = fromPromise(async ({ input }: { input: { query: string; filters: any } }) => {
  const results = await vectorSearchService.search({
    query: input.query,
    filters: input.filters,
    options: { limit: 20 }
  });
  return results;
});

// Guards
const isValidCaseData = ({ context }: { context: LegalCaseContext }) => {
  const { caseForm } = context.formData;
  return !!(caseForm.title && caseForm.description && caseForm.caseNumber);
};

const hasEvidence = ({ context }: { context: LegalCaseContext }) => {
  return context.evidence.length > 0;
};

const hasAIAnalysis = ({ context }: { context: LegalCaseContext }) => {
  return !!context.aiSummary;
};

// Actions
const assignCaseData = assign({
  case: ({ event }: { event: any }) => event.output,
  caseId: ({ event }: { event: any }) => event.output.id,
  isLoading: false,
  error: null
});

const assignEvidence = assign({
  evidence: ({ event }: { event: any }) => event.output,
  stats: ({ event, context }: { event: any; context: LegalCaseContext }) => ({
    ...context.stats,
    totalEvidence: event.output.length,
    processedEvidence: event.output.filter((e: Evidence) => e.aiSummary).length
  })
});

const assignSearchResults = assign({
  searchResults: ({ event }: { event: any }) => event.output.results,
  searchQuery: ({ event }: { event: any }) => event.input.query
});

const assignError = assign({
  error: ({ event }: { event: any }) => event.error?.message || 'An error occurred',
  isLoading: false
});

const setLoading = assign({
  isLoading: true,
  error: null
});

const clearError = assign({
  error: null
});

const updateFormData = assign({
  formData: ({ context, event }: { context: LegalCaseContext; event: any }) => ({
    ...context.formData,
    caseForm: { ...context.formData.caseForm, ...event.data }
  })
});

const switchTab = assign({
  activeTab: ({ event }: { event: any }) => event.tab
});

const updateWorkflowStage = assign({
  workflowStage: ({ event }: { event: any }) => event.stage,
  nextActions: ({ event }: { event: any }) => {
    // Generate next actions based on workflow stage
    const nextActionsMap = {
      investigation: ['Collect evidence', 'Interview witnesses', 'Review documents'],
      analysis: ['Analyze evidence', 'Generate AI summary', 'Find precedents'],
      preparation: ['Prepare legal briefs', 'Organize evidence', 'Plan strategy'],
      review: ['Final review', 'Quality check', 'Prepare for court'],
      closed: ['Archive case', 'Generate reports', 'Post-case analysis']
    };
    return nextActionsMap[event.stage] || [];
  }
});

const assignAIProgress = assign({
  aiAnalysisProgress: ({ event }: { event: any }) => event.progress
});

const assignAISummary = assign({
  aiSummary: ({ event }: { event: any }) => event.output.summary,
  aiAnalysisProgress: 100,
  stats: ({ context, event }: { context: LegalCaseContext; event: any }) => ({
    ...context.stats,
    averageConfidence: event.output.confidence,
    processingTime: event.output.processingTime
  })
});

const assignSimilarCases = assign({
  similarCases: ({ event }: { event: any }) => event.output
});

// Main state machine
export const legalCaseMachine = createMachine({
  id: 'legalCase',
  types: {
    context: {} as LegalCaseContext,
    events: {} as LegalCaseEvents,
  },
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
          actions: setLoading
        },
        CREATE_CASE: {
          target: 'creatingCase',
          actions: setLoading,
          guard: isValidCaseData
        },
        SEARCH: {
          target: 'searching',
          actions: setLoading
        },
        SWITCH_TAB: {
          actions: switchTab
        }
      }
    },

    loadingCase: {
      invoke: {
        src: loadCaseService,
        input: ({ event }) => ({ caseId: event.caseId }),
        onDone: {
          target: 'caseLoaded',
          actions: assignCaseData
        },
        onError: {
          target: 'error',
          actions: assignError
        }
      }
    },

    creatingCase: {
      invoke: {
        src: createCaseService,
        input: ({ context }) => ({ caseData: context.formData.caseForm }),
        onDone: {
          target: 'caseLoaded',
          actions: [
            assignCaseData,
            assign({ formData: { caseForm: {}, evidenceForm: {} } })
          ]
        },
        onError: {
          target: 'error',
          actions: assignError
        }
      }
    },

    caseLoaded: {
      initial: 'loadingEvidence',
      entry: [
        assign({ isLoading: false })
      ],
      states: {
        loadingEvidence: {
          invoke: {
            src: loadEvidenceService,
            input: ({ context }) => ({ caseId: context.caseId! }),
            onDone: {
              target: 'ready',
              actions: assignEvidence
            },
            onError: {
              target: 'ready',
              actions: assignError
            }
          }
        },

        ready: {
          on: {
            ADD_EVIDENCE: {
              target: 'uploadingEvidence',
              actions: [
                setLoading,
                assign({
                  uploadQueue: ({ event }) => event.files
                })
              ]
            },
            PROCESS_EVIDENCE: {
              target: 'processingEvidence',
              actions: setLoading
            },
            START_AI_ANALYSIS: {
              target: 'aiAnalysis',
              actions: setLoading,
              guard: hasEvidence
            },
            FIND_SIMILAR_CASES: {
              target: 'findingSimilarCases',
              actions: setLoading
            },
            UPDATE_CASE: {
              target: 'updatingCase',
              actions: setLoading
            },
            DELETE_CASE: {
              target: 'deletingCase',
              actions: setLoading
            }
          }
        },

        uploadingEvidence: {
          invoke: {
            src: fromPromise(async ({ input }: { input: { files: File[]; caseId: string } }) => {
              const formData = new FormData();
              input.files.forEach((file: any) => formData.append('files', file));
              formData.append('caseId', input.caseId);
              
              const response = await fetch('/api/evidence/upload', {
                method: 'POST',
                body: formData
              });
              
              if (!response.ok) throw new Error('Upload failed');
              return await response.json();
            }),
            input: ({ context }) => ({
              files: context.uploadQueue,
              caseId: context.caseId!
            }),
            onDone: {
              target: 'loadingEvidence',
              actions: [
                assign({ uploadQueue: [] }),
                assign({ notifications: ({ context }) => [
                  ...context.notifications,
                  {
                    id: Date.now().toString(),
                    message: 'Evidence uploaded successfully',
                    type: 'info' as const
                  }
                ]})
              ]
            },
            onError: {
              target: 'ready',
              actions: assignError
            }
          }
        },

        processingEvidence: {
          invoke: {
            src: processEvidenceService,
            input: ({ context }) => ({ evidenceId: context.selectedEvidence!.id }),
            onDone: {
              target: 'ready',
              actions: [
                assign({ isLoading: false }),
                assign({
                  evidence: ({ context, event }) =>
                    context.evidence.map((e: any) => e.id === context.selectedEvidence!.id
                        ? { ...e, aiSummary: event.output.summary }
                        : e
                    )
                })
              ]
            },
            onError: {
              target: 'ready',
              actions: assignError
            }
          }
        },

        aiAnalysis: {
          initial: 'analyzing',
          states: {
            analyzing: {
              invoke: {
                src: fromPromise(async ({ input }: { input: { caseId: string } }) => {
                  return await aiSummarizationService.summarizeCase(input.caseId);
                }),
                input: ({ context }) => ({ caseId: context.caseId! }),
                onDone: {
                  target: 'complete',
                  actions: assignAISummary
                },
                onError: {
                  target: '#legalCase.caseLoaded.ready',
                  actions: assignError
                }
              },
              on: {
                AI_ANALYSIS_PROGRESS: {
                  actions: assignAIProgress
                }
              }
            },
            
            complete: {
              entry: [
                assign({ isLoading: false }),
                assign({
                  notifications: ({ context }) => [
                    ...context.notifications,
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
            src: findSimilarCasesService,
            input: ({ context }) => ({ caseId: context.caseId! }),
            onDone: {
              target: 'ready',
              actions: [
                assign({ isLoading: false }),
                assignSimilarCases
              ]
            },
            onError: {
              target: 'ready',
              actions: assignError
            }
          }
        },

        updatingCase: {
          invoke: {
            src: fromPromise(async ({ input }: { input: { caseId: string; updates: Partial<Case> } }) => {
              const response = await fetch(`/api/cases/${input.caseId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(input.updates)
              });
              if (!response.ok) throw new Error('Update failed');
              return await response.json();
            }),
            input: ({ context, event }) => ({
              caseId: context.caseId!,
              updates: event.updates
            }),
            onDone: {
              target: 'ready',
              actions: [
                assign({ isLoading: false }),
                assignCaseData
              ]
            },
            onError: {
              target: 'ready',
              actions: assignError
            }
          }
        },

        deletingCase: {
          invoke: {
            src: fromPromise(async ({ input }: { input: { caseId: string } }) => {
              const response = await fetch(`/api/cases/${input.caseId}`, {
                method: 'DELETE'
              });
              if (!response.ok) throw new Error('Delete failed');
              return true;
            }),
            input: ({ context }) => ({ caseId: context.caseId! }),
            onDone: {
              target: '#legalCase.idle',
              actions: [
                assign({
                  case: null,
                  caseId: null,
                  evidence: [],
                  selectedEvidence: null,
                  aiSummary: null,
                  similarCases: [],
                  isLoading: false
                })
              ]
            },
            onError: {
              target: 'ready',
              actions: assignError
            }
          }
        }
      },
      
      on: {
        SWITCH_TAB: {
          actions: switchTab
        },
        SET_WORKFLOW_STAGE: {
          actions: updateWorkflowStage
        },
        UPDATE_CASE_FORM: {
          actions: updateFormData
        },
        SELECT_EVIDENCE: {
          actions: assign({
            selectedEvidence: ({ event }) => event.evidence
          })
        },
        APPLY_FILTERS: {
          actions: assign({
            filters: ({ event }) => event.filters
          })
        },
        REFRESH: {
          target: '.loadingEvidence'
        }
      }
    },

    searching: {
      invoke: {
        src: searchService,
        input: ({ context, event }) => ({
          query: event.query,
          filters: context.filters
        }),
        onDone: {
          target: 'idle',
          actions: [
            assign({ isLoading: false }),
            assignSearchResults
          ]
        },
        onError: {
          target: 'error',
          actions: assignError
        }
      }
    },

    error: {
      entry: assign({ isLoading: false }),
      on: {
        RETRY: {
          target: 'idle',
          actions: clearError
        },
        DISMISS_ERROR: {
          actions: clearError
        }
      }
    }
  },

  on: {
    RESET: {
      target: 'idle',
      actions: assign({
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
      })
    }
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