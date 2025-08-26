import { createMachine, assign } from 'xstate';

interface DetectiveContext {
  selectedIds: number[];
  hypothesis: string;
  activeCase: number | null;
  evidence: any[];
  connections: Array<{from: number, to: number, type: string, strength: number}>;
  analysisResults: any[];
  lastSaved: Date | null;
}

type DetectiveEvent =
  | { type: 'PIN_SELECT'; id: number }
  | { type: 'PIN_DESELECT'; id: number }
  | { type: 'HYPOTHESIS_EDIT'; text: string }
  | { type: 'SET_CASE'; caseId: number }
  | { type: 'LOAD_EVIDENCE'; evidence: any[] }
  | { type: 'ADD_CONNECTION'; from: number; to: number; type: string; strength: number }
  | { type: 'REMOVE_CONNECTION'; from: number; to: number }
  | { type: 'ANALYZE_SELECTED' }
  | { type: 'SAVE' }
  | { type: 'CLEAR_SELECTION' }
  | { type: 'RESET' };

export const detectiveMachine = createMachine<DetectiveContext, DetectiveEvent>({
  id: 'detective',
  initial: 'idle',
  context: {
    selectedIds: [],
    hypothesis: '',
    activeCase: null,
    evidence: [],
    connections: [],
    analysisResults: [],
    lastSaved: null
  },

  states: {
    idle: {
      on: {
        SET_CASE: {
          actions: assign({
            activeCase: (_, event) => event.caseId,
            evidence: [],
            selectedIds: [],
            hypothesis: '',
            connections: [],
            analysisResults: []
          }),
          target: 'case_loaded'
        }
      }
    },

    case_loaded: {
      on: {
        LOAD_EVIDENCE: {
          actions: assign({
            evidence: (_, event) => event.evidence
          })
        },

        PIN_SELECT: {
          actions: assign({
            selectedIds: (context, event) => {
              const newSelection = [...new Set([...context.selectedIds, event.id])];
              return newSelection;
            }
          }),
          cond: (context, event) => !context.selectedIds.includes(event.id)
        },

        PIN_DESELECT: {
          actions: assign({
            selectedIds: (context, event) => 
              context.selectedIds.filter(id => id !== event.id)
          })
        },

        CLEAR_SELECTION: {
          actions: assign({
            selectedIds: [],
            hypothesis: ''
          })
        },

        HYPOTHESIS_EDIT: {
          actions: assign({
            hypothesis: (_, event) => event.text
          })
        },

        ADD_CONNECTION: {
          actions: assign({
            connections: (context, event) => [
              ...context.connections.filter(
                conn => !(conn.from === event.from && conn.to === event.to)
              ),
              {
                from: event.from,
                to: event.to,
                type: event.type,
                strength: event.strength
              }
            ]
          })
        },

        REMOVE_CONNECTION: {
          actions: assign({
            connections: (context, event) =>
              context.connections.filter(
                conn => !(conn.from === event.from && conn.to === event.to)
              )
          })
        },

        ANALYZE_SELECTED: {
          target: 'analyzing',
          cond: (context) => context.selectedIds.length > 0
        },

        SAVE: {
          target: 'saving'
        },

        RESET: {
          actions: assign({
            selectedIds: [],
            hypothesis: '',
            connections: [],
            analysisResults: [],
            lastSaved: null
          })
        }
      }
    },

    analyzing: {
      invoke: {
        src: async (context) => {
          // Analyze selected evidence using AI
          const selectedEvidence = context.evidence.filter(
            item => context.selectedIds.includes(item.id)
          );

          const response = await fetch('/api/llm/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: `Analyze these evidence items and identify patterns, connections, and potential legal significance: ${JSON.stringify(selectedEvidence)}`,
              caseId: context.activeCase
            })
          });

          const data = await response.json();
          return {
            analysis: data.data.message,
            timestamp: new Date(),
            evidenceCount: selectedEvidence.length
          };
        },
        onDone: {
          actions: assign({
            analysisResults: (context, event) => [
              ...context.analysisResults,
              event.data
            ]
          }),
          target: 'case_loaded'
        },
        onError: {
          target: 'case_loaded'
        }
      }
    },

    saving: {
      invoke: {
        src: async (context) => {
          const saveData = {
            hypothesis: context.hypothesis,
            selectedIds: context.selectedIds,
            connections: context.connections,
            analysisResults: context.analysisResults,
            caseId: context.activeCase
          };

          const response = await fetch('/api/cases/hypothesis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(saveData)
          });

          if (!response.ok) {
            throw new Error('Failed to save');
          }

          return await response.json();
        },
        onDone: {
          actions: assign({
            lastSaved: () => new Date()
          }),
          target: 'case_loaded'
        },
        onError: {
          target: 'case_loaded'
        }
      }
    }
  }
});

// Helper functions for the detective machine
export const detectiveHelpers = {
  getSelectedEvidence: (context: DetectiveContext) => 
    context.evidence.filter(item => context.selectedIds.includes(item.id)),

  getConnectionsForEvidence: (context: DetectiveContext, evidenceId: number) =>
    context.connections.filter(conn => conn.from === evidenceId || conn.to === evidenceId),

  getAnalysisSummary: (context: DetectiveContext) => ({
    totalEvidence: context.evidence.length,
    selectedCount: context.selectedIds.length,
    connectionCount: context.connections.length,
    analysisCount: context.analysisResults.length,
    hasHypothesis: context.hypothesis.length > 0,
    lastSaved: context.lastSaved
  })
};