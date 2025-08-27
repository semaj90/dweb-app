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
  | { type: 'ADD_CONNECTION'; from: number; to: number; connectionType: string; strength: number }
  | { type: 'REMOVE_CONNECTION'; from: number; to: number }
  | { type: 'ANALYZE_SELECTED' }
  | { type: 'SAVE' }
  | { type: 'CLEAR_SELECTION' }
  | { type: 'RESET' };

export const detectiveMachine = createMachine<
  DetectiveContext & {
    history: {
      past: DetectiveContext[];
      future: DetectiveContext[];
      max: number;
    };
  },
  DetectiveEvent
>(
  {
    id: 'detective',
    initial: 'idle',
    context: {
      selectedIds: [],
      hypothesis: '',
      activeCase: null,
      evidence: [],
      connections: [],
      analysisResults: [],
      lastSaved: null,
      // memory/history for snapshots
      history: {
        past: [],
        future: [],
        max: 50
      }
    },

    states: {
      idle: {
        on: {
          SET_CASE: {
            actions: ['applySetCase', 'recordMemory'],
            target: 'case_loaded'
          }
        }
      },

      case_loaded: {
        on: {
          LOAD_EVIDENCE: {
            actions: ['applyLoadEvidence', 'recordMemory']
          },

          PIN_SELECT: {
            actions: ['applyPinSelect', 'recordMemory'],
            cond: (context, event) =>
              // narrow event type to PIN_SELECT
              (event as any)?.type === 'PIN_SELECT'
                ? !context.selectedIds.includes((event as any).id)
                : false
          },

          PIN_DESELECT: {
            actions: ['applyPinDeselect', 'recordMemory']
          },

          CLEAR_SELECTION: {
            actions: ['applyClearSelection', 'recordMemory']
          },

          HYPOTHESIS_EDIT: {
            actions: ['applyHypothesisEdit', 'recordMemory']
          },

          ADD_CONNECTION: {
            actions: ['applyAddConnection', 'recordMemory']
          },

          REMOVE_CONNECTION: {
            actions: ['applyRemoveConnection', 'recordMemory']
          },

          ANALYZE_SELECTED: {
            target: 'analyzing',
            cond: (context) => context.selectedIds.length > 0
          },

          SAVE: {
            target: 'saving'
          },

          RESET: {
            actions: ['applyReset', 'recordMemory']
          }
        }
      },

      analyzing: {
        invoke: {
          src: async (context) => {
            const selectedEvidence = context.evidence.filter((item) =>
              context.selectedIds.includes((item as any).id)
            );

            const response = await fetch('/api/llm/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                message: `Analyze these evidence items and identify patterns, connections, and potential legal significance: ${JSON.stringify(
                  selectedEvidence
                )}`,
                caseId: context.activeCase
              })
            });

            const data = await response.json();
            return {
              analysis: data.data?.message ?? data,
              timestamp: new Date(),
              evidenceCount: selectedEvidence.length
            };
          },
          onDone: {
            actions: ['applyAnalysisResult', 'recordMemory'],
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
            actions: ['applyLastSaved', 'recordMemory'],
            target: 'case_loaded'
          },
          onError: {
            target: 'case_loaded'
          }
        }
      }
    }
  },
  {
    actions: {
      // Named assign actions to keep types manageable and avoid inline event issues
      applySetCase: assign((_, event) => {
        const e = event as Extract<DetectiveEvent, { type: 'SET_CASE' }>;
        return {
          activeCase: e.caseId,
          evidence: [],
          selectedIds: [],
          hypothesis: '',
          connections: [],
          analysisResults: []
        };
      }),

      applyLoadEvidence: assign((_, event) => {
        const e = event as Extract<DetectiveEvent, { type: 'LOAD_EVIDENCE' }>;
        return { evidence: e.evidence };
      }),

      applyPinSelect: assign((context, event) => {
        const e = event as Extract<DetectiveEvent, { type: 'PIN_SELECT' }>;
        const newSelection = Array.from(new Set([...context.selectedIds, e.id]));
        return { selectedIds: newSelection };
      }),

      applyPinDeselect: assign((context, event) => {
        const e = event as Extract<DetectiveEvent, { type: 'PIN_DESELECT' }>;
        return { selectedIds: context.selectedIds.filter((id) => id !== e.id) };
      }),

      applyClearSelection: assign(() => ({
        selectedIds: [],
        hypothesis: ''
      })),

      applyHypothesisEdit: assign((_, event) => {
        const e = event as Extract<DetectiveEvent, { type: 'HYPOTHESIS_EDIT' }>;
        return { hypothesis: e.text };
      }),

      applyAddConnection: assign((context, event) => {
        const e = event as Extract<
          DetectiveEvent,
          { type: 'ADD_CONNECTION' }
        >;
        const filtered = context.connections.filter(
          (conn) => !(conn.from === e.from && conn.to === e.to)
        );
        return {
          connections: [
            ...filtered,
            { from: e.from, to: e.to, type: e.connectionType, strength: e.strength }
          ]
        };
      }),

      applyRemoveConnection: assign((context, event) => {
        const e = event as Extract<DetectiveEvent, { type: 'REMOVE_CONNECTION' }>;
        return {
          connections: context.connections.filter(
            (conn) => !(conn.from === e.from && conn.to === e.to)
          )
        };
      }),

      applyReset: assign(() => ({
        selectedIds: [],
        hypothesis: '',
        connections: [],
        analysisResults: [],
        lastSaved: null
      })),

      applyAnalysisResult: assign((context, event) => {
        // event.data comes from invoke onDone
        const result = (event as any).data;
        return {
          analysisResults: [...context.analysisResults, result]
        };
      }),

      applyLastSaved: assign(() => ({
        lastSaved: new Date()
      })),

      // The memory recorder: stores a snapshot of the DetectiveContext (omitting history)
      recordMemory: assign((context) => {
        // create a shallow snapshot of DetectiveContext fields (omit history)
        const snapshot: DetectiveContext = {
          selectedIds: Array.from(context.selectedIds),
          hypothesis: context.hypothesis,
          activeCase: context.activeCase,
          evidence: Array.isArray(context.evidence) ? [...context.evidence] : context.evidence,
          connections: Array.isArray(context.connections) ? [...context.connections] : context.connections,
          analysisResults: Array.isArray(context.analysisResults)
            ? [...context.analysisResults]
            : context.analysisResults,
          lastSaved: context.lastSaved instanceof Date ? new Date(context.lastSaved) : context.lastSaved
        };

        const past = [...context.history.past, snapshot];
        // cap history length
        const cappedPast = past.slice(-context.history.max);

        return {
          history: {
            ...context.history,
            past: cappedPast,
            future: []
          }
        };
      })
    }
  }
);

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