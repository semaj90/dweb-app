import { createMachine, assign } from 'xstate';
import { fromCallback } from 'xstate/lib/actors';
import type { ProgressMsg } from '$lib/types/progress';

// WebSocket actor for real-time progress updates
function createWsActor(sessionId: string) {
  return fromCallback((callback) => {
    console.log('Creating WebSocket connection for session:', sessionId);
    
    const wsUrl = `${window.location.origin.replace(/^http/, 'ws')}/api/evidence/stream/${sessionId}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected for session:', sessionId);
      callback({ type: 'WS_CONNECTED' });
    };
    
    ws.onmessage = (event) => {
      try {
        const msg: ProgressMsg = JSON.parse(event.data);
        
        // Forward different message types as machine events
        switch (msg.type) {
          case 'upload-progress':
            callback({ 
              type: 'UPLOAD_PROGRESS', 
              fileId: msg.fileId, 
              progress: msg.progress 
            });
            break;
            
          case 'processing-step':
            callback({ 
              type: 'PROCESSING_STEP', 
              fileId: msg.fileId, 
              step: msg.step, 
              progress: msg.stepProgress || 0,
              fragment: msg.fragment 
            });
            break;
            
          case 'processing-complete':
            callback({ 
              type: 'PROCESSING_COMPLETE', 
              fileId: msg.fileId, 
              result: msg.finalResult 
            });
            break;
            
          case 'error':
            callback({ 
              type: 'ERROR', 
              fileId: msg.fileId, 
              error: msg.error 
            });
            break;
            
          default:
            console.warn('Unknown message type:', msg);
        }
      } catch (error) {
        console.error('WebSocket message parse error:', error);
        callback({ type: 'ERROR', error: { message: 'Failed to parse WebSocket message' } });
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      callback({ type: 'ERROR', error: { message: 'WebSocket connection error' } });
    };
    
    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      callback({ type: 'WS_CLOSED', code: event.code, reason: event.reason });
    };
    
    // Cleanup function
    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  });
}

// Evidence processing machine
export const evidenceProcessingMachine = createMachine({
  id: 'evidenceProcessing',
  initial: 'idle',
  context: {
    sessionId: null as string | null,
    evidenceId: null as string | null,
    files: {} as Record<string, {
      status: 'queued' | 'uploading' | 'processing' | 'completed' | 'failed';
      uploadProgress?: number;
      currentStep?: string;
      stepProgress?: number;
      fragment?: unknown;
      result?: unknown;
      error?: { message: string; code?: string; meta?: unknown };
    }>,
    steps: [] as string[],
    error: null as any
  },
  states: {
    idle: {
      on: {
        START_PROCESSING: {
          target: 'connecting',
          actions: assign({
            sessionId: ({ event }) => event.sessionId,
            evidenceId: ({ event }) => event.evidenceId,
            steps: ({ event }) => event.steps || [],
            files: ({ event, context }) => ({
              ...context.files,
              [event.evidenceId]: {
                status: 'queued',
                uploadProgress: 0
              }
            })
          })
        }
      }
    },
    
    connecting: {
      invoke: {
        id: 'wsActor',
        src: ({ context }) => createWsActor(context.sessionId!),
      },
      on: {
        WS_CONNECTED: {
          target: 'processing',
          actions: () => console.log('WebSocket connected, starting to listen for progress')
        },
        ERROR: {
          target: 'failed',
          actions: assign({
            error: ({ event }) => event.error
          })
        },
        WS_CLOSED: {
          target: 'disconnected'
        }
      }
    },
    
    processing: {
      on: {
        UPLOAD_PROGRESS: {
          actions: assign({
            files: ({ context, event }) => ({
              ...context.files,
              [event.fileId]: {
                ...context.files[event.fileId],
                status: 'uploading',
                uploadProgress: event.progress
              }
            })
          })
        },
        
        PROCESSING_STEP: {
          actions: assign({
            files: ({ context, event }) => ({
              ...context.files,
              [event.fileId]: {
                ...context.files[event.fileId],
                status: 'processing',
                currentStep: event.step,
                stepProgress: event.progress,
                fragment: event.fragment
              }
            })
          })
        },
        
        PROCESSING_COMPLETE: {
          target: 'completed',
          actions: assign({
            files: ({ context, event }) => ({
              ...context.files,
              [event.fileId]: {
                ...context.files[event.fileId],
                status: 'completed',
                result: event.result,
                stepProgress: 100
              }
            })
          })
        },
        
        ERROR: {
          target: 'failed',
          actions: assign({
            error: ({ event }) => event.error,
            files: ({ context, event }) => ({
              ...context.files,
              [event.fileId]: {
                ...context.files[event.fileId],
                status: 'failed',
                error: event.error
              }
            })
          })
        },
        
        WS_CLOSED: {
          target: 'disconnected'
        },
        
        CANCEL_PROCESSING: {
          target: 'cancelling'
        }
      }
    },
    
    completed: {
      type: 'final',
      entry: () => console.log('Evidence processing completed successfully')
    },
    
    failed: {
      on: {
        RETRY_PROCESSING: {
          target: 'connecting',
          actions: assign({
            error: null,
            files: ({ context }) => {
              const updatedFiles = { ...context.files };
              Object.keys(updatedFiles).forEach(fileId => {
                if (updatedFiles[fileId].status === 'failed') {
                  updatedFiles[fileId] = {
                    ...updatedFiles[fileId],
                    status: 'queued',
                    error: undefined
                  };
                }
              });
              return updatedFiles;
            }
          })
        },
        RESET: {
          target: 'idle',
          actions: assign({
            sessionId: null,
            evidenceId: null,
            files: {},
            steps: [],
            error: null
          })
        }
      }
    },
    
    disconnected: {
      on: {
        RECONNECT: {
          target: 'connecting'
        },
        RESET: {
          target: 'idle',
          actions: assign({
            sessionId: null,
            evidenceId: null,
            files: {},
            steps: [],
            error: null
          })
        }
      }
    },
    
    cancelling: {
      // TODO: Implement cancellation logic
      // This would involve calling a cancel endpoint and waiting for confirmation
      after: {
        5000: 'cancelled' // Timeout after 5 seconds
      },
      on: {
        PROCESSING_COMPLETE: 'completed',
        ERROR: 'failed'
      }
    },
    
    cancelled: {
      type: 'final',
      entry: () => console.log('Evidence processing cancelled')
    }
  }
}, {
  types: {
    events: {} as 
      | { type: 'START_PROCESSING'; sessionId: string; evidenceId: string; steps?: string[] }
      | { type: 'WS_CONNECTED' }
      | { type: 'UPLOAD_PROGRESS'; fileId: string; progress: number }
      | { type: 'PROCESSING_STEP'; fileId: string; step: string; progress: number; fragment?: unknown }
      | { type: 'PROCESSING_COMPLETE'; fileId: string; result?: unknown }
      | { type: 'ERROR'; fileId?: string; error: { message: string; code?: string; meta?: unknown } }
      | { type: 'WS_CLOSED'; code?: number; reason?: string }
      | { type: 'CANCEL_PROCESSING' }
      | { type: 'RETRY_PROCESSING' }
      | { type: 'RECONNECT' }
      | { type: 'RESET' }
  }
});

// Helper function to start evidence processing
export async function startEvidenceProcessing(evidenceId: string, steps: string[] = ['ocr', 'embedding', 'analysis']) {
  try {
    const response = await fetch('/api/evidence/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        evidenceId,
        steps
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to start processing');
    }
    
    const { sessionId } = await response.json();
    return { sessionId, evidenceId, steps };
  } catch (error) {
    console.error('Failed to start evidence processing:', error);
    throw error;
  }
}

// Selectors for easy state access
export const evidenceProcessingSelectors = {
  getFileStatus: (state: unknown, fileId: string) => state.context.files[fileId]?.status,
  getFileProgress: (state: unknown, fileId: string) => state.context.files[fileId]?.stepProgress || 0,
  getCurrentStep: (state: unknown, fileId: string) => state.context.files[fileId]?.currentStep,
  getFileResult: (state: unknown, fileId: string) => state.context.files[fileId]?.result,
  getFileError: (state: unknown, fileId: string) => state.context.files[fileId]?.error,
  isProcessing: (state: unknown) => state.matches('processing'),
  isCompleted: (state: unknown) => state.matches('completed'),
  isFailed: (state: unknown) => state.matches('failed'),
  isDisconnected: (state: unknown) => state.matches('disconnected'),
  canRetry: (state: unknown) => state.matches('failed') || state.matches('disconnected')
};