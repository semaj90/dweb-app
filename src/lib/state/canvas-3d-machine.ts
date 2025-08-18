/**
 * XState Machine for 3D Canvas State Management
 * Implements high-performance AI assistant canvas interactions
 */

import { createMachine, assign } from 'xstate';

type Canvas3DEvents =
  | { type: 'DOCUMENT_SELECTED'; documentId: string }
  | { type: 'DOCUMENT_HOVER'; documentId: string }
  | { type: 'DOCUMENT_UNHOVER' }
  | { type: 'SOM_UPDATE'; somData: Float32Array }
  | { type: 'TOGGLE_SOM_OVERLAY' }
  | { type: 'RESET_VIEW' }
  | { type: 'SET_CAMERA_MODE'; mode: 'orbit' | 'fly' | 'fixed' }
  | { type: 'GPU_PROCESSING_START' }
  | { type: 'GPU_PROCESSING_COMPLETE'; result: any }
  | { type: 'GPU_PROCESSING_ERROR'; error: string }
  | { type: 'INTERACTION_RECORDED'; interaction: UserInteraction };

interface Canvas3DContext {
  selectedDocumentId: string | null;
  hoveredDocumentId: string | null;
  cameraMode: 'orbit' | 'fly' | 'fixed';
  showSOMOverlay: boolean;
  isGPUProcessing: boolean;
  somData: Float32Array | null;
  interactionHistory: UserInteraction[];
  viewState: {
    zoom: number;
    rotation: { x: number; y: number; z: number };
    position: { x: number; y: number; z: number };
  };
  performance: {
    fps: number;
    memoryUsage: number;
    gpuUtilization: number;
  };
}

interface UserInteraction {
  type: 'mouse_move' | 'click' | 'hover' | 'scroll' | 'key_press';
  timestamp: number;
  position: { x: number; y: number };
  documentId?: string;
  target: HTMLElement;
  metadata?: Record<string, any>;
}

export function createCanvas3DMachine() {
  return createMachine<Canvas3DContext, Canvas3DEvents>({
    id: 'canvas3D',
    initial: 'idle',
    context: {
      selectedDocumentId: null,
      hoveredDocumentId: null,
      cameraMode: 'orbit',
      showSOMOverlay: true,
      isGPUProcessing: false,
      somData: null,
      interactionHistory: [],
      viewState: {
        zoom: 1.0,
        rotation: { x: 0, y: 0, z: 0 },
        position: { x: 0, y: 0, z: 10 }
      },
      performance: {
        fps: 60,
        memoryUsage: 0,
        gpuUtilization: 0
      }
    },
    states: {
      idle: {
        on: {
          DOCUMENT_SELECTED: {
            target: 'documentSelected',
            actions: assign({
              selectedDocumentId: ({ event }) => event.documentId
            })
          },
          DOCUMENT_HOVER: {
            target: 'documentHovered',
            actions: assign({
              hoveredDocumentId: ({ event }) => event.documentId
            })
          },
          GPU_PROCESSING_START: {
            target: 'gpuProcessing',
            actions: assign({
              isGPUProcessing: true
            })
          },
          TOGGLE_SOM_OVERLAY: {
            actions: assign({
              showSOMOverlay: ({ context }) => !context.showSOMOverlay
            })
          },
          SET_CAMERA_MODE: {
            actions: assign({
              cameraMode: ({ event }) => event.mode
            })
          },
          INTERACTION_RECORDED: {
            actions: assign({
              interactionHistory: ({ context, event }) => [
                ...context.interactionHistory.slice(-99), // Keep last 100 interactions
                event.interaction
              ]
            })
          }
        }
      },
      
      documentSelected: {
        entry: ['highlightSelectedDocument', 'triggerDocumentAnalysis'],
        on: {
          DOCUMENT_SELECTED: {
            target: 'documentSelected',
            actions: assign({
              selectedDocumentId: ({ event }) => event.documentId
            })
          },
          DOCUMENT_HOVER: {
            target: 'documentSelectedAndHovered',
            actions: assign({
              hoveredDocumentId: ({ event }) => event.documentId
            })
          },
          RESET_VIEW: {
            target: 'idle',
            actions: assign({
              selectedDocumentId: null,
              hoveredDocumentId: null
            })
          }
        }
      },
      
      documentHovered: {
        entry: ['showDocumentPreview'],
        on: {
          DOCUMENT_SELECTED: {
            target: 'documentSelectedAndHovered',
            actions: assign({
              selectedDocumentId: ({ event }) => event.documentId
            })
          },
          DOCUMENT_UNHOVER: {
            target: 'idle',
            actions: assign({
              hoveredDocumentId: null
            })
          },
          DOCUMENT_HOVER: {
            actions: assign({
              hoveredDocumentId: ({ event }) => event.documentId
            })
          }
        }
      },
      
      documentSelectedAndHovered: {
        entry: ['highlightSelectedDocument', 'showDocumentPreview'],
        on: {
          DOCUMENT_UNHOVER: {
            target: 'documentSelected',
            actions: assign({
              hoveredDocumentId: null
            })
          },
          DOCUMENT_SELECTED: {
            actions: assign({
              selectedDocumentId: ({ event }) => event.documentId
            })
          },
          RESET_VIEW: {
            target: 'idle',
            actions: assign({
              selectedDocumentId: null,
              hoveredDocumentId: null
            })
          }
        }
      },
      
      gpuProcessing: {
        entry: ['startGPUComputeShader'],
        on: {
          GPU_PROCESSING_COMPLETE: {
            target: 'idle',
            actions: [
              assign({
                isGPUProcessing: false
              }),
              'updateSOMVisualization'
            ]
          },
          GPU_PROCESSING_ERROR: {
            target: 'idle',
            actions: [
              assign({
                isGPUProcessing: false
              }),
              'handleGPUError'
            ]
          }
        }
      }
    },
    
    on: {
      SOM_UPDATE: {
        actions: assign({
          somData: ({ event }) => event.somData
        })
      },
      RESET_VIEW: {
        target: 'idle',
        actions: [
          assign({
            selectedDocumentId: null,
            hoveredDocumentId: null,
            viewState: {
              zoom: 1.0,
              rotation: { x: 0, y: 0, z: 0 },
              position: { x: 0, y: 0, z: 10 }
            }
          }),
          'resetCameraView'
        ]
      }
    }
  }, {
    actions: {
      highlightSelectedDocument: ({ context }) => {
        console.log('Highlighting document:', context.selectedDocumentId);
        // This would trigger Three.js material changes
      },
      
      showDocumentPreview: ({ context }) => {
        console.log('Showing preview for:', context.hoveredDocumentId);
        // This would show a tooltip or preview panel
      },
      
      triggerDocumentAnalysis: ({ context }) => {
        console.log('Triggering analysis for:', context.selectedDocumentId);
        // This would send the document to the AI analysis pipeline
      },
      
      startGPUComputeShader: () => {
        console.log('Starting GPU compute shader for SOM update');
        // This would trigger WebGPU compute shader execution
      },
      
      updateSOMVisualization: ({ event }) => {
        console.log('Updating SOM visualization with result:', event);
        // This would update the Three.js SOM grid mesh
      },
      
      handleGPUError: ({ event }) => {
        console.error('GPU processing error:', event);
        // This would handle GPU fallback to CPU processing
      },
      
      resetCameraView: () => {
        console.log('Resetting camera to default view');
        // This would animate camera back to default position
      }
    }
  });
}

// Export types for use in components
export type { Canvas3DContext, Canvas3DEvents, UserInteraction };