import { createMachine, assign } from 'xstate';
import type { 
  EvidenceNode, 
  EvidenceData, 
  CanvasData, 
  NodeRelationship, 
  AIAnalysis,
  ProcessingRequest,
  ProcessingResult,
  WebGPUCapabilities
} from '$lib/types/evidence';

export interface EvidenceCanvasContext {
  // Canvas state
  canvas: any | null;
  evidenceNodes: EvidenceNode[];
  selectedNodes: string[];
  clipboard: EvidenceNode[];
  
  // Processing state
  isProcessing: boolean;
  processingQueue: ProcessingRequest[];
  processingResults: Record<string, ProcessingResult>;
  
  // GPU/WASM capabilities
  webgpuCapabilities: WebGPUCapabilities | null;
  useGPUAcceleration: boolean;
  
  // WebSocket connection
  wsConnected: boolean;
  wsReconnectCount: number;
  
  // Analysis results
  aiAnalysis: AIAnalysis | null;
  analysisHistory: AIAnalysis[];
  
  // Error handling
  lastError: string | null;
  retryCount: number;
  
  // Session data
  caseId: string | null;
  userId: string | null;
  sessionId: string | null;
  
  // Canvas settings
  gridEnabled: boolean;
  snapToGrid: boolean;
  showRelationships: boolean;
  zoomLevel: number;
}

export type EvidenceCanvasEvent = 
  // Canvas operations
  | { type: 'INITIALIZE_CANVAS'; canvas: any; caseId?: string; userId?: string }
  | { type: 'ADD_EVIDENCE_NODE'; evidenceData: EvidenceData; position: { x: number; y: number } }
  | { type: 'SELECT_NODE'; nodeId: string; multiSelect?: boolean }
  | { type: 'DESELECT_ALL' }
  | { type: 'DELETE_NODE'; nodeId: string }
  | { type: 'MOVE_NODE'; nodeId: string; position: { x: number; y: number } }
  | { type: 'COPY_NODES'; nodeIds: string[] }
  | { type: 'PASTE_NODES'; position: { x: number; y: number } }
  | { type: 'CREATE_RELATIONSHIP'; fromNodeId: string; toNodeId: string; type: string }
  | { type: 'DELETE_RELATIONSHIP'; relationshipId: string }
  
  // Processing operations
  | { type: 'START_PROCESSING'; request: ProcessingRequest }
  | { type: 'PROCESSING_UPDATE'; result: ProcessingResult }
  | { type: 'PROCESSING_COMPLETE'; result: ProcessingResult }
  | { type: 'PROCESSING_ERROR'; error: string }
  | { type: 'CANCEL_PROCESSING'; jobId: string }
  | { type: 'RETRY_PROCESSING' }
  
  // Analysis operations
  | { type: 'ANALYZE_CANVAS'; options?: { includeRelationships?: boolean } }
  | { type: 'ANALYSIS_COMPLETE'; analysis: AIAnalysis }
  | { type: 'ANALYSIS_ERROR'; error: string }
  
  // WebSocket events
  | { type: 'WS_CONNECT' }
  | { type: 'WS_CONNECTED' }
  | { type: 'WS_DISCONNECTED'; reason?: string }
  | { type: 'WS_MESSAGE'; data: any }
  | { type: 'WS_ERROR'; error: string }
  | { type: 'WS_RETRY_CONNECTION' }
  
  // Settings
  | { type: 'TOGGLE_GRID' }
  | { type: 'TOGGLE_SNAP_TO_GRID' }
  | { type: 'TOGGLE_RELATIONSHIPS' }
  | { type: 'SET_ZOOM'; level: number }
  | { type: 'TOGGLE_GPU_ACCELERATION' }
  
  // Data operations
  | { type: 'SAVE_CANVAS' }
  | { type: 'LOAD_CANVAS'; data: CanvasData }
  | { type: 'EXPORT_CANVAS'; format: 'json' | 'image' | 'pdf' }
  | { type: 'IMPORT_EVIDENCE'; files: File[] }
  
  // Error handling
  | { type: 'CLEAR_ERROR' }
  | { type: 'RESET_CANVAS' };

const evidenceCanvasMachine = createMachine<
  EvidenceCanvasContext,
  EvidenceCanvasEvent
>({
  id: 'evidenceCanvas',
  initial: 'idle',
  context: {
    canvas: null,
    evidenceNodes: [],
    selectedNodes: [],
    clipboard: [],
    isProcessing: false,
    processingQueue: [],
    processingResults: {},
    webgpuCapabilities: null,
    useGPUAcceleration: true,
    wsConnected: false,
    wsReconnectCount: 0,
    aiAnalysis: null,
    analysisHistory: [],
    lastError: null,
    retryCount: 0,
    caseId: null,
    userId: null,
    sessionId: null,
    gridEnabled: true,
    snapToGrid: true,
    showRelationships: true,
    zoomLevel: 1.0
  },
  
  states: {
    idle: {
      on: {
        INITIALIZE_CANVAS: {
          target: 'initializing',
          actions: assign({
            canvas: ({ event }) => event.canvas,
            caseId: ({ event }) => event.caseId || null,
            userId: ({ event }) => event.userId || null,
            sessionId: () => crypto.randomUUID(),
            lastError: () => null
          })
        },
        LOAD_CANVAS: {
          target: 'loading',
          actions: assign({
            lastError: () => null
          })
        }
      }
    },
    
    initializing: {
      invoke: {
        id: 'initializeServices',
        src: 'initializeCanvasServices',
        onDone: {
          target: 'ready',
          actions: assign({
            webgpuCapabilities: ({ event }) => event.data.webgpu,
            wsConnected: ({ event }) => event.data.wsConnected
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            lastError: ({ event }) => `Initialization failed: ${event.data}`
          })
        }
      }
    },
    
    loading: {
      invoke: {
        id: 'loadCanvas',
        src: 'loadCanvasData',
        onDone: {
          target: 'ready',
          actions: assign({
            evidenceNodes: ({ event }) => event.data.evidence_nodes || [],
            canvas: ({ event }) => event.data.canvas_json,
            caseId: ({ event }) => event.data.case_id || null,
            userId: ({ event }) => event.data.user_id || null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            lastError: ({ event }) => `Failed to load canvas: ${event.data}`
          })
        }
      }
    },
    
    ready: {
      type: 'parallel',
      states: {
        canvas: {
          initial: 'active',
          states: {
            active: {
              on: {
                ADD_EVIDENCE_NODE: {
                  actions: assign({
                    evidenceNodes: ({ context, event }) => [
                      ...context.evidenceNodes,
                      {
                        id: crypto.randomUUID(),
                        title: event.evidenceData.title,
                        type: event.evidenceData.evidenceType,
                        position: event.position,
                        data: event.evidenceData,
                        connections: [],
                        metadata: {
                          processingStatus: 'pending',
                          qualityScore: 0
                        }
                      }
                    ]
                  })
                },
                
                SELECT_NODE: {
                  actions: assign({
                    selectedNodes: ({ context, event }) => {
                      if (event.multiSelect) {
                        return context.selectedNodes.includes(event.nodeId)
                          ? context.selectedNodes.filter(id => id !== event.nodeId)
                          : [...context.selectedNodes, event.nodeId];
                      }
                      return [event.nodeId];
                    }
                  })
                },
                
                DESELECT_ALL: {
                  actions: assign({
                    selectedNodes: () => []
                  })
                },
                
                DELETE_NODE: {
                  actions: assign({
                    evidenceNodes: ({ context, event }) => 
                      context.evidenceNodes.filter(node => node.id !== event.nodeId),
                    selectedNodes: ({ context, event }) =>
                      context.selectedNodes.filter(id => id !== event.nodeId)
                  })
                },
                
                MOVE_NODE: {
                  actions: assign({
                    evidenceNodes: ({ context, event }) => 
                      context.evidenceNodes.map(node => 
                        node.id === event.nodeId 
                          ? { ...node, position: event.position }
                          : node
                      )
                  })
                },
                
                COPY_NODES: {
                  actions: assign({
                    clipboard: ({ context, event }) => 
                      context.evidenceNodes.filter(node => 
                        event.nodeIds.includes(node.id)
                      )
                  })
                },
                
                PASTE_NODES: {
                  actions: assign({
                    evidenceNodes: ({ context, event }) => [
                      ...context.evidenceNodes,
                      ...context.clipboard.map((node, index) => ({
                        ...node,
                        id: crypto.randomUUID(),
                        position: {
                          x: event.position.x + (index * 20),
                          y: event.position.y + (index * 20)
                        }
                      }))
                    ]
                  })
                },
                
                CREATE_RELATIONSHIP: {
                  actions: assign({
                    evidenceNodes: ({ context, event }) => 
                      context.evidenceNodes.map(node => {
                        if (node.id === event.fromNodeId) {
                          return {
                            ...node,
                            connections: [...(node.connections || []), event.toNodeId]
                          };
                        }
                        return node;
                      })
                  })
                }
              }
            }
          }
        },
        
        processing: {
          initial: 'idle',
          states: {
            idle: {
              on: {
                START_PROCESSING: {
                  target: 'active',
                  actions: assign({
                    isProcessing: () => true,
                    processingQueue: ({ context, event }) => [
                      ...context.processingQueue,
                      event.request
                    ],
                    lastError: () => null
                  })
                }
              }
            },
            
            active: {
              invoke: {
                id: 'processEvidence',
                src: 'processEvidenceRequest',
                onDone: {
                  target: 'idle',
                  actions: assign({
                    isProcessing: () => false,
                    processingResults: ({ context, event }) => ({
                      ...context.processingResults,
                      [event.data.jobId]: event.data
                    }),
                    processingQueue: ({ context }) => context.processingQueue.slice(1)
                  })
                },
                onError: {
                  target: 'idle',
                  actions: assign({
                    isProcessing: () => false,
                    lastError: ({ event }) => `Processing failed: ${event.data}`,
                    retryCount: ({ context }) => context.retryCount + 1
                  })
                }
              },
              
              on: {
                PROCESSING_UPDATE: {
                  actions: assign({
                    processingResults: ({ context, event }) => ({
                      ...context.processingResults,
                      [event.result.jobId]: event.result
                    })
                  })
                },
                
                CANCEL_PROCESSING: {
                  target: 'idle',
                  actions: assign({
                    isProcessing: () => false,
                    processingQueue: ({ context, event }) => 
                      context.processingQueue.filter(req => req.evidenceId !== event.jobId)
                  })
                },
                
                RETRY_PROCESSING: {
                  actions: assign({
                    retryCount: ({ context }) => context.retryCount + 1,
                    lastError: () => null
                  })
                }
              }
            }
          }
        },
        
        analysis: {
          initial: 'idle',
          states: {
            idle: {
              on: {
                ANALYZE_CANVAS: {
                  target: 'analyzing'
                }
              }
            },
            
            analyzing: {
              invoke: {
                id: 'analyzeCanvas',
                src: 'performCanvasAnalysis',
                onDone: {
                  target: 'idle',
                  actions: assign({
                    aiAnalysis: ({ event }) => event.data,
                    analysisHistory: ({ context, event }) => [
                      ...context.analysisHistory,
                      event.data
                    ].slice(-10) // Keep last 10 analyses
                  })
                },
                onError: {
                  target: 'idle',
                  actions: assign({
                    lastError: ({ event }) => `Analysis failed: ${event.data}`
                  })
                }
              }
            }
          }
        },
        
        connection: {
          initial: 'disconnected',
          states: {
            disconnected: {
              on: {
                WS_CONNECT: {
                  target: 'connecting'
                }
              }
            },
            
            connecting: {
              invoke: {
                id: 'connectWebSocket',
                src: 'connectToWebSocket',
                onDone: {
                  target: 'connected',
                  actions: assign({
                    wsConnected: () => true,
                    wsReconnectCount: () => 0
                  })
                },
                onError: {
                  target: 'disconnected',
                  actions: assign({
                    wsConnected: () => false,
                    wsReconnectCount: ({ context }) => context.wsReconnectCount + 1,
                    lastError: ({ event }) => `WebSocket connection failed: ${event.data}`
                  })
                }
              }
            },
            
            connected: {
              on: {
                WS_DISCONNECTED: {
                  target: 'disconnected',
                  actions: assign({
                    wsConnected: () => false
                  })
                },
                
                WS_MESSAGE: {
                  actions: 'handleWebSocketMessage'
                },
                
                WS_ERROR: {
                  actions: assign({
                    lastError: ({ event }) => `WebSocket error: ${event.error}`
                  })
                }
              }
            }
          }
        }
      },
      
      on: {
        SAVE_CANVAS: {
          actions: 'saveCanvasData'
        },
        
        EXPORT_CANVAS: {
          actions: 'exportCanvas'
        },
        
        IMPORT_EVIDENCE: {
          actions: 'importEvidenceFiles'
        },
        
        TOGGLE_GRID: {
          actions: assign({
            gridEnabled: ({ context }) => !context.gridEnabled
          })
        },
        
        TOGGLE_SNAP_TO_GRID: {
          actions: assign({
            snapToGrid: ({ context }) => !context.snapToGrid
          })
        },
        
        TOGGLE_RELATIONSHIPS: {
          actions: assign({
            showRelationships: ({ context }) => !context.showRelationships
          })
        },
        
        TOGGLE_GPU_ACCELERATION: {
          actions: assign({
            useGPUAcceleration: ({ context }) => !context.useGPUAcceleration
          })
        },
        
        SET_ZOOM: {
          actions: assign({
            zoomLevel: ({ event }) => Math.max(0.1, Math.min(5.0, event.level))
          })
        },
        
        CLEAR_ERROR: {
          actions: assign({
            lastError: () => null,
            retryCount: () => 0
          })
        },
        
        RESET_CANVAS: {
          target: 'idle',
          actions: assign({
            canvas: () => null,
            evidenceNodes: () => [],
            selectedNodes: () => [],
            clipboard: () => [],
            aiAnalysis: () => null,
            analysisHistory: () => [],
            processingQueue: () => [],
            processingResults: () => ({}),
            lastError: () => null,
            retryCount: () => 0
          })
        }
      }
    },
    
    error: {
      on: {
        CLEAR_ERROR: {
          target: 'idle',
          actions: assign({
            lastError: () => null,
            retryCount: () => 0
          })
        },
        
        RETRY_PROCESSING: {
          target: 'initializing',
          actions: assign({
            retryCount: ({ context }) => context.retryCount + 1,
            lastError: () => null
          })
        }
      }
    }
  }
}, {
  // Service implementations
  services: {
    initializeCanvasServices: async (context) => {
      // Initialize WebGPU capabilities
      let webgpu = null;
      if ('gpu' in navigator) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            const device = await adapter.requestDevice();
            webgpu = {
              available: true,
              device,
              adapter,
              features: Array.from(adapter.features),
              limits: adapter.limits
            };
          }
        } catch (error) {
          console.warn('WebGPU initialization failed:', error);
          webgpu = { available: false, features: [], limits: {} };
        }
      }
      
      // Test WebSocket connection
      let wsConnected = false;
      try {
        const ws = new WebSocket('ws://localhost:8090/canvas');
        await new Promise((resolve, reject) => {
          ws.onopen = () => {
            wsConnected = true;
            ws.close();
            resolve(true);
          };
          ws.onerror = reject;
          setTimeout(reject, 1000); // 1 second timeout
        });
      } catch (error) {
        console.warn('WebSocket test failed:', error);
      }
      
      return { webgpu, wsConnected };
    },
    
    loadCanvasData: async (context, event) => {
      if ('data' in event && event.data) {
        return event.data;
      }
      
      // Load from API if no data provided
      const response = await fetch(`/api/evidence/canvas/${context.caseId}`);
      if (!response.ok) {
        throw new Error(`Failed to load canvas: ${response.statusText}`);
      }
      
      return response.json();
    },
    
    processEvidenceRequest: async (context, event) => {
      const request = context.processingQueue[0];
      if (!request) {
        throw new Error('No processing request in queue');
      }
      
      const response = await fetch('/api/evidence/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceId: request.evidenceId,
          steps: request.steps,
          options: {
            ...request.options,
            useGPUAcceleration: context.useGPUAcceleration
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }
      
      return response.json();
    },
    
    performCanvasAnalysis: async (context, event) => {
      const canvasData = {
        canvas_json: context.canvas?.toJSON?.() || null,
        evidence_nodes: context.evidenceNodes,
        node_relationships: context.evidenceNodes.flatMap(node => 
          (node.connections || []).map(targetId => ({
            from: node.id,
            to: targetId,
            type: 'related_to' as const,
            confidence: 0.8
          }))
        ),
        case_id: context.caseId,
        user_id: context.userId,
        timestamp: new Date().toISOString()
      };
      
      const response = await fetch('http://localhost:8090/wasm/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          module: 'evidence_analyzer',
          function: 'analyze_canvas',
          data: canvasData,
          gpu_accelerated: context.useGPUAcceleration && context.webgpuCapabilities?.available
        })
      });
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      return {
        id: crypto.randomUUID(),
        model: 'evidence_analyzer_wasm',
        confidence: result.confidence || 0.85,
        entities: result.entities || [],
        relationships: result.relationships || [],
        classification: result.classification || 'evidence_collection',
        keywords: result.keywords || [],
        summary: result.summary || 'Canvas analysis completed',
        timestamp: new Date(),
        processingTime: result.processing_time_ms || 0,
        gpuAccelerated: context.useGPUAcceleration && context.webgpuCapabilities?.available
      };
    },
    
    connectToWebSocket: async (context) => {
      return new Promise((resolve, reject) => {
        const ws = new WebSocket('ws://localhost:8090/canvas');
        
        ws.onopen = () => {
          // Send session info
          ws.send(JSON.stringify({
            type: 'session_init',
            sessionId: context.sessionId,
            caseId: context.caseId,
            userId: context.userId
          }));
          resolve(ws);
        };
        
        ws.onerror = reject;
        
        setTimeout(() => reject(new Error('Connection timeout')), 5000);
      });
    }
  },
  
  actions: {
    handleWebSocketMessage: (context, event) => {
      const { data } = event;
      console.log('WebSocket message received:', data);
      
      // Handle different message types
      if (data.type === 'evidence_update') {
        // Update evidence node
        // This would trigger additional events
      } else if (data.type === 'processing_update') {
        // Update processing status
        // This would trigger PROCESSING_UPDATE event
      }
    },
    
    saveCanvasData: async (context) => {
      const canvasData = {
        canvas_json: context.canvas?.toJSON?.() || null,
        evidence_nodes: context.evidenceNodes,
        node_relationships: context.evidenceNodes.flatMap(node => 
          (node.connections || []).map(targetId => ({
            from: node.id,
            to: targetId,
            type: 'related_to' as const
          }))
        ),
        case_id: context.caseId,
        user_id: context.userId,
        timestamp: new Date().toISOString(),
        metadata: {
          version: '1.0',
          created: new Date(),
          lastModified: new Date(),
          author: context.userId || 'anonymous'
        }
      };
      
      await fetch('/api/evidence/canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(canvasData)
      });
    },
    
    exportCanvas: async (context, event) => {
      const format = event.format;
      const canvasData = {
        canvas_json: context.canvas?.toJSON?.() || null,
        evidence_nodes: context.evidenceNodes,
        metadata: {
          exported: new Date().toISOString(),
          format,
          version: '1.0'
        }
      };
      
      if (format === 'json') {
        const blob = new Blob([JSON.stringify(canvasData, null, 2)], {
          type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evidence-canvas-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } else if (format === 'image') {
        // Export canvas as image
        if (context.canvas?.toDataURL) {
          const dataURL = context.canvas.toDataURL('image/png');
          const a = document.createElement('a');
          a.href = dataURL;
          a.download = `evidence-canvas-${new Date().toISOString().split('T')[0]}.png`;
          a.click();
        }
      }
    },
    
    importEvidenceFiles: async (context, event) => {
      const files = event.files;
      for (const file of files) {
        // Process each file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('caseId', context.caseId || '');
        
        const response = await fetch('/api/evidence/upload', {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          const result = await response.json();
          // This would trigger ADD_EVIDENCE_NODE event
          console.log('File imported successfully:', result);
        }
      }
    }
  }
});

export { evidenceCanvasMachine };
export type { EvidenceCanvasContext, EvidenceCanvasEvent };