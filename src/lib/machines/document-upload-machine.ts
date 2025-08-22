// Document Upload State Machine with XState
// Manages file upload, processing, and validation workflows

import { createMachine, assign, type InterpreterFrom } from 'xstate';
import { routerHelpers } from '$lib/services/multi-protocol-router';

// Types for document upload
export interface DocumentFile {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  lastModified: number;
}

export interface ProcessingStage {
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  startTime?: number;
  endTime?: number;
  error?: string;
  metadata?: Record<string, any>;
}

export interface DocumentUploadContext {
  files: DocumentFile[];
  currentFile: DocumentFile | null;
  currentStageIndex: number;
  stages: ProcessingStage[];
  results: Array<{
    fileId: string;
    success: boolean;
    documentId?: string;
    error?: string;
    metadata?: Record<string, any>;
  }>;
  settings: {
    maxFileSize: number;
    allowedTypes: string[];
    maxConcurrentUploads: number;
    preferredProtocol: 'auto' | 'quic' | 'grpc' | 'rest';
    enableGPUProcessing: boolean;
  };
  performance: {
    totalStartTime: number;
    totalProcessingTime: number;
    averageFileTime: number;
    protocolUsage: Record<string, number>;
  };
}

// Events for document upload
export type DocumentUploadEvent =
  | { type: 'ADD_FILES'; files: File[] }
  | { type: 'REMOVE_FILE'; fileId: string }
  | { type: 'START_UPLOAD' }
  | { type: 'CANCEL_UPLOAD' }
  | { type: 'RETRY_FILE'; fileId: string }
  | { type: 'UPDATE_SETTINGS'; settings: Partial<DocumentUploadContext['settings']> }
  | { type: 'CLEAR_RESULTS' }
  | { type: 'STAGE_PROGRESS'; progress: number }
  | { type: 'STAGE_COMPLETE'; result: unknown }
  | { type: 'STAGE_ERROR'; error: string };

// Default processing stages
const defaultStages: ProcessingStage[] = [
  { name: 'Validation', status: 'pending', progress: 0 },
  { name: 'Upload', status: 'pending', progress: 0 },
  { name: 'Text Extraction', status: 'pending', progress: 0 },
  { name: 'Chunking', status: 'pending', progress: 0 },
  { name: 'Embedding Generation', status: 'pending', progress: 0 },
  { name: 'Vector Storage', status: 'pending', progress: 0 },
  { name: 'Indexing', status: 'pending', progress: 0 },
  { name: 'Semantic Analysis', status: 'pending', progress: 0 }
];

const defaultContext: DocumentUploadContext = {
  files: [],
  currentFile: null,
  currentStageIndex: 0,
  stages: [...defaultStages],
  results: [],
  settings: {
    maxFileSize: 100 * 1024 * 1024, // 100MB
    allowedTypes: ['.pdf', '.txt', '.docx', '.doc', '.json', '.md'],
    maxConcurrentUploads: 3,
    preferredProtocol: 'auto',
    enableGPUProcessing: true
  },
  performance: {
    totalStartTime: 0,
    totalProcessingTime: 0,
    averageFileTime: 0,
    protocolUsage: { quic: 0, grpc: 0, rest: 0 }
  }
};

// Services for document upload
const documentUploadServices = {
  validateFile: async (context: DocumentUploadContext, event: unknown) => {
    const { file } = context.currentFile!;
    
    // Check file size
    if (file.size > context.settings.maxFileSize) {
      throw new Error(`File size exceeds maximum allowed size of ${Math.round(context.settings.maxFileSize / 1024 / 1024)}MB`);
    }

    // Check file type
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!context.settings.allowedTypes.includes(extension)) {
      throw new Error(`File type ${extension} is not supported. Allowed types: ${context.settings.allowedTypes.join(', ')}`);
    }

    // Check for malicious content (basic check)
    if (file.name.includes('../') || file.name.includes('..\\')) {
      throw new Error('Invalid file name detected');
    }

    return { valid: true, fileType: extension };
  },

  uploadDocument: async (context: DocumentUploadContext, event: unknown) => {
    const { file } = context.currentFile!;
    
    const uploadData = {
      title: file.name,
      type: file.type || 'application/octet-stream',
      caseId: event.caseId || 'default',
      description: event.description || '',
      tags: event.tags || [],
      file: file,
      processingOptions: {
        extractText: true,
        generateEmbeddings: true,
        createSummary: true,
        analyzeEntities: true,
        enableGPU: context.settings.enableGPUProcessing
      }
    };

    const result = await routerHelpers.documentUpload(uploadData, {
      preferredProtocol: context.settings.preferredProtocol === 'auto' ? undefined : context.settings.preferredProtocol,
      timeout: 120000 // 2 minutes timeout
    });

    return result;
  },

  processStage: async (context: DocumentUploadContext, event: unknown) => {
    const currentStage = context.stages[context.currentStageIndex];
    
    // Simulate stage processing with realistic timing
    const stageTimings = {
      'Validation': 500,
      'Upload': 2000,
      'Text Extraction': 3000,
      'Chunking': 1500,
      'Embedding Generation': 5000,
      'Vector Storage': 2000,
      'Indexing': 1000,
      'Semantic Analysis': 4000
    };

    const duration = stageTimings[currentStage.name as keyof typeof stageTimings] || 1000;
    
    // Simulate progress updates
    return new Promise((resolve) => {
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          resolve({
            stage: currentStage.name,
            completed: true,
            duration,
            metadata: {
              processedAt: Date.now(),
              gpuAccelerated: context.settings.enableGPUProcessing && 
                ['Embedding Generation', 'Semantic Analysis'].includes(currentStage.name)
            }
          });
        }
        
        // Send progress update
        self.postMessage?.({
          type: 'STAGE_PROGRESS',
          progress: Math.min(progress, 100)
        });
      }, duration / 10);
    });
  }
};

// Document upload state machine
export const documentUploadMachine = createMachine({
  id: 'documentUpload',
  initial: 'idle',
  context: defaultContext,
  states: {
    idle: {
      on: {
        ADD_FILES: {
          actions: 'addFiles'
        },
        START_UPLOAD: [
          {
            target: 'uploading',
            guard: 'hasFilesToUpload'
          },
          {
            target: 'error',
            actions: 'setError'
          }
        ],
        UPDATE_SETTINGS: {
          actions: 'updateSettings'
        },
        CLEAR_RESULTS: {
          actions: 'clearResults'
        }
      }
    },

    uploading: {
      entry: ['initializeUpload', 'selectNextFile'],
      initial: 'processing_file',
      states: {
        processing_file: {
          initial: 'validating',
          states: {
            validating: {
              entry: 'setStageProcessing',
              invoke: {
                id: 'validateFile',
                src: 'validateFile',
                onDone: {
                  target: 'uploading_file',
                  actions: 'completeStage'
                },
                onError: {
                  target: '#documentUpload.error',
                  actions: 'setStageError'
                }
              }
            },

            uploading_file: {
              entry: 'setStageProcessing',
              invoke: {
                id: 'uploadDocument',
                src: 'uploadDocument',
                onDone: {
                  target: 'processing_stages',
                  actions: 'completeStage'
                },
                onError: {
                  target: '#documentUpload.error',
                  actions: 'setStageError'
                }
              }
            },

            processing_stages: {
              initial: 'processing',
              states: {
                processing: {
                  entry: 'setStageProcessing',
                  invoke: {
                    id: 'processStage',
                    src: 'processStage',
                    onDone: [
                      {
                        target: 'next_stage',
                        actions: 'completeStage',
                        guard: 'hasMoreStages'
                      },
                      {
                        target: 'completed',
                        actions: 'completeStage'
                      }
                    ],
                    onError: {
                      target: '#documentUpload.error',
                      actions: 'setStageError'
                    }
                  },
                  on: {
                    STAGE_PROGRESS: {
                      actions: 'updateStageProgress'
                    }
                  }
                },

                next_stage: {
                  always: {
                    target: 'processing',
                    actions: 'moveToNextStage'
                  }
                },

                completed: {
                  always: [
                    {
                      target: '#documentUpload.uploading.next_file',
                      actions: 'recordFileResult',
                      guard: 'hasMoreFiles'
                    },
                    {
                      target: '#documentUpload.completed',
                      actions: 'recordFileResult'
                    }
                  ]
                }
              }
            }
          }
        },

        next_file: {
          always: {
            target: 'processing_file',
            actions: ['selectNextFile', 'resetStages']
          }
        }
      },
      on: {
        CANCEL_UPLOAD: {
          target: 'idle',
          actions: 'cancelUpload'
        }
      }
    },

    completed: {
      entry: 'calculatePerformanceMetrics',
      on: {
        ADD_FILES: {
          target: 'idle',
          actions: 'addFiles'
        },
        START_UPLOAD: [
          {
            target: 'uploading',
            guard: 'hasFilesToUpload'
          }
        ],
        CLEAR_RESULTS: {
          target: 'idle',
          actions: 'clearResults'
        }
      }
    },

    error: {
      on: {
        RETRY_FILE: {
          target: 'uploading',
          actions: 'retryCurrentFile'
        },
        ADD_FILES: {
          target: 'idle',
          actions: 'addFiles'
        },
        CLEAR_RESULTS: {
          target: 'idle',
          actions: 'clearResults'
        }
      }
    }
  },

  on: {
    REMOVE_FILE: {
      actions: 'removeFile'
    }
  }
}, {
  services: documentUploadServices,
  guards: {
    hasFilesToUpload: (context) => context.files.length > 0,
    hasMoreFiles: (context) => {
      const currentIndex = context.files.findIndex(f => f.id === context.currentFile?.id);
      return currentIndex < context.files.length - 1;
    },
    hasMoreStages: (context) => context.currentStageIndex < context.stages.length - 3 // Skip last stages handled separately
  },
  actions: {
    addFiles: assign({
      files: (context, event) => {
        const newFiles = event.files.map((file: File) => ({
          id: `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          file,
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: file.lastModified
        }));
        return [...context.files, ...newFiles];
      }
    }),

    removeFile: assign({
      files: (context, event) => context.files.filter(f => f.id !== event.fileId)
    }),

    initializeUpload: assign({
      performance: (context) => ({
        ...context.performance,
        totalStartTime: Date.now()
      })
    }),

    selectNextFile: assign({
      currentFile: (context) => {
        if (context.files.length === 0) return null;
        
        const processedIds = context.results.map(r => r.fileId);
        const nextFile = context.files.find(f => !processedIds.includes(f.id));
        return nextFile || null;
      },
      currentStageIndex: 0
    }),

    resetStages: assign({
      stages: () => defaultStages.map(stage => ({ ...stage, status: 'pending', progress: 0 })),
      currentStageIndex: 0
    }),

    setStageProcessing: assign({
      stages: (context) => {
        const stages = [...context.stages];
        stages[context.currentStageIndex] = {
          ...stages[context.currentStageIndex],
          status: 'processing',
          startTime: Date.now()
        };
        return stages;
      }
    }),

    completeStage: assign({
      stages: (context, event) => {
        const stages = [...context.stages];
        stages[context.currentStageIndex] = {
          ...stages[context.currentStageIndex],
          status: 'completed',
          progress: 100,
          endTime: Date.now(),
          metadata: event.data
        };
        return stages;
      }
    }),

    setStageError: assign({
      stages: (context, event) => {
        const stages = [...context.stages];
        stages[context.currentStageIndex] = {
          ...stages[context.currentStageIndex],
          status: 'failed',
          error: event.data.message || 'Unknown error',
          endTime: Date.now()
        };
        return stages;
      }
    }),

    updateStageProgress: assign({
      stages: (context, event) => {
        const stages = [...context.stages];
        stages[context.currentStageIndex] = {
          ...stages[context.currentStageIndex],
          progress: event.progress
        };
        return stages;
      }
    }),

    moveToNextStage: assign({
      currentStageIndex: (context) => context.currentStageIndex + 1
    }),

    recordFileResult: assign({
      results: (context, event) => [
        ...context.results,
        {
          fileId: context.currentFile!.id,
          success: !event.data?.error,
          documentId: event.data?.documentId,
          error: event.data?.error,
          metadata: event.data
        }
      ]
    }),

    calculatePerformanceMetrics: assign({
      performance: (context) => {
        const totalTime = Date.now() - context.performance.totalStartTime;
        const successfulFiles = context.results.filter(r => r.success).length;
        
        return {
          ...context.performance,
          totalProcessingTime: totalTime,
          averageFileTime: successfulFiles > 0 ? totalTime / successfulFiles : 0
        };
      }
    }),

    updateSettings: assign({
      settings: (context, event) => ({
        ...context.settings,
        ...event.settings
      })
    }),

    clearResults: assign({
      files: [],
      currentFile: null,
      currentStageIndex: 0,
      stages: [...defaultStages],
      results: []
    }),

    cancelUpload: assign({
      currentFile: null,
      currentStageIndex: 0,
      stages: [...defaultStages]
    }),

    retryCurrentFile: assign({
      currentStageIndex: 0,
      stages: [...defaultStages]
    }),

    setError: (context, event) => {
      console.error('Document upload error:', event);
    }
  }
});

// Type for the document upload service
export type DocumentUploadService = InterpreterFrom<typeof documentUploadMachine>;

// Helper functions for common operations
export const documentUploadActions = {
  addFiles: (files: File[]) => ({
    type: 'ADD_FILES' as const,
    files
  }),

  removeFile: (fileId: string) => ({
    type: 'REMOVE_FILE' as const,
    fileId
  }),

  startUpload: () => ({
    type: 'START_UPLOAD' as const
  }),

  cancelUpload: () => ({
    type: 'CANCEL_UPLOAD' as const
  }),

  retryFile: (fileId: string) => ({
    type: 'RETRY_FILE' as const,
    fileId
  }),

  updateSettings: (settings: Partial<DocumentUploadContext['settings']>) => ({
    type: 'UPDATE_SETTINGS' as const,
    settings
  }),

  clearResults: () => ({
    type: 'CLEAR_RESULTS' as const
  })
};

// Selectors for derived state
export const documentUploadSelectors = {
  isIdle: (state: unknown) => state.matches('idle'),
  isUploading: (state: unknown) => state.matches('uploading'),
  isCompleted: (state: unknown) => state.matches('completed'),
  hasError: (state: unknown) => state.matches('error'),
  
  currentProgress: (context: DocumentUploadContext) => {
    if (!context.currentFile) return 0;
    
    const completedStages = context.stages.filter(s => s.status === 'completed').length;
    const currentStageProgress = context.stages[context.currentStageIndex]?.progress || 0;
    
    return ((completedStages + currentStageProgress / 100) / context.stages.length) * 100;
  },
  
  overallProgress: (context: DocumentUploadContext) => {
    const totalFiles = context.files.length;
    if (totalFiles === 0) return 0;
    
    const completedFiles = context.results.length;
    const currentFileProgress = documentUploadSelectors.currentProgress(context);
    
    return ((completedFiles + currentFileProgress / 100) / totalFiles) * 100;
  },
  
  successRate: (context: DocumentUploadContext) => {
    if (context.results.length === 0) return 0;
    const successful = context.results.filter(r => r.success).length;
    return (successful / context.results.length) * 100;
  }
};