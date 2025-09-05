// Document Upload State Machine - XState v5 compatible
// Manages file upload workflow with progress tracking and AI processing

import { createMachine, assign, fromPromise } from 'xstate';

export interface DocumentUploadContext {
  files: File[];
  uploadProgress: number;
  processingProgress: number;
  validationErrors: Record<string, string[]>;
  uploadedFiles: any[];
  aiResults: any;
  error: string | null;
  retryCount: number;
}

export const documentUploadMachine = createMachine({
  id: 'documentUpload',
  initial: 'idle',
  types: {
    context: {} as DocumentUploadContext,
    events: {} as 
      | { type: 'SELECT_FILES'; files: File[] }
      | { type: 'UPDATE_FORM'; data: any }
      | { type: 'VALIDATE_FORM'; data: any }
      | { type: 'SUBMIT'; data: any }
      | { type: 'RETRY' }
      | { type: 'RESET' }
  },
  context: {
    files: [],
    uploadProgress: 0,
    processingProgress: 0,
    validationErrors: {},
    uploadedFiles: [],
    aiResults: null,
    error: null,
    retryCount: 0
  },
  states: {
    idle: {
      on: {
        SELECT_FILES: {
          target: 'validating',
          actions: assign({
            files: ({ event }) => event.files,
            error: null
          })
        }
      }
    },
    validating: {
      invoke: {
        id: 'validateFiles',
        src: fromPromise(async ({ input }: { input: DocumentUploadContext }) => {
          const errors: Record<string, string[]> = {};
          
          if (input.files.length === 0) {
            errors.files = ['Please select at least one file'];
          }
          
          // Check file types and sizes
          const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'text/plain'];
          const maxSize = 10 * 1024 * 1024; // 10MB
          
          for (const file of input.files) {
            if (!allowedTypes.includes(file.type)) {
              errors.files = errors.files || [];
              errors.files.push(`${file.name}: File type not allowed`);
            }
            
            if (file.size > maxSize) {
              errors.files = errors.files || [];
              errors.files.push(`${file.name}: File too large (max 10MB)`);
            }
          }
          
          if (Object.keys(errors).length > 0) {
            throw { validationErrors: errors };
          }
          
          return input.files;
        }),
        input: ({ context }) => context,
        onDone: {
          target: 'validated',
          actions: assign({
            validationErrors: {},
            error: null
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            validationErrors: ({ event }) => (event as any).error?.validationErrors || {},
            error: 'File validation failed'
          })
        }
      }
    },
    validated: {
      on: {
        SUBMIT: 'uploading',
        SELECT_FILES: {
          target: 'validating',
          actions: assign({
            files: ({ event }) => event.files
          })
        }
      }
    },
    uploading: {
      entry: assign({
        uploadProgress: 0,
        retryCount: ({ context }) => context.retryCount + 1
      }),
      invoke: {
        id: 'uploadFiles',
        src: fromPromise(async ({ input }: { input: DocumentUploadContext }) => {
          const formData = new FormData();
          
          input.files.forEach((file, index) => {
            formData.append(`file_${index}`, file);
          });
          
          // Simulate upload progress
          const progressInterval = setInterval(() => {
            // This would normally be updated by the actual upload progress
          }, 100);
          
          try {
            const response = await fetch('/api/upload', {
              method: 'POST',
              body: formData
            });
            
            clearInterval(progressInterval);
            
            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            return response.json();
          } catch (error: any) {
            clearInterval(progressInterval);
            throw error;
          }
        }),
        input: ({ context }) => context,
        onDone: {
          target: 'processing',
          actions: assign({
            uploadedFiles: ({ event }) => event.output.files || [],
            uploadProgress: 100,
            error: null
          })
        },
        onError: [
          {
            guard: ({ context }) => context.retryCount < 3,
            target: 'retrying',
            actions: assign({
              error: ({ event }) => (event as any).error?.message || 'Upload failed'
            })
          },
          {
            target: 'failed',
            actions: assign({
              error: ({ event }) => (event as any).error?.message || 'Upload failed after retries'
            })
          }
        ]
      }
    },
    processing: {
      entry: assign({ processingProgress: 0 }),
      invoke: {
        id: 'processFiles',
        src: fromPromise(async ({ input }: { input: DocumentUploadContext }) => {
          // Process uploaded files with AI
          const processingResults = [];
          
          for (let i = 0; i < input.uploadedFiles.length; i++) {
            const file = input.uploadedFiles[i];
            
            // Simulate AI processing
            const response = await fetch('/api/ai/process-document', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                fileId: file.id,
                analysisType: 'full'
              })
            });
            
            if (!response.ok) {
              throw new Error(`Processing failed for ${file.name}`);
            }
            
            const result = await response.json();
            processingResults.push(result);
            
            // Update progress - this would normally be handled by events
            // context.processingProgress = ((i + 1) / context.uploadedFiles.length) * 100;
          }
          
          return {
            processedFiles: processingResults,
            summary: {
              totalFiles: input.uploadedFiles.length,
              successfulProcessing: processingResults.length,
              extractedText: processingResults.reduce((acc, r) => acc + (r.extractedText?.length || 0), 0)
            }
          };
        }),
        input: ({ context }) => context,
        onDone: {
          target: 'completed',
          actions: assign({
            aiResults: ({ event }) => event.output,
            processingProgress: 100,
            error: null
          })
        },
        onError: {
          target: 'failed',
          actions: assign({
            error: ({ event }) => (event as any).error?.message || 'Processing failed'
          })
        }
      }
    },
    retrying: {
      after: {
        2000: 'uploading'
      },
      on: {
        RETRY: 'uploading'
      }
    },
    completed: {
      type: 'final',
      on: {
        RESET: {
          target: 'idle',
          actions: assign({
            files: [],
            uploadProgress: 0,
            processingProgress: 0,
            validationErrors: {},
            uploadedFiles: [],
            aiResults: null,
            error: null,
            retryCount: 0
          })
        }
      }
    },
    failed: {
      on: {
        RETRY: 'uploading',
        RESET: {
          target: 'idle',
          actions: assign({
            error: null,
            retryCount: 0
          })
        }
      }
    }
  }
});

export default documentUploadMachine;