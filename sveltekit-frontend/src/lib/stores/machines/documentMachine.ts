// @ts-nocheck
/**
 * Document Processing XState Machine
 * Manages document lifecycle, AI analysis, and processing workflows
 */

import { createMachine, assign, type ActorRefFrom } from 'xstate';
import type { LegalDocument, Evidence, AIAnalysisResponse } from '$lib/types/unified-types';

export interface DocumentContext {
  documentId?: string;
  document?: LegalDocument;
  evidence?: Evidence;
  aiAnalysis?: AIAnalysisResponse;
  processingProgress: number;
  errors: string[];
  processingSteps: string[];
  currentStep?: string;
  extractedText?: string;
  embedding?: number[];
  entities?: any[];
  riskScore?: number;
  confidence?: number;
  processedAt?: Date;
}

export type DocumentEvent =
  | { type: 'UPLOAD'; document: LegalDocument }
  | { type: 'START_PROCESSING' }
  | { type: 'EXTRACT_TEXT'; text: string }
  | { type: 'ANALYZE_AI'; analysis: AIAnalysisResponse }
  | { type: 'GENERATE_EMBEDDING'; embedding: number[] }
  | { type: 'EXTRACT_ENTITIES'; entities: any[] }
  | { type: 'CALCULATE_RISK'; riskScore: number; confidence: number }
  | { type: 'PROCESSING_COMPLETE' }
  | { type: 'ERROR'; error: string }
  | { type: 'RETRY' }
  | { type: 'CANCEL' }
  | { type: 'RESET' };

export const documentMachine = createMachine({
  id: 'documentProcessor',
  types: {
    context: {} as DocumentContext,
    events: {} as DocumentEvent
  },
  initial: 'idle',
  context: {
    processingProgress: 0,
    errors: [],
    processingSteps: []
  },
  states: {
    idle: {
      on: {
        UPLOAD: {
          target: 'uploaded',
          actions: assign({
            document: ({ event }) => event.document,
            documentId: ({ event }) => event.document.id,
            processingProgress: 0,
            errors: [],
            processingSteps: ['Document uploaded']
          })
        }
      }
    },
    uploaded: {
      entry: assign({
        processingSteps: ({ context }) => [...context.processingSteps, 'Document ready for processing'],
        processingProgress: 10
      }),
      on: {
        START_PROCESSING: {
          target: 'processing',
          actions: assign({
            currentStep: 'starting_processing',
            processingSteps: ({ context }) => [...context.processingSteps, 'Starting AI processing']
          })
        },
        CANCEL: 'idle',
        RESET: 'idle'
      }
    },
    processing: {
      initial: 'extractingText',
      entry: assign({
        processingProgress: 20
      }),
      states: {
        extractingText: {
          entry: assign({
            currentStep: 'extracting_text',
            processingSteps: ({ context }) => [...context.processingSteps, 'Extracting text content']
          }),
          on: {
            EXTRACT_TEXT: {
              target: 'analyzingWithAI',
              actions: assign({
                extractedText: ({ event }) => event.text,
                processingProgress: 40,
                processingSteps: ({ context }) => [...context.processingSteps, 'Text extracted successfully']
              })
            },
            ERROR: {
              target: '#documentProcessor.error',
              actions: assign({
                errors: ({ context, event }) => [...context.errors, event.error]
              })
            }
          }
        },
        analyzingWithAI: {
          entry: assign({
            currentStep: 'analyzing_ai',
            processingSteps: ({ context }) => [...context.processingSteps, 'Running AI analysis']
          }),
          on: {
            ANALYZE_AI: {
              target: 'generatingEmbedding',
              actions: assign({
                aiAnalysis: ({ event }) => event.analysis,
                processingProgress: 60,
                processingSteps: ({ context }) => [...context.processingSteps, 'AI analysis completed']
              })
            },
            ERROR: {
              target: '#documentProcessor.error',
              actions: assign({
                errors: ({ context, event }) => [...context.errors, event.error]
              })
            }
          }
        },
        generatingEmbedding: {
          entry: assign({
            currentStep: 'generating_embedding',
            processingSteps: ({ context }) => [...context.processingSteps, 'Generating vector embeddings']
          }),
          on: {
            GENERATE_EMBEDDING: {
              target: 'extractingEntities',
              actions: assign({
                embedding: ({ event }) => event.embedding,
                processingProgress: 75,
                processingSteps: ({ context }) => [...context.processingSteps, 'Embeddings generated']
              })
            },
            ERROR: {
              target: '#documentProcessor.error',
              actions: assign({
                errors: ({ context, event }) => [...context.errors, event.error]
              })
            }
          }
        },
        extractingEntities: {
          entry: assign({
            currentStep: 'extracting_entities',
            processingSteps: ({ context }) => [...context.processingSteps, 'Extracting legal entities']
          }),
          on: {
            EXTRACT_ENTITIES: {
              target: 'calculatingRisk',
              actions: assign({
                entities: ({ event }) => event.entities,
                processingProgress: 85,
                processingSteps: ({ context }) => [...context.processingSteps, 'Entities extracted']
              })
            },
            ERROR: {
              target: '#documentProcessor.error',
              actions: assign({
                errors: ({ context, event }) => [...context.errors, event.error]
              })
            }
          }
        },
        calculatingRisk: {
          entry: assign({
            currentStep: 'calculating_risk',
            processingSteps: ({ context }) => [...context.processingSteps, 'Calculating risk assessment']
          }),
          on: {
            CALCULATE_RISK: {
              target: 'completing',
              actions: assign({
                riskScore: ({ event }) => event.riskScore,
                confidence: ({ event }) => event.confidence,
                processingProgress: 95,
                processingSteps: ({ context }) => [...context.processingSteps, 'Risk assessment completed']
              })
            },
            ERROR: {
              target: '#documentProcessor.error',
              actions: assign({
                errors: ({ context, event }) => [...context.errors, event.error]
              })
            }
          }
        },
        completing: {
          entry: assign({
            currentStep: 'completing',
            processingSteps: ({ context }) => [...context.processingSteps, 'Finalizing processing']
          }),
          on: {
            PROCESSING_COMPLETE: {
              target: '#documentProcessor.completed',
              actions: assign({
                processingProgress: 100,
                processedAt: () => new Date(),
                currentStep: 'completed',
                processingSteps: ({ context }) => [...context.processingSteps, 'Processing completed successfully']
              })
            },
            ERROR: {
              target: '#documentProcessor.error',
              actions: assign({
                errors: ({ context, event }) => [...context.errors, event.error]
              })
            }
          }
        }
      },
      on: {
        CANCEL: 'cancelled',
        ERROR: 'error'
      }
    },
    completed: {
      entry: assign({
        currentStep: 'completed'
      }),
      on: {
        RESET: {
          target: 'idle',
          actions: assign({
            document: undefined,
            documentId: undefined,
            aiAnalysis: undefined,
            processingProgress: 0,
            errors: [],
            processingSteps: [],
            currentStep: undefined,
            extractedText: undefined,
            embedding: undefined,
            entities: undefined,
            riskScore: undefined,
            confidence: undefined,
            processedAt: undefined
          })
        },
        START_PROCESSING: {
          target: 'processing',
          actions: assign({
            processingProgress: 20,
            errors: [],
            processingSteps: ({ context }) => [...context.processingSteps, 'Reprocessing started'],
            currentStep: 'starting_processing'
          })
        }
      }
    },
    error: {
      entry: assign({
        currentStep: 'error'
      }),
      on: {
        RETRY: {
          target: 'processing',
          actions: assign({
            errors: [],
            processingProgress: 20,
            processingSteps: ({ context }) => [...context.processingSteps, 'Retrying processing'],
            currentStep: 'starting_processing'
          })
        },
        RESET: {
          target: 'idle',
          actions: assign({
            document: undefined,
            documentId: undefined,
            aiAnalysis: undefined,
            processingProgress: 0,
            errors: [],
            processingSteps: [],
            currentStep: undefined,
            extractedText: undefined,
            embedding: undefined,
            entities: undefined,
            riskScore: undefined,
            confidence: undefined,
            processedAt: undefined
          })
        }
      }
    },
    cancelled: {
      entry: assign({
        currentStep: 'cancelled',
        processingSteps: ({ context }) => [...context.processingSteps, 'Processing cancelled']
      }),
      on: {
        RESET: {
          target: 'idle',
          actions: assign({
            document: undefined,
            documentId: undefined,
            aiAnalysis: undefined,
            processingProgress: 0,
            errors: [],
            processingSteps: [],
            currentStep: undefined,
            extractedText: undefined,
            embedding: undefined,
            entities: undefined,
            riskScore: undefined,
            confidence: undefined,
            processedAt: undefined
          })
        },
        START_PROCESSING: {
          target: 'processing',
          actions: assign({
            processingProgress: 20,
            errors: [],
            processingSteps: ({ context }) => [...context.processingSteps, 'Processing restarted'],
            currentStep: 'starting_processing'
          })
        }
      }
    }
  }
}, {
  // Machine options
  actions: {
    // Custom actions can be defined here if needed
  },
  guards: {
    // Custom guards can be defined here if needed
  },
  delays: {
    // Custom delays can be defined here if needed
  }
});

export type DocumentMachine = typeof documentMachine;
export type DocumentActor = ActorRefFrom<DocumentMachine>;

// Utility functions for working with the document machine
export function createDocumentActor() {
  return documentMachine;
}

export function isProcessingState(state: any): boolean {
  return state?.matches('processing');
}

export function isCompletedState(state: any): boolean {
  return state?.matches('completed');
}

export function hasErrors(context: DocumentContext): boolean {
  return context.errors && context.errors.length > 0;
}

export function getProcessingProgress(context: DocumentContext): number {
  return context.processingProgress || 0;
}

export function getCurrentStep(context: DocumentContext): string {
  return context.currentStep || 'idle';
}

export function getProcessingSteps(context: DocumentContext): string[] {
  return context.processingSteps || [];
}