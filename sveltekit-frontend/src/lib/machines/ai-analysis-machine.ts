// AI Analysis State Machine - XState v5 compatible
// Manages AI-powered legal document analysis and recommendations

import { createMachine, assign, fromPromise } from 'xstate';

export interface AIAnalysisContext {
  prompt: string;
  context: {
    caseId?: string;
    documentIds: string[];
    analysisType: 'summary' | 'recommendation' | 'risk-assessment' | 'precedent-analysis';
  };
  options: {
    includeReferences: boolean;
    maxTokens: number;
    temperature: number;
    model?: string;
  };
  analysisResults: {
    summary?: string;
    recommendations?: string[];
    riskScore?: number;
    precedents?: any[];
    references?: any[];
    confidence?: number;
  };
  processingTime: number;
  tokensUsed: number;
  confidence: number;
  isStreaming: boolean;
  validationErrors: Record<string, string[]>;
  error: string | null;
}

export const aiAnalysisMachine = createMachine({
  id: 'aiAnalysis',
  initial: 'idle',
  types: {
    context: {} as AIAnalysisContext,
    events: {} as 
      | { type: 'START_ANALYSIS'; data: any }
      | { type: 'UPDATE_PROMPT'; prompt: string }
      | { type: 'UPDATE_OPTIONS'; options: any }
      | { type: 'STREAM_CHUNK'; chunk: string }
      | { type: 'RETRY' }
      | { type: 'RESET' }
  },
  context: {
    prompt: '',
    context: {
      documentIds: [],
      analysisType: 'summary'
    },
    options: {
      includeReferences: true,
      maxTokens: 1000,
      temperature: 0.7
    },
    analysisResults: {},
    processingTime: 0,
    tokensUsed: 0,
    confidence: 0,
    isStreaming: false,
    validationErrors: {},
    error: null
  },
  states: {
    idle: {
      on: {
        UPDATE_PROMPT: {
          actions: assign({
            prompt: ({ event }) => event.prompt,
            error: null
          })
        },
        UPDATE_OPTIONS: {
          actions: assign({
            options: ({ context, event }) => ({
              ...context.options,
              ...event.options
            })
          })
        },
        START_ANALYSIS: 'validating'
      }
    },
    validating: {
      invoke: {
        id: 'validateAnalysisRequest',
        input: ({ context }) => context,
        src: fromPromise(async ({ input }) => {
          const errors: Record<string, string[]> = {};
          
          const context = input as AIAnalysisContext;
          if (!context.prompt?.trim()) {
            errors.prompt = ['Analysis prompt is required'];
          }
          
          if (context.prompt && context.prompt.length < 10) {
            errors.prompt = ['Prompt must be at least 10 characters'];
          }
          
          if (context.prompt && context.prompt.length > 2000) {
            errors.prompt = ['Prompt too long (max 2000 characters)'];
          }
          
          if (context.options.maxTokens < 100 || context.options.maxTokens > 4000) {
            errors.maxTokens = ['Max tokens must be between 100 and 4000'];
          }
          
          if (context.options.temperature < 0 || context.options.temperature > 1) {
            errors.temperature = ['Temperature must be between 0 and 1'];
          }
          
          if (Object.keys(errors).length > 0) {
            throw { validationErrors: errors };
          }
          
          return context;
        }),
        onDone: {
          target: 'analyzing',
          actions: assign({
            validationErrors: {},
            error: null
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            validationErrors: ({ event }) => {
              const error = event.error as any;
              return error?.validationErrors || {};
            },
            error: 'Validation failed'
          })
        }
      }
    },
    analyzing: {
      entry: assign({
        isStreaming: true,
        analysisResults: {},
        processingTime: 0
      }),
      invoke: {
        id: 'performAIAnalysis',
        input: ({ context }) => context,
        src: fromPromise(async ({ input }) => {
          const context = input as AIAnalysisContext;
          const startTime = Date.now();
          
          // Prepare analysis request
          const analysisRequest = {
            prompt: context.prompt,
            context: context.context,
            options: context.options,
            streaming: true
          };
          
          // Call AI analysis API
          const response = await fetch('/api/ai/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(analysisRequest)
          });
          
          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Analysis failed: HTTP ${response.status}`);
          }
          
          // Handle streaming response
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();
          let analysisResults = {};
          let tokensUsed = 0;
          
          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              
              const chunk = decoder.decode(value);
              const lines = chunk.split('\n');
              
              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.slice(6));
                    
                    if (data.type === 'chunk') {
                      // Streaming text chunk - would emit event here
                      continue;
                    } else if (data.type === 'result') {
                      analysisResults = data.analysis;
                      tokensUsed = data.tokensUsed || 0;
                    }
                  } catch (e: any) {
                    console.warn('Failed to parse SSE data:', e);
                  }
                }
              }
            }
          }
          
          const processingTime = Date.now() - startTime;
          
          return {
            analysisResults,
            processingTime,
            tokensUsed,
            confidence: (analysisResults as any)?.confidence || 0.8
          };
        }),
        onDone: {
          target: 'completed',
          actions: assign({
            analysisResults: ({ event }) => event.output.analysisResults,
            processingTime: ({ event }) => event.output.processingTime,
            tokensUsed: ({ event }) => event.output.tokensUsed,
            confidence: ({ event }) => event.output.confidence,
            isStreaming: false,
            error: null
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: ({ event }) => {
              const error = event.error as any;
              return error?.message || 'Analysis failed';
            },
            isStreaming: false
          })
        }
      },
      on: {
        STREAM_CHUNK: {
          actions: assign({
            // Handle streaming chunks if needed
            analysisResults: ({ context, event }) => ({
              ...context.analysisResults,
              streamingText: ((context.analysisResults as any).streamingText || '') + (event as any).chunk
            })
          })
        }
      }
    },
    completed: {
      type: 'final',
      entry: assign({ isStreaming: false }),
      on: {
        RESET: {
          target: 'idle',
          actions: assign({
            prompt: '',
            analysisResults: {},
            processingTime: 0,
            tokensUsed: 0,
            confidence: 0,
            error: null,
            validationErrors: {}
          })
        },
        START_ANALYSIS: 'validating'
      }
    },
    error: {
      on: {
        RETRY: 'analyzing',
        RESET: {
          target: 'idle',
          actions: assign({
            error: null
          })
        }
      }
    }
  }
});

export default aiAnalysisMachine;