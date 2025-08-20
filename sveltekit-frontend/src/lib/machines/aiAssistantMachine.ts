// AI Assistant Machine with Ollama Cluster + Context7 Integration
// Uses XState v5 best practices with actors, invoke, and services
import { createMachine, assign, fromPromise, fromCallback } from "xstate";
import { productionServiceClient, services } from "../services/productionServiceClient.js";

// AI Assistant context interface
interface AIAssistantContext {
  currentQuery: string;
  response: string;
  conversationHistory: ConversationEntry[];
  isProcessing: boolean;
  model: string;
  temperature: number;
  maxTokens: number;
  context7Analysis?: Context7Analysis;
  ollamaClusterHealth: {
    primary: boolean;
    secondary: boolean;
    embeddings: boolean;
  };
  activeStreaming: boolean;
  streamBuffer: string;
  error: string | null;
  usage: {
    totalQueries: number;
    totalTokens: number;
    averageResponseTime: number;
  };
}

interface ConversationEntry {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    model: string;
    temperature: number;
    responseTime: number;
    tokenCount: number;
    context7Used: boolean;
  };
}

interface Context7Analysis {
  suggestions: string[];
  codeExamples: any[];
  documentation: string;
  confidence: number;
}

// AI Assistant events
type AIAssistantEvent =
  | { type: "SEND_MESSAGE"; message: string; useContext7?: boolean }
  | { type: "SET_MODEL"; model: string }
  | { type: "SET_TEMPERATURE"; temperature: number }
  | { type: "CLEAR_CONVERSATION" }
  | { type: "RETRY_LAST" }
  | { type: "STOP_GENERATION" }
  | { type: "START_STREAMING" }
  | { type: "STREAM_CHUNK"; chunk: string }
  | { type: "STREAM_END" }
  | { type: "CHECK_CLUSTER_HEALTH" }
  | { type: "ANALYZE_WITH_CONTEXT7"; topic: string }
  | { type: "ENHANCE_QUERY"; originalQuery: string };

export const aiAssistantMachine = createMachine({
  id: "aiAssistant",
  initial: "idle",
  context: {
    currentQuery: "",
    response: "",
    conversationHistory: [],
    isProcessing: false,
    model: "gemma3-legal",
    temperature: 0.7,
    maxTokens: 2048,
    ollamaClusterHealth: {
      primary: false,
      secondary: false,
      embeddings: false
    },
    activeStreaming: false,
    streamBuffer: "",
    error: null,
    usage: {
      totalQueries: 0,
      totalTokens: 0,
      averageResponseTime: 0
    }
  } as AIAssistantContext,
  types: {} as {
    context: AIAssistantContext;
    events: AIAssistantEvent;
  },
  states: {
    idle: {
      entry: ["clearError"],
      on: {
        SEND_MESSAGE: {
          target: "processing",
          actions: assign({
            currentQuery: ({ event }) => (event as any).message,
            isProcessing: () => true
          })
        },
        SET_MODEL: {
          actions: assign({
            model: ({ event }) => (event as any).model
          })
        },
        SET_TEMPERATURE: {
          actions: assign({
            temperature: ({ event }) => (event as any).temperature
          })
        },
        CLEAR_CONVERSATION: {
          actions: assign({
            conversationHistory: () => [],
            usage: ({ context }) => ({
              ...context.usage,
              totalQueries: 0,
              totalTokens: 0
            })
          })
        },
        CHECK_CLUSTER_HEALTH: "checkingClusterHealth",
        ANALYZE_WITH_CONTEXT7: "analyzingWithContext7"
      }
    },

    processing: {
      initial: "preparingQuery",
      states: {
        preparingQuery: {
          invoke: {
            id: "enhanceQuery",
            src: fromPromise(async ({ input }: { input: any }) => {
              const { query, useContext7 } = input;
              
              // Add user message to conversation
              const userEntry: ConversationEntry = {
                id: `user_${Date.now()}`,
                type: 'user',
                content: query,
                timestamp: new Date()
              };

              let enhancedQuery = query;
              let context7Analysis: Context7Analysis | undefined;

              // Enhance query with Context7 if requested
              if (useContext7) {
                try {
                  const analysis = await analyzeWithContext7(query);
                  context7Analysis = analysis;
                  enhancedQuery = `${query}\n\nContext7 Analysis:\n${analysis.documentation}`;
                } catch (error) {
                  console.warn('Context7 analysis failed:', error);
                }
              }

              return {
                userEntry,
                enhancedQuery,
                context7Analysis
              };
            }),
            input: ({ context, event }) => ({
              query: context.currentQuery,
              useContext7: (event as any).useContext7
            }),
            onDone: {
              target: "generatingResponse",
              actions: assign({
                conversationHistory: ({ context, event }) => [
                  ...context.conversationHistory,
                  (event as any).output.userEntry
                ],
                currentQuery: ({ event }) => (event as any).output.enhancedQuery,
                context7Analysis: ({ event }) => (event as any).output.context7Analysis
              })
            },
            onError: {
              target: "#aiAssistant.error",
              actions: assign({
                error: ({ event }) => `Query preparation failed: ${(event as any).error}`
              })
            }
          }
        },

        generatingResponse: {
          invoke: {
            id: "generateAIResponse",
            src: fromPromise(async ({ input }: { input: any }) => {
              const { query, model, temperature, maxTokens, conversationHistory } = input;
              const startTime = Date.now();

              try {
                // Use production service client with Ollama cluster
                const response = await services.queryRAG(query, {
                  model,
                  temperature,
                  maxTokens,
                  conversationHistory: conversationHistory.slice(-10) // Last 10 messages
                });

                const responseTime = Date.now() - startTime;
                
                // Create assistant response entry
                const assistantEntry: ConversationEntry = {
                  id: `assistant_${Date.now()}`,
                  type: 'assistant',
                  content: response.response || response.data?.response || 'No response',
                  timestamp: new Date(),
                  metadata: {
                    model,
                    temperature,
                    responseTime,
                    tokenCount: response.tokenCount || 0,
                    context7Used: !!input.context7Analysis
                  }
                };

                return {
                  response: assistantEntry.content,
                  assistantEntry,
                  responseTime,
                  tokenCount: response.tokenCount || 0
                };
              } catch (error) {
                console.error('AI response generation failed:', error);
                throw new Error(`AI generation failed: ${error}`);
              }
            }),
            input: ({ context }) => ({
              query: context.currentQuery,
              model: context.model,
              temperature: context.temperature,
              maxTokens: context.maxTokens,
              conversationHistory: context.conversationHistory,
              context7Analysis: context.context7Analysis
            }),
            onDone: {
              target: "#aiAssistant.idle",
              actions: assign({
                response: ({ event }) => (event as any).output.response,
                conversationHistory: ({ context, event }) => [
                  ...context.conversationHistory,
                  (event as any).output.assistantEntry
                ],
                usage: ({ context, event }) => ({
                  totalQueries: context.usage.totalQueries + 1,
                  totalTokens: context.usage.totalTokens + (event as any).output.tokenCount,
                  averageResponseTime: (
                    (context.usage.averageResponseTime * context.usage.totalQueries + 
                     (event as any).output.responseTime) / 
                    (context.usage.totalQueries + 1)
                  )
                }),
                isProcessing: () => false,
                currentQuery: () => "",
                context7Analysis: () => undefined
              })
            },
            onError: {
              target: "#aiAssistant.error",
              actions: assign({
                error: ({ event }) => `Response generation failed: ${(event as any).error}`,
                isProcessing: () => false
              })
            }
          }
        }
      },
      on: {
        STOP_GENERATION: {
          target: "idle",
          actions: assign({
            isProcessing: () => false,
            currentQuery: () => ""
          })
        }
      }
    },

    streaming: {
      invoke: {
        id: "streamResponse",
        src: fromCallback(({ input, sendBack }: { input: any; sendBack: any }) => {
          const { query, model, temperature } = input;
          
          // WebSocket streaming implementation
          const ws = new WebSocket(`ws://localhost:8094/ws/stream`);
          
          ws.onopen = () => {
            ws.send(JSON.stringify({
              query,
              model,
              temperature,
              stream: true
            }));
          };

          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (data.chunk) {
                sendBack({ type: 'STREAM_CHUNK', chunk: data.chunk });
              } else if (data.done) {
                sendBack({ type: 'STREAM_END' });
              }
            } catch (error) {
              console.error('Stream parsing error:', error);
            }
          };

          ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            sendBack({ type: 'STREAM_END' });
          };

          // Cleanup function
          return () => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.close();
            }
          };
        }),
        input: ({ context }) => ({
          query: context.currentQuery,
          model: context.model,
          temperature: context.temperature
        })
      },
      on: {
        STREAM_CHUNK: {
          actions: assign({
            streamBuffer: ({ context, event }) => 
              context.streamBuffer + (event as any).chunk
          })
        },
        STREAM_END: {
          target: "idle",
          actions: assign({
            response: ({ context }) => context.streamBuffer,
            conversationHistory: ({ context }) => [
              ...context.conversationHistory,
              {
                id: `user_${Date.now() - 1000}`,
                type: 'user' as const,
                content: context.currentQuery,
                timestamp: new Date(Date.now() - 1000)
              },
              {
                id: `assistant_${Date.now()}`,
                type: 'assistant' as const,
                content: context.streamBuffer,
                timestamp: new Date(),
                metadata: {
                  model: context.model,
                  temperature: context.temperature,
                  responseTime: 0,
                  tokenCount: 0,
                  context7Used: false
                }
              }
            ],
            streamBuffer: () => "",
            activeStreaming: () => false,
            isProcessing: () => false
          })
        },
        STOP_GENERATION: {
          target: "idle",
          actions: assign({
            activeStreaming: () => false,
            isProcessing: () => false,
            streamBuffer: () => ""
          })
        }
      }
    },

    checkingClusterHealth: {
      invoke: {
        id: "checkOllamaCluster",
        src: fromPromise(async () => {
          try {
            // Check all Ollama instances
            const healthChecks = await Promise.allSettled([
              fetch('http://localhost:11434/api/tags'),
              fetch('http://localhost:11435/api/tags'),
              fetch('http://localhost:11436/api/tags')
            ]);

            return {
              primary: healthChecks[0].status === 'fulfilled',
              secondary: healthChecks[1].status === 'fulfilled',
              embeddings: healthChecks[2].status === 'fulfilled'
            };
          } catch (error) {
            console.error('Cluster health check failed:', error);
            return {
              primary: false,
              secondary: false,
              embeddings: false
            };
          }
        }),
        onDone: {
          target: "idle",
          actions: assign({
            ollamaClusterHealth: ({ event }) => (event as any).output
          })
        },
        onError: {
          target: "idle",
          actions: assign({
            ollamaClusterHealth: () => ({
              primary: false,
              secondary: false,
              embeddings: false
            })
          })
        }
      }
    },

    analyzingWithContext7: {
      invoke: {
        id: "context7Analysis",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { topic } = input;
          
          try {
            // Import Context7 service dynamically
            const { getSvelte5Docs, getXStateDocs } = await import('../mcp-context72-get-library-docs.js');
            
            // Get relevant documentation
            const [svelteDocsResponse, xstateDocsResponse] = await Promise.all([
              getSvelte5Docs(topic),
              getXStateDocs(topic)
            ]);

            const analysis: Context7Analysis = {
              suggestions: [
                `Consider using Svelte 5 runes for ${topic}`,
                `XState actors can handle ${topic} workflows`,
                `Modern patterns available for ${topic} implementation`
              ],
              codeExamples: [
                ...(svelteDocsResponse.snippets || []),
                ...(xstateDocsResponse.snippets || [])
              ],
              documentation: `${svelteDocsResponse.content}\n\n${xstateDocsResponse.content}`,
              confidence: 0.85
            };

            return analysis;
          } catch (error) {
            console.error('Context7 analysis failed:', error);
            throw error;
          }
        }),
        input: ({ event }) => ({
          topic: (event as any).topic
        }),
        onDone: {
          target: "idle",
          actions: assign({
            context7Analysis: ({ event }) => (event as any).output
          })
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => `Context7 analysis failed: ${(event as any).error}`
          })
        }
      }
    },

    error: {
      entry: ["logError"],
      after: {
        5000: {
          target: "idle",
          actions: assign({
            error: () => null,
            isProcessing: () => false
          })
        }
      },
      on: {
        RETRY_LAST: {
          target: "processing",
          actions: assign({
            error: () => null
          })
        },
        CLEAR_CONVERSATION: {
          target: "idle",
          actions: assign({
            error: () => null,
            conversationHistory: () => [],
            isProcessing: () => false
          })
        }
      }
    }
  }
});

// Service implementations
export const aiAssistantServices = {
  // Services are now defined inline using fromPromise and fromCallback
};

// Action implementations
export const aiAssistantActions = {
  clearError: assign({
    error: () => null
  }),

  logError: ({ context }: { context: AIAssistantContext }) => {
    if (context.error) {
      console.error('AI Assistant Error:', context.error);
    }
  }
};

// Helper function for Context7 analysis
async function analyzeWithContext7(query: string): Promise<Context7Analysis> {
  try {
    // Import Context7 service
    const { getSvelte5Docs, getBitsUIv2Docs, getXStateDocs } = await import('../mcp-context72-get-library-docs.js');
    
    // Extract topics from query
    const topics = extractTopicsFromQuery(query);
    
    // Get documentation for relevant topics
    const docsPromises = topics.map(async (topic) => {
      try {
        return await getSvelte5Docs(topic);
      } catch {
        return null;
      }
    });

    const docsResults = await Promise.allSettled(docsPromises);
    const validDocs = docsResults
      .filter(result => result.status === 'fulfilled')
      .map(result => (result as any).value)
      .filter(Boolean);

    return {
      suggestions: [
        "Use modern Svelte 5 runes for reactive state",
        "Consider XState for complex state management",
        "Leverage Bits UI for accessible components"
      ],
      codeExamples: validDocs.flatMap(doc => doc.snippets || []),
      documentation: validDocs.map(doc => doc.content).join('\n\n'),
      confidence: validDocs.length > 0 ? 0.8 : 0.3
    };
  } catch (error) {
    console.error('Context7 analysis error:', error);
    throw error;
  }
}

// Helper function to extract topics from query
function extractTopicsFromQuery(query: string): string[] {
  const keywords = [
    'svelte', 'runes', 'state', 'component', 'store', 'reactive',
    'xstate', 'machine', 'actor', 'transition', 'context',
    'ui', 'button', 'form', 'input', 'dialog', 'modal'
  ];

  const queryLower = query.toLowerCase();
  return keywords.filter(keyword => queryLower.includes(keyword));
}

// Additional helper function for XState docs
async function getXStateDocs(topic: string) {
  // Placeholder for XState documentation retrieval
  return {
    content: `XState documentation for ${topic}`,
    snippets: []
  };
}