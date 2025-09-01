import { setup, assign, createActor, fromPromise } from 'xstate';
import { writable } from 'svelte/store';
import { productionServiceClient } from '$lib/services/production-service-client.js';

// Legal AI Application State Machine - XState v5
export interface Case {
  id: string;
  title: string;
  status: string;
  priority: string;
  category: string;
  [key: string]: unknown;
}

export interface Evidence {
  id: string;
  caseId: string;
  type: string;
  description?: string;
  fileUrl?: string;
  metadata?: Record<string, any>;
  [key: string]: unknown;
}

export interface LegalAIContext {
  user: {
    id: string | null;
    email: string | null;
    role: string | null;
    permissions: string[];
    isAuthenticated: boolean;
  };
  cases: {
    items: Case[];
    currentCase: Case | null;
    filters: {
      search: string;
      status: string;
      priority: string;
      category: string;
    };
    pagination: {
      page: number;
      limit: number;
      total: number;
    };
    loading: boolean;
    error: string | null;
  };
  ai: {
    isProcessing: boolean;
    currentQuery: string;
    lastResponse: any;
    error: string | null;
    models: {
      primary: string;
      embedding: string;
      available: string[];
    };
  };
  system: {
    connected: boolean;
    services: {
      database: boolean;
      redis: boolean;
      ollama: boolean;
      gpu: boolean;
    };
    metrics: {
      errorCount: number;
      performanceScore: number;
      uptime: number;
    };
  };
}

export type LegalAIEvent =
  | { type: 'AUTH.LOGIN'; credentials: { email: string; password: string } }
  | { type: 'AUTH.LOGOUT' }
  | { type: 'AUTH.REGISTER'; userData: any }
  | { type: 'CASES.LOAD'; filters?: any }
  | { type: 'CASES.SELECT'; case: Case }
  | { type: 'CASES.CREATE'; caseData: any }
  | { type: 'CASES.SEARCH'; query: string }
  | { type: 'AI.QUERY'; prompt: string; context?: any }
  | { type: 'SYSTEM.CHECK_STATUS' };

const initialContext: LegalAIContext = {
  user: {
    id: null,
    email: null,
    role: null,
    permissions: [],
    isAuthenticated: false,
  },
  cases: {
    items: [],
    currentCase: null,
    filters: {
      search: '',
      status: 'all',
      priority: 'all',
      category: 'all',
    },
    pagination: {
      page: 1,
      limit: 10,
      total: 0,
    },
    loading: false,
    error: null,
  },
  ai: {
    isProcessing: false,
    currentQuery: '',
    lastResponse: null,
    error: null,
    models: {
      primary: 'gemma3-legal',
      embedding: 'nomic-embed-text',
      available: ['gemma3-legal', 'gpt4-legal', 'llama2-legal']
    }
  },
  system: {
    connected: false,
    services: {
      database: false,
      redis: false,
      ollama: false,
      gpu: false
    },
    metrics: {
      errorCount: 0,
      performanceScore: 0,
      uptime: 0
    }
  }
};

export const legalAIMachine = setup({
  types: {} as {
    context: LegalAIContext;
    events: LegalAIEvent;
  },
  actions: {
    updateSystem: assign({
      system: ({ event }) => (event as any).output || {}
    }),
    setSystemError: assign({
      system: ({ context }) => ({
        ...context.system,
        connected: false
      })
    }),
    setUser: assign({
      user: ({ event }) => ({
        ...(event as any).output,
        isAuthenticated: true
      })
    }),
    clearUser: assign({
      user: () => ({
        id: null,
        email: null,
        role: null,
        permissions: [],
        isAuthenticated: false
      })
    }),
    setCases: assign({
      cases: ({ context, event }) => ({
        ...context.cases,
        items: (event as any).output || [],
        loading: false
      })
    }),
    setCurrentCase: assign({
      cases: ({ context, event }) => ({
        ...context.cases,
        currentCase: (event as any).case
      })
    }),
    setAIResponse: assign({
      ai: ({ context, event }) => ({
        ...context.ai,
        lastResponse: (event as any).output,
        isProcessing: false
      })
    }),
    setAIError: assign({
      ai: ({ context, event }) => ({
        ...context.ai,
        error: (event as any).error || 'AI processing failed',
        isProcessing: false
      })
    }),
    startAIProcessing: assign({
      ai: ({ context, event }) => ({
        ...context.ai,
        isProcessing: true,
        currentQuery: (event as any).prompt || '',
        error: null
      })
    })
  },
  actors: {
    checkSystemStatus: fromPromise(async () => {
      try {
        const clusterStatus = await productionServiceClient.getClusterStatus();
        const serviceHealth = await productionServiceClient.getServiceHealth();
        
        // Calculate overall health metrics
        const totalServices = clusterStatus.totalServices;
        const healthyServices = clusterStatus.healthyServices;
        const performanceScore = totalServices > 0 
          ? Math.round((healthyServices / totalServices) * 100) 
          : 0;

        return {
          connected: healthyServices > 0,
          services: {
            database: serviceHealth.some(s => s.service.includes('postgres') && s.status === 'healthy'),
            redis: serviceHealth.some(s => s.service.includes('redis') && s.status === 'healthy'),
            ollama: serviceHealth.some(s => s.service.includes('ollama') && s.status === 'healthy'),
            gpu: serviceHealth.some(s => s.service.includes('gpu') && s.status === 'healthy'),
            pgvector: serviceHealth.some(s => s.service.includes('pgvector') && s.status === 'healthy'),
            qdrant: serviceHealth.some(s => s.service.includes('qdrant') && s.status === 'healthy'),
            neo4j: serviceHealth.some(s => s.service.includes('neo4j') && s.status === 'healthy')
          },
          metrics: {
            errorCount: serviceHealth.reduce((acc, s) => acc + s.errorCount, 0),
            performanceScore,
            uptime: Date.now()
          }
        };
      } catch (error: any) {
        console.error('System status check failed:', error);
        return {
          connected: false,
          services: { database: false, redis: false, ollama: false, gpu: false, pgvector: false, qdrant: false, neo4j: false },
          metrics: { errorCount: 1, performanceScore: 0, uptime: 0 }
        };
      }
    }),
    authenticateUser: fromPromise(async ({ input }: { input: any }) => {
      try {
        const response = await productionServiceClient.callService('/api/auth/login', input.credentials, {
          timeout: 15000,
          priority: 'reliability'
        });

        if (response.success && response.data) {
          return {
            id: response.data.id || response.data.user?.id,
            email: response.data.email || input.credentials?.email,
            role: response.data.role || 'legal_professional',
            permissions: response.data.permissions || ['read:cases', 'write:cases', 'ai:query']
          };
        } else {
          throw new Error(response.error || 'Authentication failed');
        }
      } catch (error: any) {
        console.error('Authentication error:', error);
        throw new Error('Authentication service unavailable');
      }
    }),
    loadCases: fromPromise(async ({ input }: { input: any }) => {
      try {
        const response = await productionServiceClient.callService('/api/cases', input?.filters, {
          timeout: 10000,
          priority: 'performance'
        });

        if (response.success && response.data) {
          // Ensure returned data is array of cases
          const cases = Array.isArray(response.data) ? response.data : response.data.cases || [];
          return cases.map((caseData: any) => ({
            id: caseData.id,
            title: caseData.title,
            status: caseData.status || 'pending',
            priority: caseData.priority || 'medium', 
            category: caseData.category || 'general',
            createdAt: caseData.created_at || caseData.createdAt,
            updatedAt: caseData.updated_at || caseData.updatedAt,
            description: caseData.description,
            assignedTo: caseData.assigned_to || caseData.assignedTo
          }));
        } else {
          console.warn('Failed to load cases:', response.error);
          return [];
        }
      } catch (error: any) {
        console.error('Error loading cases:', error);
        return [];
      }
    }),
    processAIQuery: fromPromise(async ({ input }: { input: any }) => {
      try {
        const response = await productionServiceClient.queryRAG(input.prompt, input.context);

        if (response.success && response.data) {
          return {
            response: response.data.response || response.data.answer,
            confidence: response.data.confidence || 0.85,
            sources: response.data.sources || response.data.references || [],
            timestamp: new Date().toISOString(),
            model: response.data.model || 'gemma3-legal',
            protocol: response.protocol,
            latency: response.latency,
            metadata: response.data.metadata || {}
          };
        } else {
          throw new Error(response.error || 'AI query failed');
        }
      } catch (error: any) {
        console.error('AI query error:', error);
        throw new Error('AI service unavailable');
      }
    })
  }
}).createMachine({
  id: 'legalAI',
  initial: 'initializing',
  context: initialContext,
  states: {
    initializing: {
      invoke: {
        src: 'checkSystemStatus',
        onDone: {
          target: 'idle',
          actions: ['updateSystem']
        },
        onError: {
          target: 'error',
          actions: ['setSystemError']
        }
      }
    },
    idle: {
      on: {
        'AUTH.LOGIN': 'authenticating',
        'AUTH.REGISTER': 'registering',
        'CASES.LOAD': 'loadingCases',
        'CASES.CREATE': 'creatingCase',
        'AI.QUERY': 'processingAI',
        'SYSTEM.CHECK_STATUS': 'checkingStatus'
      }
    },
    authenticating: {
      invoke: {
        src: 'authenticateUser',
        input: ({ event }) => ({ credentials: (event as any).credentials }),
        onDone: {
          target: 'authenticated',
          actions: ['setUser']
        },
        onError: {
          target: 'idle',
          actions: ['clearUser']
        }
      }
    },
    authenticated: {
      initial: 'ready',
      states: {
        ready: {
          on: {
            'CASES.LOAD': '#legalAI.loadingCases',
            'AI.QUERY': '#legalAI.processingAI',
            'AUTH.LOGOUT': '#legalAI.idle'
          }
        }
      }
    },
    loadingCases: {
      invoke: {
        src: 'loadCases',
        onDone: {
          target: 'authenticated',
          actions: 'setCases'
        },
        onError: {
          target: 'authenticated'
        }
      }
    },
    processingAI: {
      entry: 'startAIProcessing',
      invoke: {
        src: 'processAIQuery',
        input: ({ event }) => ({ prompt: (event as any).prompt }),
        onDone: {
          target: 'authenticated',
          actions: 'setAIResponse'
        },
        onError: {
          target: 'authenticated',
          actions: 'setAIError'
        }
      }
    },
    error: {
      on: {
        'SYSTEM.CHECK_STATUS': 'initializing'
      }
    },
    // Placeholder states
    registering: {
      after: {
        1000: 'idle'
      }
    },
    creatingCase: {
      after: {
        1000: 'authenticated'
      }
    },
    checkingStatus: {
      after: {
        500: 'idle'
      }
    }
  }
});

// Create the actor
export const legalAIActor = createActor(legalAIMachine);
;
// Create Svelte store for reactive state
export const legalAIState = writable(legalAIActor.getSnapshot());
;
// Update store when state changes
legalAIActor.subscribe((snapshot) => {
  legalAIState.set(snapshot);
});

// Start the actor
legalAIActor.start();

export default legalAIActor;