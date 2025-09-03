/**
 * XState Idle Detection Machine with RabbitMQ Self-Prompting Integration
 * Detects user idle states and triggers autonomous background processing
 */

import { createMachine, assign, interpret, type ActorRefFrom } from 'xstate';

// Types for idle detection and self-prompting
export interface IdleDetectionContext {
  userId?: string;
  sessionId: string;
  lastActivity: number;
  idleTimeout: number; // milliseconds
  backgroundJobsEnabled: boolean;
  currentJob?: BackgroundJob;
  jobQueue: BackgroundJob[];
  neo4jConnected: boolean;
  minioConnected: boolean;
  rabbitmqConnected: boolean;
  selfPromptingHistory: SelfPrompt[];
  performanceMetrics: {
    jobsCompleted: number;
    averageProcessingTime: number;
    successRate: number;
    lastJobTimestamp: number;
  };
}

export interface BackgroundJob {
  id: string;
  type: 'document_analysis' | 'case_clustering' | 'legal_research' | 'citation_validation' | 'self_prompting';
  priority: 'low' | 'medium' | 'high' | 'critical';
  payload: any;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  retryCount: number;
  maxRetries: number;
  estimatedDuration: number;
  dependencies?: string[]; // Other job IDs this depends on
}

export interface SelfPrompt {
  id: string;
  prompt: string;
  context: any;
  response?: string;
  confidence: number;
  timestamp: number;
  triggerReason: 'idle_detected' | 'pattern_recognition' | 'scheduled' | 'user_behavior';
  processedByNeo4j: boolean;
  minioArtifacts: string[]; // File paths stored in MinIO
}

// XState Events
export type IdleDetectionEvent =
  | { type: 'USER_ACTIVITY'; timestamp: number }
  | { type: 'START_IDLE_DETECTION' }
  | { type: 'STOP_IDLE_DETECTION' }
  | { type: 'IDLE_TIMEOUT' }
  | { type: 'QUEUE_BACKGROUND_JOB'; job: Omit<BackgroundJob, 'id' | 'createdAt' | 'status'> }
  | { type: 'JOB_COMPLETED'; jobId: string; result: any }
  | { type: 'JOB_FAILED'; jobId: string; error: string }
  | { type: 'ENABLE_SELF_PROMPTING' }
  | { type: 'DISABLE_SELF_PROMPTING' }
  | { type: 'NEO4J_CONNECTED'; connected: boolean }
  | { type: 'MINIO_CONNECTED'; connected: boolean }
  | { type: 'RABBITMQ_CONNECTED'; connected: boolean }
  | { type: 'GENERATE_SELF_PROMPT'; context: any }
  | { type: 'SELF_PROMPT_COMPLETED'; promptId: string; response: string; artifacts: string[] };

// Services for background operations
const idleDetectionServices = {
  // Monitor user activity patterns
  monitorActivity: () => (callback: any) => {
    const handleActivity = (event: Event) => {
      callback({ type: 'USER_ACTIVITY', timestamp: Date.now() });
    };

    // Monitor various user activity types
    const events = ['mousedown', 'mousemove', 'keydown', 'scroll', 'focus', 'blur'];
    events.forEach(eventType => {
      window.addEventListener(eventType, handleActivity, { passive: true });
    });

    return () => {
      events.forEach(eventType => {
        window.removeEventListener(eventType, handleActivity);
      });
    };
  },

  // Idle timeout checker
  idleTimer: (context: IdleDetectionContext) => (callback: any) => {
    const timer = setTimeout(() => {
      callback({ type: 'IDLE_TIMEOUT' });
    }, context.idleTimeout);

    return () => clearTimeout(timer);
  },

  // Process background jobs via RabbitMQ
  processBackgroundJobs: (context: IdleDetectionContext) => async () => {
    console.log('🐰 Starting RabbitMQ background job processing...');
    
    const job = context.jobQueue[0];
    if (!job) return;

    try {
      // Publish job to RabbitMQ
      const response = await fetch('/api/rabbitmq/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exchange: 'legal.background',
          routingKey: job.type,
          message: {
            jobId: job.id,
            type: job.type,
            priority: job.priority,
            payload: job.payload,
            sessionId: context.sessionId,
            userId: context.userId,
            timestamp: Date.now()
          },
          headers: {
            messageType: 'background_job',
            priority: job.priority,
            retryCount: job.retryCount.toString()
          }
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ Background job queued:', result);
        return { jobId: job.id, status: 'queued', messageId: result.messageId };
      } else {
        throw new Error(`Failed to queue job: ${response.statusText}`);
      }

    } catch (error) {
      console.error('❌ Background job processing failed:', error);
      throw error;
    }
  },

  // Generate self-prompting queries
  generateSelfPrompt: (context: IdleDetectionContext) => async () => {
    console.log('🧠 Generating self-prompting query...');
    
    try {
      // Analyze current system state and generate intelligent prompts
      const systemContext = {
        lastActivity: context.lastActivity,
        sessionDuration: Date.now() - context.lastActivity,
        completedJobs: context.performanceMetrics.jobsCompleted,
        successRate: context.performanceMetrics.successRate,
        availableServices: {
          neo4j: context.neo4jConnected,
          minio: context.minioConnected,
          rabbitmq: context.rabbitmqConnected
        }
      };

      // Generate contextual prompts based on system state
      const prompts = await generateContextualPrompts(systemContext);
      const selectedPrompt = selectBestPrompt(prompts, context.selfPromptingHistory);

      if (selectedPrompt) {
        // Store prompt context in MinIO for persistence
        const minioResponse = await fetch('/api/v1/minio/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            bucket: 'self-prompting',
            key: `prompts/${selectedPrompt.id}.json`,
            data: selectedPrompt,
            metadata: {
              type: 'self_prompt',
              sessionId: context.sessionId,
              timestamp: selectedPrompt.timestamp.toString()
            }
          })
        });

        // Send prompt to Neo4j for graph analysis
        if (context.neo4jConnected) {
          await storePromptInNeo4j(selectedPrompt, context.sessionId);
        }

        return selectedPrompt;
      }

    } catch (error) {
      console.error('❌ Self-prompt generation failed:', error);
      throw error;
    }
  },

  // Connect to backend services
  connectBackendServices: () => async () => {
    console.log('🔌 Connecting to backend services...');
    
    const serviceChecks = await Promise.allSettled([
      checkNeo4jConnection(),
      checkMinioConnection(),
      checkRabbitMQConnection()
    ]);

    return {
      neo4j: serviceChecks[0].status === 'fulfilled' && serviceChecks[0].value,
      minio: serviceChecks[1].status === 'fulfilled' && serviceChecks[1].value,
      rabbitmq: serviceChecks[2].status === 'fulfilled' && serviceChecks[2].value
    };
  }
};

// Actions for state transitions
const idleDetectionActions = {
  // Update last activity timestamp
  updateActivity: assign({
    lastActivity: (_, event) => 
      event.type === 'USER_ACTIVITY' ? event.timestamp : Date.now()
  }),

  // Queue a new background job
  queueBackgroundJob: assign({
    jobQueue: (context, event) => {
      if (event.type !== 'QUEUE_BACKGROUND_JOB') return context.jobQueue;
      
      const newJob: BackgroundJob = {
        ...event.job,
        id: crypto.randomUUID(),
        createdAt: Date.now(),
        status: 'pending',
        retryCount: 0
      };
      
      return [...context.jobQueue, newJob].sort((a, b) => {
        // Sort by priority: critical > high > medium > low
        const priorities = { critical: 4, high: 3, medium: 2, low: 1 };
        return priorities[b.priority] - priorities[a.priority];
      });
    }
  }),

  // Start processing the next job
  startJobProcessing: assign({
    currentJob: (context) => context.jobQueue[0] || null,
    jobQueue: (context) => context.jobQueue.slice(1)
  }),

  // Mark job as completed
  completeJob: assign({
    currentJob: () => null,
    performanceMetrics: (context, event) => {
      if (event.type !== 'JOB_COMPLETED') return context.performanceMetrics;
      
      const processingTime = Date.now() - (context.currentJob?.startedAt || Date.now());
      const totalJobs = context.performanceMetrics.jobsCompleted + 1;
      const avgTime = (context.performanceMetrics.averageProcessingTime * context.performanceMetrics.jobsCompleted + processingTime) / totalJobs;
      
      return {
        jobsCompleted: totalJobs,
        averageProcessingTime: avgTime,
        successRate: totalJobs > 0 ? (totalJobs / (totalJobs + 1)) * 100 : 100, // Simplified success rate
        lastJobTimestamp: Date.now()
      };
    }
  }),

  // Store self-prompt in history
  storeSelfPrompt: assign({
    selfPromptingHistory: (context, event) => {
      if (event.type !== 'SELF_PROMPT_COMPLETED') return context.selfPromptingHistory;
      
      const prompt: SelfPrompt = {
        id: event.promptId,
        prompt: '', // Would be populated from the original prompt
        context: {},
        response: event.response,
        confidence: 0.8, // Would be calculated based on response quality
        timestamp: Date.now(),
        triggerReason: 'idle_detected',
        processedByNeo4j: true,
        minioArtifacts: event.artifacts
      };
      
      return [prompt, ...context.selfPromptingHistory].slice(0, 100); // Keep last 100 prompts
    }
  }),

  // Update service connection status
  updateServiceConnections: assign({
    neo4jConnected: (context, event) => {
      if (event.type === 'NEO4J_CONNECTED') return event.connected;
      return context.neo4jConnected;
    },
    minioConnected: (context, event) => {
      if (event.type === 'MINIO_CONNECTED') return event.connected;
      return context.minioConnected;
    },
    rabbitmqConnected: (context, event) => {
      if (event.type === 'RABBITMQ_CONNECTED') return event.connected;
      return context.rabbitmqConnected;
    }
  })
};

// Guards for conditional logic
const idleDetectionGuards = {
  isIdle: (context: IdleDetectionContext) => {
    return Date.now() - context.lastActivity > context.idleTimeout;
  },
  
  hasBackgroundJobsEnabled: (context: IdleDetectionContext) => {
    return context.backgroundJobsEnabled;
  },
  
  hasQueuedJobs: (context: IdleDetectionContext) => {
    return context.jobQueue.length > 0;
  },
  
  allServicesConnected: (context: IdleDetectionContext) => {
    return context.neo4jConnected && context.minioConnected && context.rabbitmqConnected;
  }
};

// Main XState machine
export const idleDetectionMachine = createMachine<IdleDetectionContext, IdleDetectionEvent>({
  id: 'idleDetection',
  initial: 'initializing',
  
  context: {
    sessionId: crypto.randomUUID(),
    lastActivity: Date.now(),
    idleTimeout: 5 * 60 * 1000, // 5 minutes default
    backgroundJobsEnabled: true,
    jobQueue: [],
    neo4jConnected: false,
    minioConnected: false,
    rabbitmqConnected: false,
    selfPromptingHistory: [],
    performanceMetrics: {
      jobsCompleted: 0,
      averageProcessingTime: 0,
      successRate: 100,
      lastJobTimestamp: 0
    }
  },

  states: {
    initializing: {
      entry: ['updateActivity'],
      invoke: {
        id: 'connectServices',
        src: 'connectBackendServices',
        onDone: {
          target: 'monitoring',
          actions: [
            assign({
              neo4jConnected: (_, event) => event.data.neo4j,
              minioConnected: (_, event) => event.data.minio,
              rabbitmqConnected: (_, event) => event.data.rabbitmq
            })
          ]
        },
        onError: {
          target: 'monitoring',
          actions: [(_, event) => console.warn('⚠️ Service connection partially failed:', event.data)]
        }
      }
    },

    monitoring: {
      invoke: {
        id: 'activityMonitor',
        src: 'monitorActivity'
      },
      
      initial: 'active',
      
      states: {
        active: {
          invoke: {
            id: 'idleTimer',
            src: 'idleTimer'
          },
          
          on: {
            USER_ACTIVITY: {
              actions: ['updateActivity']
            },
            IDLE_TIMEOUT: {
              target: 'idle',
              cond: 'hasBackgroundJobsEnabled'
            }
          }
        },

        idle: {
          entry: [() => console.log('😴 User idle detected - starting background processing')],
          
          initial: 'checking_services',
          
          states: {
            checking_services: {
              always: [
                {
                  target: 'generating_prompts',
                  cond: 'allServicesConnected'
                },
                {
                  target: 'waiting_for_services'
                }
              ]
            },

            waiting_for_services: {
              after: {
                10000: 'generating_prompts' // Wait 10 seconds then proceed anyway
              },
              on: {
                NEO4J_CONNECTED: {
                  actions: ['updateServiceConnections']
                },
                MINIO_CONNECTED: {
                  actions: ['updateServiceConnections']
                },
                RABBITMQ_CONNECTED: {
                  actions: ['updateServiceConnections']
                }
              }
            },

            generating_prompts: {
              entry: [() => console.log('🧠 Generating self-prompting queries...')],
              invoke: {
                id: 'generateSelfPrompt',
                src: 'generateSelfPrompt',
                onDone: {
                  target: 'processing_jobs',
                  actions: [
                    assign({
                      jobQueue: (context, event) => {
                        const selfPromptJob: BackgroundJob = {
                          id: crypto.randomUUID(),
                          type: 'self_prompting',
                          priority: 'medium',
                          payload: event.data,
                          createdAt: Date.now(),
                          status: 'pending',
                          retryCount: 0,
                          maxRetries: 3,
                          estimatedDuration: 30000 // 30 seconds
                        };
                        return [selfPromptJob, ...context.jobQueue];
                      }
                    })
                  ]
                },
                onError: {
                  target: 'processing_jobs',
                  actions: [(_, event) => console.warn('⚠️ Self-prompt generation failed:', event.data)]
                }
              }
            },

            processing_jobs: {
              always: [
                {
                  target: 'job_execution',
                  cond: 'hasQueuedJobs'
                },
                {
                  target: '../active' // Return to active monitoring
                }
              ]
            },

            job_execution: {
              entry: ['startJobProcessing'],
              invoke: {
                id: 'processJob',
                src: 'processBackgroundJobs',
                onDone: {
                  target: 'processing_jobs',
                  actions: ['completeJob']
                },
                onError: {
                  target: 'processing_jobs',
                  actions: [(context, event) => {
                    console.error('❌ Job processing failed:', event.data);
                    // Could implement retry logic here
                  }]
                }
              }
            }
          },

          on: {
            USER_ACTIVITY: {
              target: 'active',
              actions: ['updateActivity']
            }
          }
        }
      },

      on: {
        START_IDLE_DETECTION: '.active',
        STOP_IDLE_DETECTION: 'stopped',
        QUEUE_BACKGROUND_JOB: {
          actions: ['queueBackgroundJob']
        },
        ENABLE_SELF_PROMPTING: {
          actions: assign({ backgroundJobsEnabled: true })
        },
        DISABLE_SELF_PROMPTING: {
          actions: assign({ backgroundJobsEnabled: false })
        },
        NEO4J_CONNECTED: {
          actions: ['updateServiceConnections']
        },
        MINIO_CONNECTED: {
          actions: ['updateServiceConnections']
        },
        RABBITMQ_CONNECTED: {
          actions: ['updateServiceConnections']
        },
        SELF_PROMPT_COMPLETED: {
          actions: ['storeSelfPrompt']
        }
      }
    },

    stopped: {
      entry: [() => console.log('🛑 Idle detection stopped')],
      on: {
        START_IDLE_DETECTION: 'monitoring'
      }
    }
  }
}, {
  services: idleDetectionServices,
  actions: idleDetectionActions,
  guards: idleDetectionGuards
});

// Helper functions for self-prompting
async function generateContextualPrompts(systemContext: any): Promise<SelfPrompt[]> {
  const prompts: SelfPrompt[] = [];
  
  // Generate prompts based on system state
  if (systemContext.sessionDuration > 30 * 60 * 1000) { // 30 minutes
    prompts.push({
      id: crypto.randomUUID(),
      prompt: "Analyze the current legal research session for potential gaps and suggest next steps",
      context: systemContext,
      confidence: 0.8,
      timestamp: Date.now(),
      triggerReason: 'idle_detected',
      processedByNeo4j: false,
      minioArtifacts: []
    });
  }
  
  if (systemContext.completedJobs > 5) {
    prompts.push({
      id: crypto.randomUUID(),
      prompt: "Identify patterns in completed legal document processing tasks to optimize future workflows",
      context: systemContext,
      confidence: 0.9,
      timestamp: Date.now(),
      triggerReason: 'pattern_recognition',
      processedByNeo4j: false,
      minioArtifacts: []
    });
  }
  
  return prompts;
}

function selectBestPrompt(prompts: SelfPrompt[], history: SelfPrompt[]): SelfPrompt | null {
  if (prompts.length === 0) return null;
  
  // Avoid repeating recent prompts
  const recentPrompts = history.slice(0, 10).map(p => p.prompt);
  const uniquePrompts = prompts.filter(p => !recentPrompts.includes(p.prompt));
  
  if (uniquePrompts.length === 0) return prompts[0];
  
  // Select highest confidence prompt
  return uniquePrompts.reduce((best, current) => 
    current.confidence > best.confidence ? current : best
  );
}

async function storePromptInNeo4j(prompt: SelfPrompt, sessionId: string): Promise<void> {
  // Store prompt in Neo4j graph database for relationship analysis
  console.log('📊 Storing self-prompt in Neo4j:', prompt.id);
  
  // In production, this would call your Neo4j API endpoint
  // For now, we'll simulate the storage
  await new Promise(resolve => setTimeout(resolve, 100));
}

async function checkNeo4jConnection(): Promise<boolean> {
  try {
    // Check Neo4j connection
    const response = await fetch('/api/health/neo4j');
    return response.ok;
  } catch {
    return false;
  }
}

async function checkMinioConnection(): Promise<boolean> {
  try {
    // Check MinIO connection
    const response = await fetch('/api/v1/minio/health');
    return response.ok;
  } catch {
    return false;
  }
}

async function checkRabbitMQConnection(): Promise<boolean> {
  try {
    // Check RabbitMQ connection
    const response = await fetch('/api/rabbitmq/health');
    return response.ok;
  } catch {
    return false;
  }
}

// Export actor type for TypeScript
export type IdleDetectionActor = ActorRefFrom<typeof idleDetectionMachine>;

// Helper to create and start the machine
export function createIdleDetectionService(config?: Partial<IdleDetectionContext>) {
  const machine = idleDetectionMachine.withContext({
    ...idleDetectionMachine.context,
    ...config
  });
  
  const service = interpret(machine);
  service.start();
  
  console.log('🚀 Idle detection service started');
  
  return service;
}

export default idleDetectionMachine;