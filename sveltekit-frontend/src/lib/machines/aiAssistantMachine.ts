/**
 * Enhanced AI Assistant Machine - Full-Stack Legal AI Integration
 *
 * Complete XState 5 state machine with:
 * - PostgreSQL + pgvector database integration
 * - NATS messaging for real-time features
 * - Context7 documentation retrieval
 * - Multi-modal processing (text, documents, images)
 * - 37 Go microservice endpoints with protocol switching
 * - Advanced error handling and recovery
 * - Performance monitoring and caching
 * - Document processing workflows
 * - Legal domain-specific analysis
 */

// xstate imports are declared at the top of the file
import { productionServiceRegistry, getServiceUrl, getOptimalServiceForRoute } from "../services/production-service-registry.js";
import { natsMessaging } from "../services/nats-messaging-service.js";
import { semanticAnalyzer, type SemanticAnalysisResult, type RAGQuery, type RAGResponse } from "../services/enhanced-rag-semantic-analyzer.js";
import type { Document, DocumentChunk, AIInteraction, EmbeddingJob } from "../database/enhanced-schema.js";

// Enhanced AI Assistant context interface
interface AIAssistantContext {
  // Core query state
  currentQuery: string;
  response: string;
  conversationHistory: ConversationEntry[];
  sessionId: string;

  // AI Configuration
  isProcessing: boolean;
  model: string;
  temperature: number;
  maxTokens: number;

  // Database Integration
  databaseConnected: boolean;
  vectorSearchEnabled: boolean;
  currentCaseId?: string;
  currentDocumentId?: string;

  // Context7 Integration
  context7Analysis?: Context7Analysis;
  context7Available: boolean;

  // Multi-modal Processing
  currentDocuments: Document[];
  currentImages: ImageAnalysis[];
  processingQueue: ProcessingJob[];

  // Service Health & Protocol Management
  serviceHealth: ServiceHealthStatus;
  preferredProtocol: 'http' | 'grpc' | 'quic' | 'websocket';
  activeProtocol: 'http' | 'grpc' | 'quic' | 'websocket';

  // Real-time Features
  natsConnected: boolean;
  activeStreaming: boolean;
  streamBuffer: string;
  collaborationUsers: CollaborationUser[];

  // Performance & Monitoring
  performance: PerformanceMetrics;
  cache: CacheStatus;
  error: ExtendedError | null;

  // Legal Domain Features
  legalAnalysis?: LegalAnalysisResult;
  evidenceChain: EvidenceItem[];
  caseContext?: CaseContext;

  // Advanced AI Capabilities
  semanticAnalysis?: SemanticAnalysisResult;
  embeddingJobs: EmbeddingJob[];
  aiInteractions: AIInteraction[];
}

interface ConversationEntry {
  id: string;
  type: 'user' | 'assistant' | 'system' | 'document' | 'image';
  content: string;
  timestamp: Date;
  metadata?: {
    model: string;
    temperature: number;
    responseTime: number;
    tokenCount: number;
    context7Used: boolean;
    protocol: string;
    serviceEndpoint: string;
    documentId?: string;
    imageId?: string;
    semanticScore?: number;
    legalRelevance?: number;
  };
}

interface Context7Analysis {
  suggestions: string[];
  codeExamples: any[];
  documentation: string;
  confidence: number;
  libraries: string[];
  apiEndpoints: string[];
}

interface ImageAnalysis {
  id: string;
  url: string;
  type: 'document' | 'evidence' | 'chart' | 'diagram';
  extractedText?: string;
  ocrConfidence?: number;
  analysis?: {
    entities: string[];
    classification: string;
    relevanceScore: number;
  };
}

interface ProcessingJob {
  id: string;
  type: 'document_analysis' | 'image_ocr' | 'semantic_analysis' | 'embedding_generation' | 'legal_analysis';
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'retrying';
  progress: number;
  input: any;
  output?: any;
  error?: string;
  retryCount: number;
  maxRetries: number;
  createdAt: Date;
  updatedAt: Date;
}

interface ServiceHealthStatus {
  database: { postgres: boolean; qdrant: boolean; neo4j: boolean; redis: boolean };
  ai: { ollama: boolean; enhanced_rag: boolean; context7: boolean };
  microservices: { available: number; total: number; failing: string[] };
  messaging: { nats: boolean; websockets: boolean };
  storage: { minio: boolean; filesystem: boolean };
}

interface CollaborationUser {
  id: string;
  name: string;
  role: string;
  lastActive: Date;
  currentDocument?: string;
  cursor?: { line: number; column: number };
}

interface PerformanceMetrics {
  totalQueries: number;
  totalTokens: number;
  averageResponseTime: number;
  cacheHitRate: number;
  vectorSearchLatency: number;
  databaseLatency: number;
  lastResponseTime: number;
  errorRate: number;
}

interface CacheStatus {
  enabled: boolean;
  hitRate: number;
  size: number;
  maxSize: number;
  ttl: number;
  vectorCacheEnabled: boolean;
}

interface ExtendedError {
  message: string;
  code: string;
  type: 'network' | 'database' | 'ai' | 'processing' | 'validation' | 'auth' | 'permission';
  details?: any;
  recoverable: boolean;
  retryCount: number;
  timestamp: Date;
  context?: string;
}

interface LegalAnalysisResult {
  entities: Array<{ text: string; type: string; confidence: number }>;
  concepts: Array<{ concept: string; relevance: number; category: string }>;
  precedents: Array<{ caseId: string; similarity: number; citation: string }>;
  riskAssessment: {
    level: 'low' | 'medium' | 'high' | 'critical';
    factors: string[];
    score: number;
  };
  recommendations: string[];
}

interface EvidenceItem {
  id: string;
  type: string;
  hash: string;
  timestamp: Date;
  custodyChain: Array<{ actor: string; action: string; timestamp: Date }>;
  verified: boolean;
}

interface CaseContext {
  caseId: string;
  title: string;
  status: string;
  priority: string;
  documents: Document[];
  evidence: EvidenceItem[];
  timeline: Array<{ event: string; timestamp: Date; significance: number }>;
}

// Enhanced AI Assistant events
type AIAssistantEvent =
  | { type: "SEND_MESSAGE"; message: string; useContext7?: boolean; caseId?: string }
  | { type: "UPLOAD_DOCUMENT"; file: File; caseId?: string }
  | { type: "UPLOAD_IMAGE"; file: File; type: 'evidence' | 'document' }
  | { type: "ANALYZE_DOCUMENT"; documentId: string; analysisType: 'semantic' | 'legal' | 'full' }
  | { type: "PERFORM_OCR"; imageId: string }
  | { type: "SEARCH_SEMANTIC"; query: string; filters?: any }
  | { type: "SEARCH_VECTOR"; embedding: number[]; filters?: any }
  | { type: "SEARCH_LEGAL"; query: string; jurisdiction?: string; category?: string }
  | { type: "SET_MODEL"; model: string }
  | { type: "SET_TEMPERATURE"; temperature: number }
  | { type: "SET_PROTOCOL"; protocol: 'http' | 'grpc' | 'quic' | 'websocket' }
  | { type: "SET_CASE_CONTEXT"; caseId: string }
  | { type: "CLEAR_CONVERSATION" }
  | { type: "RETRY_LAST" }
  | { type: "STOP_GENERATION" }
  | { type: "START_STREAMING" }
  | { type: "STREAM_CHUNK"; chunk: string }
  | { type: "STREAM_END" }
  | { type: "CHECK_SERVICE_HEALTH" }
  | { type: "ANALYZE_WITH_CONTEXT7"; topic: string }
  | { type: "ENHANCE_QUERY"; originalQuery: string }
  | { type: "CONNECT_NATS" }
  | { type: "DISCONNECT_NATS" }
  | { type: "COLLABORATION_USER_JOINED"; user: CollaborationUser }
  | { type: "COLLABORATION_USER_LEFT"; userId: string }
  | { type: "DOCUMENT_EDITED"; documentId: string; changes: any }
  | { type: "EVIDENCE_UPDATED"; evidenceId: string; updates: any }
  | { type: "CACHE_CLEAR" }
  | { type: "CACHE_OPTIMIZE" }
  | { type: "PERFORMANCE_RESET" }
  | { type: "ERROR_RECOVER"; errorId: string }
  | { type: "JOB_RETRY"; jobId: string }
  | { type: "SWITCH_TO_FALLBACK_SERVICE"; serviceType: string };

import { createMachine, assign, fromPromise, fromCallback } from "xstate";

function safeNow() {
  try {
    // @ts-ignore
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      // @ts-ignore
      return performance.now();
    }
  } catch (e) {
    // ignore
  }
  return Date.now();
}

export const aiAssistantMachine = createMachine({
  id: "enhancedAiAssistant",
  initial: "initializing",
  context: {
    // Core query state
    currentQuery: "",
    response: "",
    conversationHistory: [],
    sessionId: `session_${Date.now()}_${Math.random().toString(36).substring(2)}`,

    // AI Configuration
    isProcessing: false,
    model: "gemma3-legal",
    temperature: 0.7,
    maxTokens: 2048,

    // Database Integration
    databaseConnected: false,
    vectorSearchEnabled: false,

    // Context7 Integration
    context7Available: false,

    // Multi-modal Processing
    currentDocuments: [],
    currentImages: [],
    processingQueue: [],

    // Service Health & Protocol Management
    serviceHealth: {
      database: { postgres: false, qdrant: false, neo4j: false, redis: false },
      ai: { ollama: false, enhanced_rag: false, context7: false },
      microservices: { available: 0, total: 37, failing: [] },
      messaging: { nats: false, websockets: false },
      storage: { minio: false, filesystem: false }
    },
    preferredProtocol: 'quic',
    activeProtocol: 'http',

    // Real-time Features
    natsConnected: false,
    activeStreaming: false,
    streamBuffer: "",
    collaborationUsers: [],

    // Performance & Monitoring
    performance: {
      totalQueries: 0,
      totalTokens: 0,
      averageResponseTime: 0,
      cacheHitRate: 0,
      vectorSearchLatency: 0,
      databaseLatency: 0,
      lastResponseTime: 0,
      errorRate: 0
    },
    cache: {
      enabled: true,
      hitRate: 0,
      size: 0,
      maxSize: 1000,
      ttl: 3600,
      vectorCacheEnabled: true
    },
    error: null,

    // Legal Domain Features
    evidenceChain: [],

    // Advanced AI Capabilities
    embeddingJobs: [],
    aiInteractions: []
  } as AIAssistantContext,
  types: {} as {
    context: AIAssistantContext;
    events: AIAssistantEvent;
  },
  states: {
    initializing: {
      invoke: {
        id: "initializeServices",
        src: fromPromise(async () => {
          console.log('🚀 Initializing Enhanced AI Assistant...');

          // Check service health
          const healthStatus = await productionServiceRegistry.getClusterHealth();

          // Initialize NATS connection
          let natsConnected = false;
          try {
            natsConnected = await natsMessaging.connect();
          } catch (error) {
            console.warn('NATS connection failed during initialization:', error);
          }

          // Check database connectivity
          let databaseConnected = false;
          try {
            const dbResponse = await fetch('/api/health/database');
            databaseConnected = dbResponse.ok;
          } catch (error) {
            console.warn('Database health check failed:', error);
          }

          // Check Context7 availability
          let context7Available = false;
          try {
            const context7Response = await fetch('http://localhost:40000/health');
            context7Available = context7Response.ok;
          } catch (error) {
            console.warn('Context7 not available:', error);
          }

          // Check vector search capability
          let vectorSearchEnabled = false;
          try {
            const qdrantResponse = await fetch('http://localhost:6333/health');
            vectorSearchEnabled = qdrantResponse.ok;
          } catch (error) {
            console.warn('Qdrant vector search not available:', error);
          }

          return {
            healthStatus,
            natsConnected,
            databaseConnected,
            context7Available,
            vectorSearchEnabled,
            initialization_time: Date.now()
          };
        }),
        onDone: {
          target: "idle",
          actions: assign({
            serviceHealth: ({ event }) => {
              const health = (event as any).output.healthStatus;
              return {
                database: {
                  postgres: (event as any).output.databaseConnected,
                  qdrant: (event as any).output.vectorSearchEnabled,
                  neo4j: false,
                  redis: false
                },
                ai: {
                  ollama: health.services?.['enhanced-rag'] || false,
                  enhanced_rag: health.services?.['enhanced-rag'] || false,
                  context7: (event as any).output.context7Available
                },
                microservices: {
                  available: Object.values(health.services || {}).filter(Boolean).length,
                  total: 37,
                  failing: Object.entries(health.services || {})
                    .filter(([_, healthy]) => !healthy)
                    .map(([name]) => name)
                },
                messaging: {
                  nats: (event as any).output.natsConnected,
                  websockets: false
                },
                storage: {
                  minio: false,
                  filesystem: false
                }
              };
            },
            natsConnected: ({ event }) => (event as any).output.natsConnected,
            databaseConnected: ({ event }) => (event as any).output.databaseConnected,
            context7Available: ({ event }) => (event as any).output.context7Available,
            vectorSearchEnabled: ({ event }) => (event as any).output.vectorSearchEnabled
          })
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Initialization failed: ${(event as any).error}`,
              code: 'INIT_FAILED',
              type: 'processing',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },
    // Lightweight integration runner: links to reports, triggers DB migration action
    integration: {
      invoke: {
        id: 'runIntegrationTasks',
        src: fromPromise(async (context) => {
          // fetch essential docs and provide migration hint
          const docs = {} as any;
          try {
            const [impl, analysis, migration] = await Promise.all([
              fetch('/COMPLETE-IMPLEMENTATION-REPORT.md').then(r => r.text()).catch(() => ''),
              fetch('/COMPLETE-CODEBASE-ANALYSIS.md').then(r => r.text()).catch(() => ''),
              fetch('/setup-postgres-vector-integration.sql').then(r => r.text()).catch(() => '')
            ]);
            docs.implementation = impl;
            docs.analysis = analysis;
            docs.migration = migration;
          } catch (e) {
            // best-effort only
          }

          // return a small summary for machine context
          return {
            docsAvailable: !!(docs.implementation || docs.analysis),
            migrationSqlPresent: !!docs.migration,
            timestamp: Date.now()
          };
        })
      },
    after: {
      0: { target: 'idle' }
      }
    },

    idle: {
      entry: ["clearError", "subscribeToNATSEvents"],
      on: {
        SEND_MESSAGE: {
          target: "processing",
          actions: assign({
            currentQuery: ({ event }) => (event as any).message,
            isProcessing: () => true,
            currentCaseId: ({ event }) => (event as any).caseId
          })
        },
        UPLOAD_DOCUMENT: {
          target: "processingDocument",
          actions: assign({
            isProcessing: () => true
          })
        },
        UPLOAD_IMAGE: {
          target: "processingImage",
          actions: assign({
            isProcessing: () => true
          })
        },
        ANALYZE_DOCUMENT: {
          target: "analyzingDocument"
        },
        PERFORM_OCR: {
          target: "performingOCR"
        },
        SEARCH_SEMANTIC: {
          target: "searchingSemantic"
        },
        SEARCH_VECTOR: {
          target: "searchingVector"
        },
        SEARCH_LEGAL: {
          target: "searchingLegal"
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
        SET_PROTOCOL: {
          actions: assign({
            preferredProtocol: ({ event }) => (event as any).protocol
          })
        },
        SET_CASE_CONTEXT: {
          target: "loadingCaseContext",
          actions: assign({
            currentCaseId: ({ event }) => (event as any).caseId
          })
        },
        CLEAR_CONVERSATION: {
          actions: assign({
            conversationHistory: () => [],
            performance: ({ context }) => ({
              ...context.performance,
              totalQueries: 0,
              totalTokens: 0
            })
          })
        },
        CHECK_SERVICE_HEALTH: "checkingServiceHealth",
        ANALYZE_WITH_CONTEXT7: "analyzingWithContext7",
        CONNECT_NATS: "connectingNATS",
        DISCONNECT_NATS: "disconnectingNATS",
        COLLABORATION_USER_JOINED: {
          actions: assign({
            collaborationUsers: ({ context, event }) => [
              ...context.collaborationUsers,
              (event as any).user
            ]
          })
        },
        COLLABORATION_USER_LEFT: {
          actions: assign({
            collaborationUsers: ({ context, event }) =>
              context.collaborationUsers.filter(user => user.id !== (event as any).userId)
          })
        },
        CACHE_CLEAR: {
          actions: assign({
            cache: ({ context }) => ({
              ...context.cache,
              size: 0,
              hitRate: 0
            })
          })
        },
        PERFORMANCE_RESET: {
          actions: assign({
            performance: () => ({
              totalQueries: 0,
              totalTokens: 0,
              averageResponseTime: 0,
              cacheHitRate: 0,
              vectorSearchLatency: 0,
              databaseLatency: 0,
              lastResponseTime: 0,
              errorRate: 0
            })
          })
        }
      }
    },

    processing: {
      initial: "preparingQuery",
      states: {
        preparingQuery: {
          invoke: {
            id: "enhanceQuery",
            src: fromPromise(async ({ input }: { input: any }) => {
              const { query, useContext7, caseId, context } = input;
              const startTime = performance.now();

              // Add user message to conversation
              const userEntry: ConversationEntry = {
                id: `user_${Date.now()}`,
                type: 'user',
                content: query,
                timestamp: new Date()
              };

              let enhancedQuery = query;
              let context7Analysis: Context7Analysis | undefined;
              let caseContext: CaseContext | undefined;

              // Load case context if provided
              if (caseId && context.databaseConnected) {
                try {
                  const caseResponse = await fetch(`/api/cases/${caseId}`);
                  if (caseResponse.ok) {
                    caseContext = await caseResponse.json();
                    enhancedQuery = `${query}\n\nCase Context: ${caseContext.title}`;
                  }
                } catch (error) {
                  console.warn('Failed to load case context:', error);
                }
              }

              // Enhance query with Context7 if requested and available
              if (useContext7 && context.context7Available) {
                try {
                  const { getSvelte5Docs, getBitsUIv2Docs, getXStateDocs } = await import('../mcp-context72-get-library-docs.js');

                  const [svelteDocsResponse, bitsUIResponse, xstateDocsResponse] = await Promise.all([
                    getSvelte5Docs(query).catch(() => null),
                    getBitsUIv2Docs(query).catch(() => null),
                    getXStateDocs(query).catch(() => null)
                  ]);

                  context7Analysis = {
                    suggestions: [
                      "Consider using Svelte 5 runes for reactive state",
                      "Use bits-ui for accessible component primitives",
                      "Leverage XState for complex workflow management"
                    ],
                    codeExamples: [
                      ...(svelteDocsResponse?.snippets || []),
                      ...(bitsUIResponse?.snippets || []),
                      ...(xstateDocsResponse?.snippets || [])
                    ],
                    documentation: [
                      svelteDocsResponse?.content,
                      bitsUIResponse?.content,
                      xstateDocsResponse?.content
                    ].filter(Boolean).join('\n\n'),
                    confidence: 0.85,
                    libraries: ['svelte', 'bits-ui', 'xstate'].filter(lib =>
                      query.toLowerCase().includes(lib)
                    ),
                    apiEndpoints: []
                  };

                  if (context7Analysis.documentation) {
                    enhancedQuery = `${query}\n\nContext7 Documentation:\n${context7Analysis.documentation.substring(0, 1000)}`;
                  }
                } catch (error) {
                  console.warn('Context7 analysis failed:', error);
                }
              }

              const processingTime = performance.now() - startTime;

              return {
                userEntry,
                enhancedQuery,
                context7Analysis,
                caseContext,
                processingTime
              };
            }),
            input: ({ context, event }) => ({
              query: context.currentQuery,
              useContext7: (event as any).useContext7,
              caseId: context.currentCaseId,
              context
            }),
            onDone: {
              target: "selectingOptimalService",
              actions: assign({
                conversationHistory: ({ context, event }) => [
                  ...context.conversationHistory,
                  (event as any).output.userEntry
                ],
                currentQuery: ({ event }) => (event as any).output.enhancedQuery,
                context7Analysis: ({ event }) => (event as any).output.context7Analysis,
                caseContext: ({ event }) => (event as any).output.caseContext
              })
            },
            onError: {
              target: "#enhancedAiAssistant.error",
              actions: assign({
                error: ({ event }) => ({
                  message: `Query preparation failed: ${(event as any).error}`,
                  code: 'QUERY_PREP_FAILED',
                  type: 'processing',
                  recoverable: true,
                  retryCount: 0,
                  timestamp: new Date()
                })
              })
            }
          }
        },

        selectingOptimalService: {
          invoke: {
            id: "selectOptimalService",
            src: fromPromise(async ({ input }: { input: any }) => {
              const { query, preferredProtocol } = input;

              // Determine optimal service based on query type and system health
              let selectedService = 'enhanced-rag';
              let selectedProtocol = preferredProtocol;

              // Check if legal analysis is needed
              const legalKeywords = ['contract', 'lawsuit', 'evidence', 'case', 'legal', 'court', 'precedent'];
              const isLegalQuery = legalKeywords.some(keyword =>
                query.toLowerCase().includes(keyword)
              );

              if (isLegalQuery) {
                // Try legal-specific services first
                const legalServices = ['enhanced-legal-ai', 'enhanced-legal-ai-clean', 'enhanced-legal-ai-fixed'];
                for (const service of legalServices) {
                  const isHealthy = await productionServiceRegistry.checkServiceHealth(service);
                  if (isHealthy) {
                    selectedService = service;
                    break;
                  }
                }
              }

              // Determine protocol based on query complexity and service load
              const isComplexQuery = query.length > 500 || query.includes('analyze') || query.includes('summary');
              if (isComplexQuery && preferredProtocol === 'quic') {
                // Use QUIC for complex queries if available
                selectedProtocol = 'quic';
              } else if (isComplexQuery) {
                selectedProtocol = 'grpc';
              } else {
                selectedProtocol = 'http';
              }

              // Get service URL
              const serviceUrl = getServiceUrl(selectedService, selectedProtocol);

              return {
                service: selectedService,
                protocol: selectedProtocol,
                url: serviceUrl,
                isLegalQuery
              };
            }),
            input: ({ context }) => ({
              query: context.currentQuery,
              preferredProtocol: context.preferredProtocol
            }),
            onDone: {
              target: "generatingResponse",
              actions: assign({
                activeProtocol: ({ event }) => (event as any).output.protocol
              })
            },
            onError: {
              target: "#enhancedAiAssistant.error",
              actions: assign({
                error: ({ event }) => ({
                  message: `Service selection failed: ${(event as any).error}`,
                  code: 'SERVICE_SELECTION_FAILED',
                  type: 'processing',
                  recoverable: true,
                  retryCount: 0,
                  timestamp: new Date()
                })
              })
            }
          }
        },

        generatingResponse: {
          invoke: {
            id: "generateAIResponse",
            src: fromPromise(async ({ input }: { input: any }) => {
              const { query, model, temperature, maxTokens, conversationHistory, service, protocol, url, caseContext } = input;
              const startTime = Date.now();

              try {
                // Prepare request payload based on protocol
                const requestPayload = {
                  query,
                  model,
                  temperature,
                  maxTokens,
                  conversationHistory: conversationHistory.slice(-10),
                  caseContext: caseContext || undefined,
                  protocol_hint: protocol
                };

                let response;

                // Use appropriate client based on protocol
                switch (protocol) {
                  case 'quic':
                    // QUIC implementation would go here
                    response = await fetch(`${url}/api/rag/query`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(requestPayload)
                    });
                    break;

                  case 'grpc':
                    // gRPC implementation would go here
                    response = await fetch(`${url}/api/rag/query`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(requestPayload)
                    });
                    break;

                  default:
                    // HTTP fallback
                    response = await fetch(`${url}/api/rag/query`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(requestPayload)
                    });
                }

                if (!response.ok) {
                  throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const responseData = await response.json();
                const responseTime = Date.now() - startTime;

                // Create assistant response entry
                const assistantEntry: ConversationEntry = {
                  id: `assistant_${Date.now()}`,
                  type: 'assistant',
                  content: responseData.response || responseData.data?.response || 'No response generated',
                  timestamp: new Date(),
                  metadata: {
                    model,
                    temperature,
                    responseTime,
                    tokenCount: responseData.tokenCount || 0,
                    context7Used: !!input.context7Analysis,
                    protocol,
                    serviceEndpoint: service,
                    semanticScore: responseData.semanticScore || 0,
                    legalRelevance: responseData.legalRelevance || 0
                  }
                };

                // Store interaction in database if connected
                const aiInteraction: Partial<AIInteraction> = {
                  sessionId: input.sessionId,
                  prompt: query,
                  response: assistantEntry.content,
                  model,
                  tokensUsed: responseData.tokenCount || 0,
                  responseTime,
                  confidence: responseData.confidence || 0,
                  metadata: {
                    protocol,
                    service,
                    caseId: caseContext?.caseId
                  }
                };

                return {
                  response: assistantEntry.content,
                  assistantEntry,
                  responseTime,
                  tokenCount: responseData.tokenCount || 0,
                  aiInteraction,
                  semanticAnalysis: responseData.semanticAnalysis,
                  legalAnalysis: responseData.legalAnalysis
                };
              } catch (error) {
                console.error('AI response generation failed:', error);
                throw new Error(`AI generation failed: ${error}`);
              }
            }),
            input: ({ context, event }) => ({
              query: context.currentQuery,
              model: context.model,
              temperature: context.temperature,
              maxTokens: context.maxTokens,
              conversationHistory: context.conversationHistory,
              context7Analysis: context.context7Analysis,
              caseContext: context.caseContext,
              sessionId: context.sessionId,
              ...(event as any).output
            }),
            onDone: {
              target: "#enhancedAiAssistant.idle",
              actions: [
                assign({
                  response: ({ event }) => (event as any).output.response,
                  conversationHistory: ({ context, event }) => [
                    ...context.conversationHistory,
                    (event as any).output.assistantEntry
                  ],
                  performance: ({ context, event }) => ({
                    totalQueries: context.performance.totalQueries + 1,
                    totalTokens: context.performance.totalTokens + (event as any).output.tokenCount,
                    averageResponseTime: (
                      (context.performance.averageResponseTime * context.performance.totalQueries +
                        (event as any).output.responseTime) /
                      (context.performance.totalQueries + 1)
                    ),
                    lastResponseTime: (event as any).output.responseTime,
                    cacheHitRate: context.performance.cacheHitRate,
                    vectorSearchLatency: context.performance.vectorSearchLatency,
                    databaseLatency: context.performance.databaseLatency,
                    errorRate: context.performance.errorRate
                  }),
                  isProcessing: () => false,
                  currentQuery: () => "",
                  context7Analysis: () => undefined,
                  semanticAnalysis: ({ event }) => (event as any).output.semanticAnalysis,
                  legalAnalysis: ({ event }) => (event as any).output.legalAnalysis,
                  aiInteractions: ({ context, event }) => [
                    ...context.aiInteractions,
                    (event as any).output.aiInteraction
                  ]
                }),
                "publishToNATS"
              ]
            },
            onError: {
              target: "#enhancedAiAssistant.error",
              actions: assign({
                error: ({ event }) => ({
                  message: `Response generation failed: ${(event as any).error}`,
                  code: 'RESPONSE_GENERATION_FAILED',
                  type: 'ai',
                  recoverable: true,
                  retryCount: 0,
                  timestamp: new Date()
                }),
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

    processingDocument: {
      invoke: {
        id: "processDocument",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { file, caseId } = input;

          // Upload document to MinIO via upload service
          const formData = new FormData();
          formData.append('file', file);
          if (caseId) formData.append('caseId', caseId);

          const uploadResponse = await fetch('http://localhost:8093/upload', {
            method: 'POST',
            body: formData
          });

          if (!uploadResponse.ok) {
            throw new Error(`Upload failed: ${uploadResponse.status}`);
          }

          const uploadResult = await uploadResponse.json();

          // Trigger semantic analysis
          if (uploadResult.documentId) {
            await semanticAnalyzer.analyzeDocument(
              uploadResult.extractedText || '',
              uploadResult.documentId
            );
          }

          return {
            documentId: uploadResult.documentId,
            filename: file.name,
            fileSize: file.size,
            extractedText: uploadResult.extractedText,
            analysisId: uploadResult.analysisId
          };
        }),
        input: ({ event }) => ({
          file: (event as any).file,
          caseId: (event as any).caseId
        }),
        onDone: {
          target: "idle",
          actions: [
            assign({
              currentDocuments: ({ context, event }) => [
                ...context.currentDocuments,
                {
                  id: (event as any).output.documentId,
                  title: (event as any).output.filename,
                  filename: (event as any).output.filename,
                  fileSize: (event as any).output.fileSize,
                  extractedText: (event as any).output.extractedText,
                  isIndexed: false
                } as Document
              ],
              isProcessing: () => false
            }),
            "publishToNATS"
          ]
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Document processing failed: ${(event as any).error}`,
              code: 'DOCUMENT_PROCESSING_FAILED',
              type: 'processing',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            }),
            isProcessing: () => false
          })
        }
      }
    },

    processingImage: {
      invoke: {
        id: "processImage",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { file, type } = input;

          // Upload image to MinIO
          const formData = new FormData();
          formData.append('file', file);
          formData.append('type', type);

          const uploadResponse = await fetch('http://localhost:8093/upload/image', {
            method: 'POST',
            body: formData
          });

          if (!uploadResponse.ok) {
            throw new Error(`Image upload failed: ${uploadResponse.status}`);
          }

          const uploadResult = await uploadResponse.json();

          // Perform OCR if needed
          let extractedText = '';
          let ocrConfidence = 0;

          if (uploadResult.imageId) {
            try {
              const ocrResponse = await fetch(`http://localhost:8095/ocr/${uploadResult.imageId}`, {
                method: 'POST'
              });

              if (ocrResponse.ok) {
                const ocrResult = await ocrResponse.json();
                extractedText = ocrResult.text || '';
                ocrConfidence = ocrResult.confidence || 0;
              }
            } catch (error) {
              console.warn('OCR processing failed:', error);
            }
          }

          return {
            imageId: uploadResult.imageId,
            filename: file.name,
            fileSize: file.size,
            type,
            extractedText,
            ocrConfidence,
            url: uploadResult.url
          };
        }),
        input: ({ event }) => ({
          file: (event as any).file,
          type: (event as any).type
        }),
        onDone: {
          target: "idle",
          actions: [
            assign({
              currentImages: ({ context, event }) => [
                ...context.currentImages,
                {
                  id: (event as any).output.imageId,
                  url: (event as any).output.url,
                  type: (event as any).output.type,
                  extractedText: (event as any).output.extractedText,
                  ocrConfidence: (event as any).output.ocrConfidence
                }
              ],
              isProcessing: () => false
            }),
            "publishToNATS"
          ]
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Image processing failed: ${(event as any).error}`,
              code: 'IMAGE_PROCESSING_FAILED',
              type: 'processing',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            }),
            isProcessing: () => false
          })
        }
      }
    },

    analyzingDocument: {
      invoke: {
        id: "analyzeDocument",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { documentId, analysisType } = input;

          // Get document content
          const documentResponse = await fetch(`/api/documents/${documentId}`);
          if (!documentResponse.ok) {
            throw new Error(`Document not found: ${documentId}`);
          }

          const document = await documentResponse.json();
          const content = document.extractedText || document.content || '';

          let analysisResult: any = {};

          switch (analysisType) {
            case 'semantic':
              analysisResult = await semanticAnalyzer.analyzeDocument(content, documentId);
              break;

            case 'legal':
              // Legal analysis via specialized service
              const legalResponse = await fetch('http://localhost:8202/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  documentId,
                  content,
                  analysisType: 'legal'
                })
              });

              if (legalResponse.ok) {
                analysisResult = await legalResponse.json();
              }
              break;

            case 'full':
              // Comprehensive analysis
              const [semantic, legal] = await Promise.allSettled([
                semanticAnalyzer.analyzeDocument(content, documentId),
                fetch('http://localhost:8202/api/analyze', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ documentId, content, analysisType: 'legal' })
                }).then(r => r.ok ? r.json() : null)
              ]);

              analysisResult = {
                semantic: semantic.status === 'fulfilled' ? semantic.value : null,
                legal: legal.status === 'fulfilled' ? legal.value : null
              };
              break;
          }

          return {
            documentId,
            analysisType,
            result: analysisResult,
            timestamp: new Date()
          };
        }),
        input: ({ event }) => ({
          documentId: (event as any).documentId,
          analysisType: (event as any).analysisType
        }),
        onDone: {
          target: "idle",
          actions: [
            assign({
              semanticAnalysis: ({ event }) =>
                (event as any).output.analysisType === 'semantic' || (event as any).output.analysisType === 'full'
                  ? (event as any).output.result
                  : undefined,
              legalAnalysis: ({ event }) =>
                (event as any).output.analysisType === 'legal' || (event as any).output.analysisType === 'full'
                  ? (event as any).output.result
                  : undefined
            }),
            "publishToNATS"
          ]
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Document analysis failed: ${(event as any).error}`,
              code: 'DOCUMENT_ANALYSIS_FAILED',
              type: 'ai',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    searchingSemantic: {
      invoke: {
        id: "semanticSearch",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { query, filters } = input;

          const ragQuery: RAGQuery = {
            query,
            filters,
            semantic: {
              useEmbeddings: true,
              expandConcepts: true,
              includeRelated: true
            }
          };

          const result = await semanticAnalyzer.enhancedQuery(ragQuery);
          return result;
        }),
        input: ({ event }) => ({
          query: (event as any).query,
          filters: (event as any).filters
        }),
        onDone: {
          target: "idle",
          actions: assign({
            response: ({ event }) => {
              const result = (event as any).output as RAGResponse;
              return `Found ${result.totalFound} relevant documents:\n\n${result.results.map((r, i) => `${i + 1}. ${r.title} (${(r.relevanceScore * 100).toFixed(1)}% relevant)\n${r.excerpt}\n`).join('\n')
                }`;
            }
          })
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Semantic search failed: ${(event as any).error}`,
              code: 'SEMANTIC_SEARCH_FAILED',
              type: 'ai',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    searchingVector: {
      invoke: {
        id: "vectorSearch",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { embedding, filters } = input;

          // Direct vector search in Qdrant
          const searchPayload = {
            vector: embedding,
            limit: 20,
            with_payload: true,
            score_threshold: filters?.confidenceThreshold || 0.7
          };

          const response = await fetch('http://localhost:6333/collections/legal_documents/points/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchPayload)
          });

          if (!response.ok) {
            throw new Error(`Vector search failed: ${response.status}`);
          }

          const result = await response.json();
          return result.result || [];
        }),
        input: ({ event }) => ({
          embedding: (event as any).embedding,
          filters: (event as any).filters
        }),
        onDone: {
          target: "idle",
          actions: assign({
            response: ({ event }) => {
              const results = (event as any).output;
              return `Vector search found ${results.length} similar documents:\n\n${results.map((r: any, i: number) =>
                `${i + 1}. Score: ${(r.score * 100).toFixed(1)}%\n${r.payload?.content || 'No content'}\n`
              ).join('\n')
                }`;
            }
          })
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Vector search failed: ${(event as any).error}`,
              code: 'VECTOR_SEARCH_FAILED',
              type: 'database',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    searchingLegal: {
      invoke: {
        id: "legalSearch",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { query, jurisdiction, category } = input;

          // Legal-specific search via specialized service
          const searchPayload = {
            query,
            jurisdiction: jurisdiction || 'federal',
            category: category || 'general',
            includePrecedents: true,
            includeStatutes: true
          };

          const response = await fetch('http://localhost:8202/api/search/legal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchPayload)
          });

          if (!response.ok) {
            throw new Error(`Legal search failed: ${response.status}`);
          }

          const result = await response.json();
          return result;
        }),
        input: ({ event }) => ({
          query: (event as any).query,
          jurisdiction: (event as any).jurisdiction,
          category: (event as any).category
        }),
        onDone: {
          target: "idle",
          actions: assign({
            response: ({ event }) => {
              const result = (event as any).output;
              return `Legal search results:\n\n${result.precedents?.map((p: any, i: number) =>
                `${i + 1}. ${p.citation}\n${p.summary}\nRelevance: ${(p.relevance * 100).toFixed(1)}%\n`
              ).join('\n') || 'No precedents found'
                }`;
            },
            legalAnalysis: ({ event }) => (event as any).output
          })
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Legal search failed: ${(event as any).error}`,
              code: 'LEGAL_SEARCH_FAILED',
              type: 'ai',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    loadingCaseContext: {
      invoke: {
        id: "loadCaseContext",
        src: fromPromise(async ({ input }: { input: any }) => {
          const { caseId } = input;

          // Load comprehensive case data
          const [caseResponse, documentsResponse, evidenceResponse] = await Promise.allSettled([
            fetch(`/api/cases/${caseId}`),
            fetch(`/api/cases/${caseId}/documents`),
            fetch(`/api/cases/${caseId}/evidence`)
          ]);

          const caseData = caseResponse.status === 'fulfilled' && caseResponse.value.ok
            ? await caseResponse.value.json() : null;
          const documents = documentsResponse.status === 'fulfilled' && documentsResponse.value.ok
            ? await documentsResponse.value.json() : [];
          const evidence = evidenceResponse.status === 'fulfilled' && evidenceResponse.value.ok
            ? await evidenceResponse.value.json() : [];

          if (!caseData) {
            throw new Error(`Case not found: ${caseId}`);
          }

          // Build timeline from documents and evidence
          const timeline = [
            ...documents.map((d: any) => ({
              event: `Document uploaded: ${d.title}`,
              timestamp: new Date(d.createdAt),
              significance: 3
            })),
            ...evidence.map((e: any) => ({
              event: `Evidence added: ${e.title}`,
              timestamp: new Date(e.createdAt),
              significance: 4
            }))
          ].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

          return {
            caseId,
            title: caseData.title,
            status: caseData.status,
            priority: caseData.priority,
            documents,
            evidence,
            timeline
          };
        }),
        input: ({ context }) => ({
          caseId: context.currentCaseId
        }),
        onDone: {
          target: "idle",
          actions: assign({
            caseContext: ({ event }) => (event as any).output,
            currentDocuments: ({ event }) => (event as any).output.documents,
            evidenceChain: ({ event }) => (event as any).output.evidence
          })
        },
        onError: {
          target: "error",
          actions: assign({
            error: ({ event }) => ({
              message: `Case context loading failed: ${(event as any).error}`,
              code: 'CASE_CONTEXT_FAILED',
              type: 'database',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    checkingServiceHealth: {
      invoke: {
        id: "checkServiceHealth",
        src: fromPromise(async () => {
          const healthStatus = await productionServiceRegistry.getClusterHealth();

          // Check individual service categories
          const databaseHealth = await Promise.allSettled([
            fetch('/api/health/postgres').then(r => r.ok),
            fetch('http://localhost:6333/health').then(r => r.ok),
            fetch('http://localhost:7474/').then(r => r.ok),
            fetch('http://localhost:6379/ping').then(r => r.ok)
          ]);

          const aiHealth = await Promise.allSettled([
            fetch('http://localhost:11434/api/tags').then(r => r.ok),
            fetch('http://localhost:8094/health').then(r => r.ok),
            fetch('http://localhost:40000/health').then(r => r.ok)
          ]);

          return {
            overall: healthStatus.overall,
            database: {
              postgres: databaseHealth[0].status === 'fulfilled' ? databaseHealth[0].value : false,
              qdrant: databaseHealth[1].status === 'fulfilled' ? databaseHealth[1].value : false,
              neo4j: databaseHealth[2].status === 'fulfilled' ? databaseHealth[2].value : false,
              redis: databaseHealth[3].status === 'fulfilled' ? databaseHealth[3].value : false
            },
            ai: {
              ollama: aiHealth[0].status === 'fulfilled' ? aiHealth[0].value : false,
              enhanced_rag: aiHealth[1].status === 'fulfilled' ? aiHealth[1].value : false,
              context7: aiHealth[2].status === 'fulfilled' ? aiHealth[2].value : false
            },
            microservices: {
              available: Object.values(healthStatus.services).filter(Boolean).length,
              total: Object.keys(healthStatus.services).length,
              failing: Object.entries(healthStatus.services)
                .filter(([_, healthy]) => !healthy)
                .map(([name]) => name)
            },
            messaging: {
              nats: natsMessaging.isConnected(),
              websockets: false // TODO: implement WebSocket health check
            },
            storage: {
              minio: false, // TODO: implement MinIO health check
              filesystem: true
            }
          };
        }),
        onDone: {
          target: "idle",
          actions: assign({
            serviceHealth: ({ event }) => (event as any).output
          })
        },
        onError: {
          target: "idle",
          actions: assign({
            serviceHealth: ({ context }) => ({
              ...context.serviceHealth,
            // Keep existing state on error
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
            const { getSvelte5Docs, getBitsUIv2Docs, getXStateDocs } = await import('../mcp-context72-get-library-docs.js');

            // Get relevant documentation for multiple libraries
            const [svelteDocsResponse, bitsUIResponse, xstateDocsResponse] = await Promise.allSettled([
              getSvelte5Docs(topic),
              getBitsUIv2Docs(topic),
              getXStateDocs(topic)
            ]);

            const validResponses = [svelteDocsResponse, bitsUIResponse, xstateDocsResponse]
              .filter(result => result.status === 'fulfilled')
              .map(result => (result as any).value);

            const analysis: Context7Analysis = {
              suggestions: [
                `Modern Svelte 5 approaches for ${topic}`,
                `Accessible component patterns with bits-ui for ${topic}`,
                `State management patterns with XState for ${topic}`,
                `Performance optimization techniques for ${topic}`
              ],
              codeExamples: validResponses.flatMap(response => response.snippets || []),
              documentation: validResponses.map(response => response.content).join('\n\n'),
              confidence: validResponses.length > 0 ? 0.85 : 0.3,
              libraries: ['svelte', 'bits-ui', 'xstate'].filter(lib =>
                topic.toLowerCase().includes(lib) || validResponses.some(r => r.content?.toLowerCase().includes(lib))
              ),
              apiEndpoints: validResponses.flatMap(response => response.apiEndpoints || [])
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
            error: ({ event }) => ({
              message: `Context7 analysis failed: ${(event as any).error}`,
              code: 'CONTEXT7_ANALYSIS_FAILED',
              type: 'ai',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    connectingNATS: {
      invoke: {
        id: "connectNATS",
        src: fromPromise(async () => {
          const connected = await natsMessaging.connect();
          if (!connected) {
            throw new Error('Failed to connect to NATS server');
          }
          return { connected: true };
        }),
        onDone: {
          target: "idle",
          actions: [
            assign({
              natsConnected: () => true,
              serviceHealth: ({ context }) => ({
                ...context.serviceHealth,
                messaging: {
                  ...context.serviceHealth.messaging,
                  nats: true
                }
              })
            }),
            "subscribeToNATSEvents"
          ]
        },
        onError: {
          target: "idle",
          actions: assign({
            natsConnected: () => false,
            error: ({ event }) => ({
              message: `NATS connection failed: ${(event as any).error}`,
              code: 'NATS_CONNECTION_FAILED',
              type: 'network',
              recoverable: true,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    disconnectingNATS: {
      invoke: {
        id: "disconnectNATS",
        src: fromPromise(async () => {
          await natsMessaging.disconnect();
          return { disconnected: true };
        }),
        onDone: {
          target: "idle",
          actions: assign({
            natsConnected: () => false,
            collaborationUsers: () => [],
            serviceHealth: ({ context }) => ({
              ...context.serviceHealth,
              messaging: {
                ...context.serviceHealth.messaging,
                nats: false
              }
            })
          })
        },
        onError: {
          target: "idle",
          actions: assign({
            error: ({ event }) => ({
              message: `NATS disconnection failed: ${(event as any).error}`,
              code: 'NATS_DISCONNECTION_FAILED',
              type: 'network',
              recoverable: false,
              retryCount: 0,
              timestamp: new Date()
            })
          })
        }
      }
    },

    streaming: {
      invoke: {
        id: "streamResponse",
        src: fromCallback(({ input, sendBack }: { input: any; sendBack: any }) => {
          const { query, model, temperature, service } = input;

          // WebSocket streaming implementation
          const serviceUrl = getServiceUrl(service || 'enhanced-rag', 'websocket');
          const ws = new WebSocket(`${serviceUrl}/ws/stream`);

          ws.onopen = () => {
            ws.send(JSON.stringify({
              query,
              model,
              temperature,
              stream: true,
              session_id: input.sessionId
            }));
          };

          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (data.chunk) {
                sendBack({ type: 'STREAM_CHUNK', chunk: data.chunk });
              } else if (data.done) {
                sendBack({ type: 'STREAM_END' });
              } else if (data.error) {
                sendBack({ type: 'error', error: data.error });
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
          temperature: context.temperature,
          sessionId: context.sessionId
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
          actions: [
            assign({
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
                    tokenCount: context.streamBuffer.length / 4, // rough estimate
                    context7Used: false,
                    protocol: 'websocket',
                    serviceEndpoint: 'enhanced-rag'
                  }
                }
              ],
              streamBuffer: () => "",
              activeStreaming: () => false,
              isProcessing: () => false
            }),
            "publishToNATS"
          ]
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
            error: ({ context }) => context.error ? {
              ...context.error,
              retryCount: context.error.retryCount + 1
            } : null
          })
        },
        ERROR_RECOVER: {
          target: "idle",
          actions: assign({
            error: () => null,
            isProcessing: () => false
          })
        },
        CLEAR_CONVERSATION: {
          target: "idle",
          actions: assign({
            error: () => null,
            conversationHistory: () => [],
            isProcessing: () => false,
            streamBuffer: () => ""
          })
        }
      }
    }
  }
});

// Enhanced action implementations
export const aiAssistantActions = {
  clearError: assign({
    error: () => null
  }),

  logError: ({ context }: { context: AIAssistantContext }) => {
    if (context.error) {
      console.error('Enhanced AI Assistant Error:', {
        message: context.error.message,
        code: context.error.code,
        type: context.error.type,
        timestamp: context.error.timestamp,
        context: context.error.context,
        recoverable: context.error.recoverable,
        retryCount: context.error.retryCount
      });

      // Send error to monitoring service
      if (context.natsConnected) {
        natsMessaging.publishSystemHealth({
          type: 'error',
          error: context.error,
          sessionId: context.sessionId
        }).catch(err => console.warn('Failed to publish error to NATS:', err));
      }
    }
  },

  subscribeToNATSEvents: ({ context }: { context: AIAssistantContext }) => {
    if (context.natsConnected) {
      // Subscribe to relevant NATS subjects for real-time updates
      natsMessaging.subscribeToSystemEvents((message) => {
        console.log('Received system event:', message);
        // Handle system events (could send events to machine)
      });

      if (context.currentCaseId) {
        natsMessaging.subscribeToCase(context.currentCaseId, (message) => {
          console.log('Received case event:', message);
          // Handle case-specific events
        });
      }

      // Subscribe to AI analysis completion events
      natsMessaging.subscribeToAIAnalysis((message) => {
        console.log('Received AI analysis event:', message);
      // Handle AI analysis completion
      });
    }
  },

  publishToNATS: ({ context }: { context: AIAssistantContext }) => {
    if (context.natsConnected && context.response) {
      // Publish AI response completion
      natsMessaging.notifyAIAnalysisCompleted(
        `response_${Date.now()}`,
        {
          sessionId: context.sessionId,
          response: context.response,
          caseId: context.currentCaseId,
          timestamp: new Date().toISOString()
        }
      ).catch(err => console.warn('Failed to publish to NATS:', err));
    }
  }
};

// Helper function for service URL resolution
function getServiceUrl(serviceName: string, protocol: 'http' | 'grpc' | 'quic' | 'websocket' = 'http'): string {
  const service = productionServiceRegistry.getServiceByName(serviceName);
  if (!service) {
    console.warn(`Service not found: ${serviceName}, using fallback`);
    return 'http://localhost:8094'; // Enhanced RAG fallback
  }

  const protocolMap = {
    http: 'http',
    grpc: 'grpc',
    quic: 'quic',
    websocket: 'ws'
  };

  return `${protocolMap[protocol]}://localhost:${service.port}`;
}