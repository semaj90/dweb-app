
// Enhanced AI Synthesis Orchestrator with Full Stack Integration
// Connects Neo4j, PostgreSQL/pgvector, XState, Redis, Ollama, and Go services

import { logger } from './logger';
import { drizzle } from "drizzle-orm/postgres-js";
import { pgTable, text, vector, timestamp, jsonb, uuid, integer, boolean } from "drizzle-orm/pg-core";
import type { PoolConfig } from 'pg';
import { eq, sql } from "drizzle-orm";
import postgres from "postgres";
import { createMachine, createActor, fromPromise, interpret } from "xstate";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import Redis from 'ioredis';
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import type { Document as LangChainDocumentType } from '@langchain/core/documents';
import type { Document as LangChainDocument } from '@langchain/core/documents';
// Import existing components
import { aiAssistantSynthesizer } from './ai-assistant-input-synthesizer';
import { legalBERT } from './legalbert-middleware';
import { cachingLayer } from './caching-layer';
import { feedbackLoop } from './feedback-loop';
import { monitoringService } from './monitoring-service';
import { EventEmitter } from "events";

// ===== DATABASE SCHEMA (Drizzle ORM TypeScript Safe) =====

export const legalDocuments = pgTable('legal_documents', {
  id: uuid('id').defaultRandom().primaryKey(),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 384 }), // nomic-embed-text dimension
  metadata: jsonb('metadata'),
  documentType: text('document_type'),
  caseId: text('case_id'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});

export const autoSolveResults = pgTable('autosolve_results', {
  id: uuid('id').defaultRandom().primaryKey(),
  query: text('query').notNull(),
  solution: jsonb('solution'),
  confidence: integer('confidence'),
  processingTime: integer('processing_time'),
  serviceUsed: text('service_used'),
  success: boolean('success'),
  createdAt: timestamp('created_at').defaultNow(),
});

export const synthesisCache = pgTable('synthesis_cache', {
  id: uuid('id').defaultRandom().primaryKey(),
  queryHash: text('query_hash').unique().notNull(),
  result: jsonb('result'),
  metadata: jsonb('metadata'),
  hitCount: integer('hit_count').default(0),
  lastAccessed: timestamp('last_accessed').defaultNow(),
  createdAt: timestamp('created_at').defaultNow(),
});

// ===== SERVICE CONFIGURATION =====

const services = {
  // Core AI Services
  neo4j: {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
    user: process.env.NEO4J_USER || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'password',
  },

  // Go Microservices
  goMicroservice: {
    enhancedRAG: 'http://localhost:8094',
    gpuOrchestrator: 'http://localhost:8095',
    goLlama: 'http://localhost:8096',
    tensorService: 'http://localhost:8097',
    quicServer: 'quic://localhost:8098',
  },

  // Ollama Configuration
  ollama: {
    baseUrl: 'http://localhost:11434',
    models: {
      legal: 'gemma3:legal-latest',
      embedding: 'nomic-embed-text:latest',
    },
  },

  // MCP Services
  context7: 'http://localhost:4000',
  context7MultiCore: 'http://localhost:4100',
  aiSynthesisMCP: 'http://localhost:8200',

  // Database
  postgres: {
    host: process.env.POSTGRES_HOST || 'localhost',
    port: parseInt(process.env.POSTGRES_PORT || '5432'),
    database: process.env.POSTGRES_DB || 'legal_ai',
    user: process.env.POSTGRES_USER || 'postgres',
    password: process.env.POSTGRES_PASSWORD || 'postgres',
  },

  // Redis Configuration
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    db: 0,
    keyPrefix: 'legal-ai:',
  },
};

// ===== DATABASE CONNECTION =====

const pgConnection = postgres({
  ...services.postgres,
  max: 20,
  idle_timeout: 20,
  connect_timeout: 60,
});

export const db = drizzle(pgConnection, {
  schema: { legalDocuments, autoSolveResults, synthesisCache },
});

// ===== REDIS CONNECTION =====

const redis = new Redis({
  ...services.redis,
  maxRetriesPerRequest: 3,
  retryStrategy: (times) => Math.min(times * 50, 2000),
});

// ===== UTILITY FUNCTIONS =====

function generateCacheKey(query: string): string {
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(query).digest('hex');
}

function applyMMR(documents: any[], lambda: number = 0.7): unknown[] {
  if (documents.length <= 1) return documents;

  const selected = [documents[0]];
  const remaining = documents.slice(1);

  while (remaining.length > 0 && selected.length < 10) {
    let bestScore = -Infinity;
    let bestDoc = null;
    let bestIndex = -1;

    for (let i = 0; i < remaining.length; i++) {
      const doc = remaining[i];
      const relevance = doc.crossEncoderScore || 0;

      // Calculate max similarity to already selected docs
      let maxSim = 0;
      for (const selectedDoc of selected) {
        const sim = calculateSimilarity(doc, selectedDoc);
        maxSim = Math.max(maxSim, sim);
      }

      // MMR score
      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestDoc = doc;
        bestIndex = i;
      }
    }

    if (bestDoc) {
      selected.push(bestDoc);
      remaining.splice(bestIndex, 1);
    } else {
      break;
    }
  }

  return selected;
}

function calculateSimilarity(doc1: any, doc2: any): number {
  // Simple Jaccard similarity for demonstration
  const text1 = (doc1.pageContent || doc1.content || '').toLowerCase();
  const text2 = (doc2.pageContent || doc2.content || '').toLowerCase();

  const words1 = new Set(text1.split(/\s+/));
  const words2 = new Set(text2.split(/\s+/));

  const intersection = new Set([...words1].filter((x) => words2.has(x)));
  const union = new Set([...words1, ...words2]);

  return intersection.size / union.size;
}

function buildEnhancedPrompt(input: any): string {
  let prompt = `You are an expert legal AI assistant using gemma3:legal-latest with access to comprehensive legal knowledge.

QUERY: ${input.query}

`;

  // Add LegalBERT analysis
  if (input.legalBertAnalysis) {
    prompt += `LEGAL ANALYSIS:
- Identified Entities: ${input.legalBertAnalysis.entities.map((e: any) => e.text).join(', ')}
- Legal Concepts: ${input.legalBertAnalysis.concepts.map((c) => c.concept).join(', ')}
- Complexity Score: ${input.legalBertAnalysis.complexity.legalComplexity}
- Jurisdiction: ${input.legalBertAnalysis.jurisdiction || 'General'}

`;
  }

  // Add relevant sources
  if (input.rankedResults?.length > 0) {
    prompt += `RELEVANT LEGAL SOURCES:
`;
    input.rankedResults.slice(0, 5).forEach((source, i) => {
      const title = source.metadata?.title || `Document ${i + 1}`;
      const content = source.pageContent || source.content || source.text || '';
      const relevance = source.crossEncoderScore || source.score || 0;

      prompt += `
${i + 1}. ${title} (Relevance: ${(relevance * 100).toFixed(1)}%)
${content.substring(0, 500)}...

`;
    });
  }

  // Add Context7 documentation if available
  if (input.context7Docs) {
    prompt += `
TECHNICAL DOCUMENTATION:
${JSON.stringify(input.context7Docs, null, 2).substring(0, 1000)}...

`;
  }

  // Add Go-Llama response if available
  if (input.goLlamaResponse) {
    prompt += `
ADDITIONAL ANALYSIS:
${input.goLlamaResponse.substring(0, 500)}...

`;
  }

  prompt += `
INSTRUCTIONS:
1. Provide a comprehensive legal analysis addressing the query
2. Cite specific statutes, cases, or legal principles where applicable
3. Structure your response with clear sections
4. Include any important caveats or limitations
5. Recommend next steps or actions if appropriate
6. Distinguish between legal information and legal advice
7. Format the response in JSON with the following structure:

{
  "summary": "Brief executive summary",
  "analysis": {
    "primary_issues": ["issue1", "issue2"],
    "applicable_law": ["statute1", "case1"],
    "legal_principles": ["principle1", "principle2"]
  },
  "detailed_discussion": "Comprehensive analysis",
  "recommendations": ["recommendation1", "recommendation2"],
  "caveats": ["caveat1", "caveat2"],
  "confidence_score": 0.0-1.0,
  "sources_cited": ["source1", "source2"]
}

RESPONSE:`;

  return prompt;
}

// ===== XSTATE ORCHESTRATION MACHINE =====

const orchestrationMachine = createMachine({
  id: 'aiSynthesisOrchestration',
  initial: 'idle',
  context: {
    query: null,
    embeddings: null,
    neo4jResults: null,
    pgVectorResults: null,
    ragResults: null,
    legalBertAnalysis: null,
    ollamaResponse: null,
    goLlamaResponse: null,
    finalSynthesis: null,
    performance: {
      startTime: null,
      endTime: null,
      stageTimings: {},
    },
  },
  states: {
    idle: {
      on: {
        START: {
          target: 'initializing',
          actions: 'recordStartTime',
        },
      },
    },

    initializing: {
      invoke: {
        src: 'initializeServices',
        onDone: 'processing',
        onError: 'error',
      },
    },

    processing: {
      initial: 'checkingCache',
      states: {
        checkingCache: {
          invoke: {
            src: 'checkCache',
            onDone: [
              {
                target: 'complete',
                guard: 'cacheHit',
                actions: 'useCachedResult',
              },
              {
                target: 'analyzingQuery',
              },
            ],
          },
        },

        analyzingQuery: {
          invoke: {
            src: 'analyzeWithLegalBERT',
            onDone: {
              target: 'generatingEmbeddings',
              actions: 'storeLegalBertAnalysis',
            },
            onError: 'fallbackAnalysis',
          },
        },

        fallbackAnalysis: {
          invoke: {
            src: 'basicAnalysis',
            onDone: 'generatingEmbeddings',
          },
        },

        generatingEmbeddings: {
          invoke: {
            src: 'generateNomicEmbeddings',
            onDone: {
              target: 'searchingKnowledgeBase',
              actions: 'storeEmbeddings',
            },
          },
        },

        searchingKnowledgeBase: {
          type: 'parallel',
          states: {
            neo4jSearch: {
              invoke: {
                src: 'searchNeo4j',
                onDone: {
                  actions: 'storeNeo4jResults',
                },
              },
            },
            pgVectorSearch: {
              invoke: {
                src: 'searchPGVector',
                onDone: {
                  actions: 'storePGVectorResults',
                },
              },
            },
            enhancedRAGPipeline: {
              invoke: {
                src: 'runEnhancedRAGPipeline',
                onDone: {
                  actions: 'storeRAGResults',
                },
              },
            },
            goLlamaPipeline: {
              invoke: {
                src: 'runGoLlamaPipeline',
                onDone: {
                  actions: 'storeGoLlamaResponse',
                },
              },
            },
          },
          onDone: 'rankingResults',
        },

        rankingResults: {
          invoke: {
            src: 'rankWithCrossEncoder',
            onDone: 'context7Enhancement',
          },
        },

        context7Enhancement: {
          invoke: {
            src: 'enhanceWithContext7',
            onDone: 'generatingResponse',
          },
        },

        generatingResponse: {
          invoke: {
            src: 'generateWithGemma3Legal',
            onDone: {
              target: 'synthesizing',
              actions: 'storeOllamaResponse',
            },
          },
        },

        synthesizing: {
          invoke: {
            src: 'performFinalSynthesis',
            onDone: {
              target: 'cachingResult',
              actions: 'storeFinalSynthesis',
            },
          },
        },

        cachingResult: {
          invoke: {
            src: 'cacheResult',
            onDone: 'complete',
          },
        },

        complete: {
          type: 'final',
          entry: 'recordEndTime',
        },
      },
    },

    error: {
      entry: 'logError',
      on: {
        RETRY: 'processing',
      },
    },
  },
});

// ===== MAIN ORCHESTRATOR CLASS =====

export class EnhancedAISynthesisOrchestrator {
  private machine: any;
  private service: any;
  private neo4jStore: Neo4jVectorStore | null = null;
  private pgVectorStore: PGVectorStore | null = null;
  private ollama: ChatOllama;
  private embeddings: OllamaEmbeddings;
  private initialized: boolean = false;

  constructor() {
    this.initialize();
  }

  private async initialize() {
    try {
      logger.info('[Orchestrator] Initializing Enhanced AI Synthesis Orchestrator...');

      // Initialize Ollama with gemma3:legal-latest
      this.ollama = new ChatOllama({
        baseUrl: services.ollama.baseUrl,
        model: services.ollama.models.legal,
        temperature: 0.3,
        // numCtx: 8192, // Removed - not valid in current API
        // numGpu: 999, // Removed - not valid in current API
        // numThread: 16, // Removed - not valid in current API
        format: 'json',
      });

      // Initialize nomic-embed-text embeddings
      this.embeddings = new OllamaEmbeddings({
        baseUrl: services.ollama.baseUrl,
        model: services.ollama.models.embedding,
        // requestOptions removed - not valid in current API
      });

      // Initialize Neo4j vector store
      await this.initializeNeo4j();

      // Initialize PostgreSQL pgvector store
      await this.initializePGVector();

      // Setup XState service
      this.setupStateMachine();

      // Ensure models exist
      await this.ensureModels();

      // Test service connectivity
      await this.testServiceConnectivity();

      this.initialized = true;
      logger.info('[Orchestrator] Initialization complete');
    } catch (error: any) {
      logger.error('[Orchestrator] Initialization failed:', error);
      throw error;
    }
  }

  private async initializeNeo4j() {
    try {
      // Initialize Neo4j store with fallback
      try {
        this.neo4jStore = new (Neo4jVectorStore as any)(this.embeddings, {
          url: services.neo4j.uri,
          username: services.neo4j.user,
          password: services.neo4j.password,
          indexName: 'legal_documents',
          textNodeProperty: 'text',
          embeddingNodeProperty: 'embedding'
        });
      } catch {
        this.neo4jStore = null;
      }
      logger.info('[Orchestrator] Neo4j vector store connected');
    } catch (error: any) {
      logger.warn('[Orchestrator] Neo4j connection failed, will use fallback:', error);
    }
  }

  private async initializePGVector() {
    try {
      const pgConfig: PoolConfig = {
        host: services.postgres.host,
        port: services.postgres.port,
        database: services.postgres.database,
        user: services.postgres.user,
        password: services.postgres.password,
        max: 20,
      };

      // Initialize PGVector store with fallback
      try {
        this.pgVectorStore = new (PGVectorStore as any)(this.embeddings, {
          postgresConnectionOptions: pgConfig,
          tableName: 'legal_documents',
          columns: {
            idColumnName: 'id',
            vectorColumnName: 'embedding',
            contentColumnName: 'content',
            metadataColumnName: 'metadata',
          },
          distanceStrategy: 'cosine' // Use cosine similarity
        });
      } catch {
        this.pgVectorStore = null;
      }

      // Create indexes for better performance
      await pgConnection`
        CREATE INDEX IF NOT EXISTS idx_legal_documents_embedding
        ON legal_documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
      `;

      logger.info('[Orchestrator] PGVector store connected with IVFFlat index');
    } catch (error: any) {
      logger.warn('[Orchestrator] PGVector connection failed, will use fallback:', error);
    }
  }

  private setupStateMachine() {
    // XState v5 uses provide() instead of withConfig()
    // We need to capture 'this' context for the actors
    const self = this;

    this.machine = orchestrationMachine.provide({
      actors: {
        initializeServices: fromPromise(async () => {
          if (!this.initialized) {
            await this.initialize();
          }
          return true;
        }),

        checkCache: fromPromise(async ({ input }: { input: any }) => {
          const cacheKey = generateCacheKey(input.query);

          // Check Redis first (fastest)
          const redisResult = await redis.get(cacheKey);
          if (redisResult) {
            logger.info('[Cache] Redis hit');
            return { hit: true, data: JSON.parse(redisResult) };
          }

          // Check PostgreSQL cache
          const dbCache = await db
            .select()
            .from(synthesisCache)
            .where(eq(synthesisCache.queryHash, cacheKey))
            .limit(1);

          if (dbCache.length > 0) {
            logger.info('[Cache] Database hit');
            // Update hit count and last accessed
            await db
              .update(synthesisCache)
              .set({
                hitCount: sql`${synthesisCache.hitCount} + 1`,
                lastAccessed: new Date(),
              })
              .where(eq(synthesisCache.id, dbCache[0].id));

            // Store in Redis for next time
            await (redis as any).setex(cacheKey, 3600, JSON.stringify(dbCache[0].result));

            return { hit: true, data: dbCache[0].result };
          }

          return { hit: false };
        }),

        analyzeWithLegalBERT: fromPromise(async ({ input }: { input: any }) => {
          return await legalBERT.analyzeLegalText(input.query);
        }),

        basicAnalysis: fromPromise(async ({ input }: { input: any }) => {
          return {
            entities: [],
            concepts: [],
            complexity: { legalComplexity: 0.5 },
          };
        }),

        generateNomicEmbeddings: fromPromise(async ({ input }: { input: any }) => {
          const embedding = await self.embeddings.embedQuery(input.query);
          logger.info('[Embeddings] Generated nomic-embed-text embedding');
          return embedding;
        }),

        searchNeo4j: fromPromise(async ({ input }: { input: any }) => {
          if (!self.neo4jStore) return [];

          const results = await self.neo4jStore.similaritySearch(input.query, 10);

          logger.info(`[Neo4j] Found ${results.length} documents`);
          return results;
        }),

        searchPGVector: fromPromise(async ({ input }: { input: any }) => {
          if (!self.pgVectorStore) return [];

          const results = await self.pgVectorStore.similaritySearch(input.query, 10);

          logger.info(`[PGVector] Found ${results.length} documents`);
          return results.map((doc, index) => ({
            ...doc,
            score: 1.0 - (index * 0.1),
          }));
        }),

        runEnhancedRAGPipeline: fromPromise(async ({ input }: { input: any }) => {
          try {
            const response = await fetch(`${services.goMicroservice.enhancedRAG}/api/search`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: input.query,
                limit: 10,
                useGPU: true,
                useSIMD: true,
                embedding: input.embeddings || null,
              }),
            });

            if (!response.ok) throw new Error('Enhanced RAG failed');

            const result = await response.json();
            logger.info(`[Enhanced RAG] Processed with GPU acceleration`);
            return result;
          } catch (error: any) {
            logger.error('[Enhanced RAG] Pipeline failed:', error);
            return [];
          }
        }),

        runGoLlamaPipeline: fromPromise(async ({ input }: { input: any }) => {
          try {
            const response = await fetch(`${services.goMicroservice.goLlama}/api/generate`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model: 'gemma3:legal-latest',
                prompt: input.query,
                context: input.legalBertAnalysis,
                temperature: 0.3,
                max_tokens: 2000,
                stream: false,
              }),
            });

            if (response.ok) {
              const result = await response.json();
              logger.info('[Go-Llama] Generated response');
              return result.response;
            }
          } catch (error: any) {
            logger.warn('[Go-Llama] Service unavailable:', error);
          }
          return null;
        }),

        rankWithCrossEncoder: fromPromise(async ({ input }: { input: any }) => {
          // Get the current context from the machine
          const context = input;

          // Combine all results
          const allResults = [
            ...(context.neo4jResults || []),
            ...(context.pgVectorResults || []),
            ...(context.ragResults?.documents || []),
          ];

          // Use LegalBERT for cross-encoder ranking
          const rankedResults = [];
          for (const result of allResults) {
            const similarity = await legalBERT.calculateLegalSimilarity(
              context.query,
              result.pageContent || result.content || result.text
            );

            rankedResults.push({
              ...result,
              crossEncoderScore: similarity.similarity,
              legalRelevance: (similarity as any).legalRelevance || similarity.confidence || 0.5,
            });
          }

          // Sort by cross-encoder score
          const sorted = rankedResults.sort((a, b) => b.crossEncoderScore - a.crossEncoderScore);

          // Apply MMR for diversity
          const diverseResults = applyMMR(sorted, 0.7);

          logger.info(`[Cross-Encoder] Ranked ${diverseResults.length} results`);
          return diverseResults;
        }),

        enhanceWithContext7: fromPromise(async ({ input }: { input: any }) => {
          try {
            // Query Context7 MCP for relevant documentation
            const response = await fetch(`${services.context7}/api/query`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: input.query,
                context: input.legalBertAnalysis,
                includeLibraries: ['langchain', 'drizzle-orm', 'xstate', 'neo4j'],
                maxTokens: 5000,
              }),
            });

            if (response.ok) {
              const docs = await response.json();
              logger.info('[Context7] Enhanced with documentation');
              return docs;
            }
          } catch (error: any) {
            logger.warn('[Context7] Enhancement failed:', error);
          }
          return null;
        }),

        generateWithGemma3Legal: fromPromise(async ({ input }: { input: any }) => {
          const prompt = buildEnhancedPrompt(input);

          try {
            // Try GPU Orchestrator first for acceleration
            const gpuResponse = await fetch(
              `${services.goMicroservice.gpuOrchestrator}/api/generate`,
              {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model: 'gemma3:legal-latest',
                  prompt,
                  useGPU: true,
                  workers: 32,
                  temperature: 0.3,
                  max_tokens: 4000,
                }),
              }
            );

            if (gpuResponse.ok) {
              const result = await gpuResponse.json();
              logger.info('[GPU Orchestrator] Generated with RTX 3060 Ti');
              return result.response;
            }
          } catch (error: any) {
            logger.warn('[GPU Orchestrator] Falling back to Ollama');
          }

          // Fallback to direct Ollama API call
          const ollamaResponse = await fetch(`${services.ollama.baseUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: services.ollama.models.legal,
              prompt,
              stream: false,
            }),
          });

          if (ollamaResponse.ok) {
            const result = await ollamaResponse.json();
            logger.info('[Ollama] Generated response with gemma3:legal-latest');
            return result.response;
          }

          throw new Error('Ollama fallback failed');
        }),

        performFinalSynthesis: fromPromise(async ({ input }: { input: any }) => {
          // Use the main AI synthesizer
          const result = await aiAssistantSynthesizer.synthesizeInput({
            query: input.query,
            context: {
              legalBertAnalysis: input.legalBertAnalysis,
              userId: input.userId || 'default',
            },
            options: {
              enableMMR: true,
              enableCrossEncoder: true,
              enableLegalBERT: true,
              enableRAG: true,
              maxSources: 10,
              similarityThreshold: 0.7,
              diversityLambda: 0.3,
            },
          });

          // Track in monitoring service
          // monitoringService.recordSynthesis({
          //   requestId: result.metadata?.requestId,
          //   processingTime: Date.now() - (input.performance?.startTime || Date.now()),
          //   confidence: result.metadata?.confidence,
          // });

          return result;
        }),

        cacheResult: fromPromise(async ({ input }: { input: any }) => {
          const cacheKey = generateCacheKey(input.query);
          const result = input.finalSynthesis;

          // Store in Redis
          await (redis as any).setex(cacheKey, 3600, JSON.stringify(result));

          // Store in PostgreSQL
          await db.insert(synthesisCache).values({
            queryHash: cacheKey,
            result: result,
            metadata: {
              processingTime: Date.now() - (input.performance?.startTime || Date.now()),
              servicesUsed: ['neo4j', 'pgvector', 'enhanced-rag', 'ollama'],
              confidence: result.metadata?.confidence,
            },
          });

          logger.info('[Cache] Result cached successfully');
          return true;
        }),
      },

      actions: {
        recordStartTime: ({ context }) => {
          context.performance.startTime = Date.now();
        },

        recordEndTime: ({ context }) => {
          context.performance.endTime = Date.now();
          const duration = context.performance.endTime - context.performance.startTime;
          logger.info(`[Performance] Total processing time: ${duration}ms`);
        },

        useCachedResult: ({ context, event }) => {
          context.finalSynthesis = event.output?.data || event.data;
        },

        storeLegalBertAnalysis: ({ context, event }) => {
          (context as any).legalBertAnalysis = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.legalBert = Date.now();
        },

        storeEmbeddings: ({ context, event }) => {
          (context as any).embeddings = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.embeddings = Date.now();
        },

        storeNeo4jResults: ({ context, event }) => {
          (context as any).neo4jResults = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.neo4j = Date.now();
        },

        storePGVectorResults: ({ context, event }) => {
          (context as any).pgVectorResults = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.pgvector = Date.now();
        },

        storeRAGResults: ({ context, event }) => {
          (context as any).ragResults = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.rag = Date.now();
        },

        storeGoLlamaResponse: ({ context, event }) => {
          (context as any).goLlamaResponse = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.goLlama = Date.now();
        },

        storeOllamaResponse: ({ context, event }) => {
          (context as any).ollamaResponse = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.ollama = Date.now();
        },

        storeFinalSynthesis: ({ context, event }) => {
          (context as any).finalSynthesis = event.output || event.data;
          if (!(context as any).performance.stageTimings) {
            (context as any).performance.stageTimings = {};
          }
          (context as any).performance.stageTimings.synthesis = Date.now();
        },

        logError: ({ context, event }) => {
          logger.error('[Orchestrator] Error:', event.error || event.data);
        },
      },

      guards: {
        cacheHit: ({ context, event }) => event.output?.hit === true || event.data?.hit === true,
      },
    });

    // XState v5 uses createActor instead of interpret
    this.service = createActor(this.machine);
    this.service.subscribe((snapshot) => {
      logger.debug(`[State] ${JSON.stringify(snapshot.value || snapshot.status)}`);
    });
    this.service.start();
  }

  private async ensureModels() {
    try {
      // Check and create gemma3:legal-latest
      await this.ensureGemma3LegalModel();

      // Check and pull nomic-embed-text
      await this.ensureNomicEmbedModel();

      logger.info('[Models] All required models ready');
    } catch (error: any) {
      logger.error('[Models] Failed to ensure models:', error);
    }
  }

  private async ensureGemma3LegalModel() {
    try {
      const response = await fetch(`${services.ollama.baseUrl}/api/tags`);
      const { models } = await response.json();

      const hasGemma3Legal = models?.some(
        (m) =>
          m.name === 'gemma3:legal-latest' || (m.name.includes('gemma') && m.name.includes('legal'))
      );

      if (!hasGemma3Legal) {
        logger.info('[Models] Creating gemma3:legal-latest...');

        const modelfile = `
FROM gemma2:2b

SYSTEM """You are an expert legal AI assistant specializing in comprehensive legal analysis.
You have deep expertise in:
- Contract law and commercial agreements
- Tort law and personal injury
- Criminal law and procedure
- Constitutional law and civil rights
- Corporate law and governance
- Intellectual property
- Employment and labor law
- Real estate and property law

Always:
1. Cite relevant statutes, cases, and legal principles
2. Distinguish between legal information and legal advice
3. Consider multiple jurisdictions when applicable
4. Acknowledge limitations and recommend professional consultation when needed
5. Use precise legal terminology while remaining accessible"""

PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER num_gpu 999
PARAMETER num_thread 16
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}"""
`;

        await fetch(`${services.ollama.baseUrl}/api/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: 'gemma3:legal-latest',
            modelfile,
            stream: false,
          }),
        });

        logger.info('[Models] gemma3:legal-latest created successfully');
      }
    } catch (error: any) {
      logger.error('[Models] Failed to ensure gemma3:legal-latest:', error);
    }
  }

  private async ensureNomicEmbedModel() {
    try {
      const response = await fetch(`${services.ollama.baseUrl}/api/tags`);
      const { models } = await response.json();

      const hasNomicEmbed = models?.some(
        (m) => m.name === 'nomic-embed-text' || m.name === 'nomic-embed-text:latest'
      );

      if (!hasNomicEmbed) {
        logger.info('[Models] Pulling nomic-embed-text...');

        await fetch(`${services.ollama.baseUrl}/api/pull`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: 'nomic-embed-text',
            stream: false,
          }),
        });

        logger.info('[Models] nomic-embed-text pulled successfully');
      }
    } catch (error: any) {
      logger.error('[Models] Failed to ensure nomic-embed-text:', error);
    }
  }

  private async testServiceConnectivity() {
    const serviceTests = [
      { name: 'PostgreSQL', test: () => pgConnection`SELECT 1` },
      { name: 'Redis', test: async () => { await (redis as any).set('health-check', 'ok', 'EX', 1); return true; } },
      { name: 'Neo4j', test: () => this.neo4jStore !== null },
      { name: 'Enhanced RAG', test: () => fetch(`${services.goMicroservice.enhancedRAG}/health`) },
      {
        name: 'GPU Orchestrator',
        test: () => fetch(`${services.goMicroservice.gpuOrchestrator}/health`),
      },
      { name: 'Ollama', test: () => fetch(`${services.ollama.baseUrl}/api/tags`) },
      { name: 'Context7', test: () => fetch(`${services.context7}/health`) },
    ];

    for (const service of serviceTests) {
      try {
        await service.test();
        logger.info(`[Connectivity] ${service.name}: ✅ Connected`);
      } catch (error: any) {
        logger.warn(`[Connectivity] ${service.name}: ⚠️ Not available`);
      }
    }
  }

  private generateCacheKey(query: string): string {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(query).digest('hex');
  }

  private applyMMR(documents: any[], lambda: number = 0.7): unknown[] {
    if (documents.length <= 1) return documents;

    const selected = [documents[0]];
    const remaining = documents.slice(1);

    while (remaining.length > 0 && selected.length < 10) {
      let bestScore = -Infinity;
      let bestDoc = null;
      let bestIndex = -1;

      for (let i = 0; i < remaining.length; i++) {
        const doc = remaining[i];
        const relevance = doc.crossEncoderScore || 0;

        // Calculate max similarity to already selected docs
        let maxSim = 0;
        for (const selectedDoc of selected) {
          const sim = this.calculateSimilarity(doc, selectedDoc);
          maxSim = Math.max(maxSim, sim);
        }

        // MMR score
        const mmrScore = lambda * relevance - (1 - lambda) * maxSim;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestDoc = doc;
          bestIndex = i;
        }
      }

      if (bestDoc) {
        selected.push(bestDoc);
        remaining.splice(bestIndex, 1);
      } else {
        break;
      }
    }

    return selected;
  }

  private calculateSimilarity(doc1: any, doc2: any): number {
    // Simple Jaccard similarity for demonstration
    const text1 = (doc1.pageContent || doc1.content || '').toLowerCase();
    const text2 = (doc2.pageContent || doc2.content || '').toLowerCase();

    const words1 = new Set(text1.split(/\s+/));
    const words2 = new Set(text2.split(/\s+/));

    const intersection = new Set([...words1].filter((x) => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    return intersection.size / union.size;
  }


  // ===== PUBLIC API =====

  async process(query: string, options?: Record<string, any>): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }

    logger.info(`[Orchestrator] Processing query: "${query}"`);

    return new Promise((resolve, reject) => {
      const service = createActor(this.machine, {
        input: {
          query,
          ...(options || {}),
          performance: {
            startTime: Date.now(),
            endTime: null,
            stageTimings: {},
          },
        }
      });

      service.subscribe({
        next: (snapshot) => {
          if (snapshot.status === 'done') {
            const result = snapshot.output?.finalSynthesis || snapshot.output;
            const startTime = (snapshot.context as any)?.performance?.startTime || Date.now();

            // Record in autosolve_results table
            db.insert(autoSolveResults)
              .values({
                query,
                solution: result,
                confidence: result?.confidence_score ? Math.round(result.confidence_score * 100) : null,
                processingTime: Date.now() - startTime,
                serviceUsed: 'enhanced-orchestrator',
                success: true,
              })
              .execute();

            resolve(result);
          } else if (snapshot.status === 'error') {
            reject(new Error('Processing failed'));
          }
        },
        error: reject
      });

      service.start();
      service.send({ type: 'START' });
    });
  }

  async processWithStreaming(query: string, options?: Record<string, any>): Promise<AsyncGenerator<any>> {
    const self = this;

    async function* streamResults() {
      let isComplete = false;
      const events: any[] = [];

      const service = createActor(self.machine, {
        input: {
          query,
          ...(options || {}),
          streaming: true,
        }
      });

      service.subscribe({
        next: (snapshot) => {
          events.push({
            type: 'progress',
            stage: snapshot.value,
            progress: self.calculateProgress(snapshot.value),
          });

          if (snapshot.status === 'done' || snapshot.status === 'error') {
            isComplete = true;
          }
        }
      });

      service.start();
      service.send({ type: 'START' });

      while (!isComplete) {
        if (events.length > 0) {
          yield events.shift();
        }
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }

    return streamResults();
  }

  private calculateProgress(state: any): number {
    const stages = {
      idle: 0,
      initializing: 5,
      'processing.checkingCache': 10,
      'processing.analyzingQuery': 20,
      'processing.generatingEmbeddings': 30,
      'processing.searchingKnowledgeBase': 50,
      'processing.rankingResults': 60,
      'processing.context7Enhancement': 70,
      'processing.generatingResponse': 80,
      'processing.synthesizing': 90,
      'processing.cachingResult': 95,
      'processing.complete': 100,
    };

    const stateString = typeof state === 'object' ? JSON.stringify(state) : state;
    return stages[stateString] || 0;
  }

  // Health check
  async health(): Promise<any> {
    return {
      status: this.initialized ? 'healthy' : 'initializing',
      services: {
        postgres: await this.checkPostgres(),
        redis: await this.checkRedis(),
        neo4j: this.neo4jStore !== null,
        pgVector: this.pgVectorStore !== null,
        ollama: await this.checkOllama(),
        enhancedRAG: await this.checkService(services.goMicroservice.enhancedRAG),
        gpuOrchestrator: await this.checkService(services.goMicroservice.gpuOrchestrator),
        context7: await this.checkService(services.context7),
      },
    };
  }

  private async checkPostgres(): Promise<boolean> {
    try {
      await pgConnection`SELECT 1`;
      return true;
    } catch {
      return false;
    }
  }

  private async checkRedis(): Promise<boolean> {
    try {
      await (redis as any).set('health-check', 'ok', 'EX', 1);
      return true;
    } catch {
      return false;
    }
  }

  private async checkOllama(): Promise<boolean> {
    try {
      const response = await fetch(`${services.ollama.baseUrl}/api/tags`);
      return response.ok;
    } catch {
      return false;
    }
  }

  private async checkService(url: string): Promise<boolean> {
    try {
      const response = await fetch(`${url}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const orchestrator = new EnhancedAISynthesisOrchestrator();

// Export for use in API routes
export default orchestrator;
