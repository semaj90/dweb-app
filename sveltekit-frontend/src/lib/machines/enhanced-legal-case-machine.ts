/**
 * Enhanced Legal Case Management Machine with Full Postgres/Drizzle Integration
 * Includes LOAD_CASE, ADD_EVIDENCE, START_AI_ANALYSIS services with DB operations
 * Compatible with your Gemma3 + PostgreSQL + pgvector setup
 */

import { createMachine, assign, fromPromise } from 'xstate';
import { db } from '../server/db/index';
import { cases, evidence, legal_documents, documentChunks, users } from '../server/db/schema-postgres';
import { sql, eq, and, desc } from 'drizzle-orm';

// Temporary type definition to fix import issues
export type CaseForm = {
  caseNumber: string;
  title: string;
  description?: string;
  priority: 'low' | 'medium' | 'high';
  status?: 'draft' | 'active' | 'pending' | 'closed';
  assignedTo?: string;
  dueDate?: string;
  tags?: string[];
  isConfidential?: boolean;
  notifyAssignee?: boolean;
};

// Enhanced context with full database integration
export interface EnhancedLegalCaseContext {
  // Case Management
  currentCase: {
    id?: string;
    title?: string;
    description?: string;
    status?: 'draft' | 'active' | 'pending' | 'closed';
    priority?: 'low' | 'medium' | 'high';
    assigned_attorney?: string;
    created_at?: Date;
    updated_at?: Date;
  } | null;

  // Evidence Management
  evidenceList: Array<{
    id: string;
    case_id: string;
    title: string;
    description?: string;
    evidence_type: string;
    created_at: Date;
    embedding_status?: 'pending' | 'processing' | 'completed' | 'failed';
  }>;

  // AI Analysis
  aiAnalysis: {
    status: 'idle' | 'processing' | 'completed' | 'failed';
    results?: {
      summary?: string;
      keyFindings?: string[];
      recommendations?: string[];
      confidence?: number;
      similarity_cases?: Array<{
        id: string;
        title: string;
        similarity_score: number;
      }>;
    };
    processingStep?: 'embedding' | 'analysis' | 'similarity_search' | 'report_generation';
  };

  // Form state
  formData: Partial<CaseForm>;
  validationErrors: Record<string, string[]>;

  // System state
  loading: boolean;
  error: string | null;
  retryCount: number;
  lastSyncTime?: Date;

  // Database connection status
  dbStatus: 'connected' | 'disconnected' | 'error';

  // Background tasks
  backgroundTasks: Array<{
    id: string;
    type: 'vector_embedding' | 'ai_analysis' | 'similarity_search';
    status: 'queued' | 'processing' | 'completed' | 'failed';
    progress?: number;
  }>;
}

export type EnhancedLegalCaseEvent =
  // Case operations
  | { type: 'LOAD_CASE'; caseId: string; includeEvidence?: boolean }
  | { type: 'CREATE_CASE'; data: CaseForm }
  | { type: 'UPDATE_CASE'; caseId: string; data: Partial<CaseForm> }
  | { type: 'DELETE_CASE'; caseId: string }

  // Evidence operations
  | { type: 'ADD_EVIDENCE'; caseId: string; evidence: {
      title: string;
      description?: string;
      evidence_type: string;
      file_content?: string;
      metadata?: Record<string, any>;
    }}
  | { type: 'REMOVE_EVIDENCE'; evidenceId: string }
  | { type: 'UPDATE_EVIDENCE'; evidenceId: string; data: any }

  // AI Analysis operations
  | { type: 'START_AI_ANALYSIS'; caseId: string; analysisType: 'summary' | 'recommendation' | 'similarity' | 'full' }
  | { type: 'GENERATE_EMBEDDINGS'; documentId: string }
  | { type: 'FIND_SIMILAR_CASES'; caseId: string; threshold?: number }

  // Form operations
  | { type: 'UPDATE_FORM'; data: Partial<CaseForm> }
  | { type: 'VALIDATE_FORM' }
  | { type: 'RESET_FORM' }

  // System operations
  | { type: 'RETRY' }
  | { type: 'CANCEL' }
  | { type: 'SYNC_DB' }
  | { type: 'REFRESH' };

export const enhancedLegalCaseMachine = createMachine({
  id: 'enhancedLegalCase',
  initial: 'initializing',
  types: {
    context: {} as EnhancedLegalCaseContext,
    events: {} as EnhancedLegalCaseEvent
  },
  context: {
    currentCase: null,
    evidenceList: [],
    aiAnalysis: {
      status: 'idle'
    },
    formData: {},
    validationErrors: {},
    loading: false,
    error: null,
    retryCount: 0,
    dbStatus: 'disconnected',
    backgroundTasks: []
  },
  states: {
    initializing: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'initializeSystem',
        src: 'initializeSystem',
        onDone: {
          target: 'idle',
          actions: assign({
            loading: false,
            dbStatus: 'connected',
            lastSyncTime: () => new Date()
          })
        },
        onError: {
          target: 'systemError',
          actions: assign({
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'System initialization failed',
            dbStatus: 'error'
          })
        }
      }
    },

    idle: {
      entry: assign({ loading: false, error: null }),
      on: {
        LOAD_CASE: 'loadingCase',
        CREATE_CASE: 'creatingCase',
        START_AI_ANALYSIS: 'startingAnalysis',
        UPDATE_FORM: {
          actions: assign({
            formData: ({ context, event }) => ({
              ...context.formData,
              ...(event as any).data
            })
          })
        },
        SYNC_DB: 'syncing'
      }
    },

    loadingCase: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'loadCase',
        src: 'loadCase',
        input: ({ event }) => ({
          caseId: (event as any).caseId,
          includeEvidence: (event as any).includeEvidence || false
        }),
        onDone: {
          target: 'caseLoaded',
          actions: assign({
            currentCase: ({ event }) => (event as any).output.case,
            evidenceList: ({ event }) => (event as any).output.evidence || [],
            loading: false,
            error: null,
            lastSyncTime: () => new Date()
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'Failed to load case'
          })
        }
      }
    },

    caseLoaded: {
      on: {
        ADD_EVIDENCE: 'addingEvidence',
        UPDATE_CASE: 'updatingCase',
        START_AI_ANALYSIS: 'startingAnalysis',
        GENERATE_EMBEDDINGS: 'generatingEmbeddings',
        FIND_SIMILAR_CASES: 'findingSimilarCases',
        LOAD_CASE: 'loadingCase',
        REFRESH: 'loadingCase'
      }
    },

    creatingCase: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'createCase',
        src: 'createCase',
        input: ({ event }) => (event as any).data,
        onDone: {
          target: 'caseLoaded',
          actions: assign({
            currentCase: ({ event }) => (event as any).output,
            loading: false,
            error: null,
            formData: {},
            lastSyncTime: () => new Date()
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'Failed to create case'
          })
        }
      }
    },

    addingEvidence: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'addEvidence',
        src: 'addEvidence',
        input: ({ event }) => ({
          caseId: (event as any).caseId,
          evidence: (event as any).evidence
        }),
        onDone: {
          target: 'caseLoaded',
          actions: [
            assign({
              evidenceList: ({ context, event }) => [
                ...context.evidenceList,
                (event as any).output
              ],
              loading: false,
              error: null
            }),
            // Auto-trigger embedding generation for new evidence
            ({ self, event }) => {
              if ((event as any).output?.id) {
                self.send({
                  type: 'GENERATE_EMBEDDINGS',
                  documentId: (event as any).output.id
                });
              }
            }
          ]
        },
        onError: {
          target: 'caseLoaded',
          actions: assign({
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'Failed to add evidence'
          })
        }
      }
    },

    startingAnalysis: {
      entry: assign({
        loading: true,
        aiAnalysis: ({ context }) => ({
          ...context.aiAnalysis,
          status: 'processing',
          processingStep: 'embedding'
        })
      }),
      invoke: {
        id: 'startAIAnalysis',
        src: 'startAIAnalysis',
        input: ({ event }) => ({
          caseId: (event as any).caseId,
          analysisType: (event as any).analysisType
        }),
        onDone: {
          target: 'caseLoaded',
          actions: assign({
            aiAnalysis: ({ context, event }) => ({
              ...context.aiAnalysis,
              status: 'completed',
              results: (event as any).output,
              processingStep: undefined
            }),
            loading: false,
            error: null
          })
        },
        onError: {
          target: 'caseLoaded',
          actions: assign({
            aiAnalysis: ({ context }) => ({
              ...context.aiAnalysis,
              status: 'failed'
            }),
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'AI analysis failed'
          })
        }
      }
    },

    generatingEmbeddings: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'generateEmbeddings',
        src: 'generateEmbeddings',
        input: ({ event }) => ({ documentId: (event as any).documentId }),
        onDone: {
          target: 'caseLoaded',
          actions: assign({
            loading: false,
            error: null,
            // Update evidence embedding status
            evidenceList: ({ context, event }) =>
              context.evidenceList.map(e =>
                e.id === (event as any).input.documentId
                  ? { ...e, embedding_status: 'completed' }
                  : e
              )
          })
        },
        onError: {
          target: 'caseLoaded',
          actions: assign({
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'Embedding generation failed'
          })
        }
      }
    },

    findingSimilarCases: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'findSimilarCases',
        src: 'findSimilarCases',
        input: ({ event }) => ({
          caseId: (event as any).caseId,
          threshold: (event as any).threshold || 0.7
        }),
        onDone: {
          target: 'caseLoaded',
          actions: assign({
            aiAnalysis: ({ context, event }) => ({
              ...context.aiAnalysis,
              results: {
                ...context.aiAnalysis.results,
                similarity_cases: (event as any).output
              }
            }),
            loading: false,
            error: null
          })
        },
        onError: {
          target: 'caseLoaded',
          actions: assign({
            loading: false,
            error: ({ event }) => (event as any).error?.message || 'Similarity search failed'
          })
        }
      }
    },

    syncing: {
      entry: assign({ loading: true }),
      invoke: {
        id: 'syncDatabase',
        src: 'syncDatabase',
        onDone: {
          target: 'idle',
          actions: assign({
            loading: false,
            dbStatus: 'connected',
            lastSyncTime: () => new Date(),
            error: null
          })
        },
        onError: {
          target: 'idle',
          actions: assign({
            loading: false,
            dbStatus: 'error',
            error: ({ event }) => (event as any).error?.message || 'Database sync failed'
          })
        }
      }
    },

    systemError: {
      on: {
        RETRY: 'initializing',
        SYNC_DB: 'syncing'
      }
    }
  }
}, {
  actors: {
    // Initialize system and check database connection
    initializeSystem: fromPromise(async () => {
      try {
        // Test database connection
        await db.select().from(users).limit(1);

        // Check pgvector extension
        const vectorCheck = await db.execute(
          sql`SELECT extname FROM pg_extension WHERE extname = 'vector';`
        );

        if (vectorCheck.length === 0) {
          throw new Error('pgvector extension not found');
        }

        return { status: 'initialized' };
      } catch (error: any) {
        throw new Error(`Database initialization failed: ${(error as any).message}`);
      }
    }),

    // LOAD_CASE service - Load case with optional evidence
    loadCase: fromPromise(async ({ input }: { input: { caseId: string; includeEvidence: boolean } }) => {
      try {
        // Load case data
        const caseResult = await db
          .select()
          .from(cases)
          .where(eq(cases.id, input.caseId))
          .limit(1);

        if (caseResult.length === 0) {
          throw new Error('Case not found');
        }

        const caseData = caseResult[0];
        let evidenceData = [];

        // Load evidence if requested
        if (input.includeEvidence) {
          evidenceData = await db
            .select({
              id: evidence.id,
              case_id: evidence.case_id,
              title: evidence.title,
              description: evidence.description,
              evidence_type: evidence.evidence_type,
              created_at: evidence.created_at,
              // Check embedding status from document_chunks
              embedding_status: sql<'pending' | 'completed' | 'failed'>`CASE
                  WHEN EXISTS (
                    SELECT 1 FROM document_chunks
                    WHERE document_chunks.document_id = evidence.id
                    AND document_chunks.embedding IS NOT NULL
                  ) THEN 'completed'
                  ELSE 'pending'
                END`
            })
            .from(evidence)
            .where(eq(evidence.case_id, input.caseId))
            .orderBy(desc(evidence.created_at));
        }

        return {
          case: caseData,
          evidence: evidenceData
        };
      } catch (error: any) {
        throw new Error(`Failed to load case: ${(error as any).message}`);
      }
    }),

    // Create new case
    createCase: fromPromise(async ({ input }: { input: CaseForm }) => {
      try {
        const [newCase] = await db
          .insert(cases)
          .values({
            title: input.title,
            description: input.description,
            status: input.status || 'draft',
            // Add other fields as needed
          })
          .returning();

        return newCase;
      } catch (error: any) {
        throw new Error(`Failed to create case: ${(error as any).message}`);
      }
    }),

    // ADD_EVIDENCE service - Add evidence with automatic embedding
    addEvidence: fromPromise(async ({ input }: {
      input: {
        caseId: string;
        evidence: {
          title: string;
          description?: string;
          evidence_type: string;
          file_content?: string;
          metadata?: Record<string, any>;
        }
      }
    }) => {
      try {
        // Insert evidence record
        const [newEvidence] = await db
          .insert(evidence)
          .values({
            case_id: input.caseId,
            title: input.evidence.title,
            description: input.evidence.description,
            evidence_type: input.evidence.evidence_type
          })
          .returning();

        // If there's file content, create document chunks for embedding
        if (input.evidence.file_content) {
          const chunks = chunkText(input.evidence.file_content, 500, 50);

          for (let i = 0; i < chunks.length; i++) {
            await db
              .insert(documentChunks)
              .values({
                document_id: newEvidence.id,
                document_type: 'evidence',
                chunk_index: i.toString(),
                content: chunks[i],
                // embedding will be generated asynchronously
              });
          }
        }

        return {
          ...newEvidence,
          embedding_status: 'pending' as const
        };
      } catch (error: any) {
        throw new Error(`Failed to add evidence: ${(error as any).message}`);
      }
    }),

    // START_AI_ANALYSIS service - Full AI analysis using Gemma3
    startAIAnalysis: fromPromise(async ({ input }: {
      input: { caseId: string; analysisType: 'summary' | 'recommendation' | 'similarity' | 'full' }
    }) => {
      try {
        // Gather case data and evidence
        const caseData = await db
          .select()
          .from(cases)
          .where(eq(cases.id, input.caseId))
          .limit(1);

        if (caseData.length === 0) {
          throw new Error('Case not found');
        }

        const evidenceData = await db
          .select()
          .from(evidence)
          .where(eq(evidence.case_id, input.caseId));

        // Get document chunks for context
        const documentChunksData = await db
          .select()
          .from(documentChunks)
          .where(
            sql`${documentChunks.document_id} IN (
              SELECT ${evidence.id} FROM ${evidence} WHERE ${evidence.case_id} = ${input.caseId}
            )`
          );

        // Prepare context for Gemma3
        const analysisContext = {
          case: caseData[0],
          evidence: evidenceData,
          content: documentChunksData.map(chunk => chunk.content).join('\n\n')
        };

        // Call Gemma3 for analysis (your existing Ollama setup)
        const analysisResponse = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: 'gemma3-legal',
            prompt: buildAnalysisPrompt(analysisContext, input.analysisType),
            stream: false
          })
        });

        if (!analysisResponse.ok) {
          throw new Error('AI analysis request failed');
        }

        const aiResult = await analysisResponse.json();

        // Parse AI response and return structured results
        return parseAnalysisResults(aiResult.response, input.analysisType);
      } catch (error: any) {
        throw new Error(`AI analysis failed: ${(error as any).message}`);
      }
    }),

    // Generate embeddings using nomic-embed-text
    generateEmbeddings: fromPromise(async ({ input }: { input: { documentId: string } }) => {
      try {
        // Get document chunks that need embeddings
        const chunks = await db
          .select()
          .from(documentChunks)
          .where(
            and(
              eq(documentChunks.document_id, input.documentId),
              sql`${documentChunks.embedding} IS NULL`
            )
          );

        for (const chunk of chunks) {
          // Generate embedding using nomic-embed-text
          const embeddingResponse = await fetch('http://localhost:11436/api/embeddings', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              model: 'nomic-embed-text',
              prompt: chunk.content
            })
          });

          if (!embeddingResponse.ok) {
            throw new Error('Embedding generation failed');
          }

          const embeddingResult = await embeddingResponse.json();

          // Update chunk with embedding
          await db
            .update(documentChunks)
            .set({
              embedding: sql`${JSON.stringify(embeddingResult.embedding)}`
            })
            .where(eq(documentChunks.id, chunk.id));
        }

        return { status: 'completed', chunksProcessed: chunks.length };
      } catch (error: any) {
        throw new Error(`Embedding generation failed: ${(error as any).message}`);
      }
    }),

    // Find similar cases using pgvector
    findSimilarCases: fromPromise(async ({ input }: { input: { caseId: string; threshold: number } }) => {
      try {
        // Get case embedding (computed from case content)
        const caseChunks = await db
          .select({
            embedding: documentChunks.embedding,
            chunkId: documentChunks.id,
            evidenceId: evidence.id
          })
          .from(documentChunks)
          .innerJoin(evidence, eq(documentChunks.document_id, evidence.id))
          .where(eq(evidence.case_id, input.caseId));

        if (caseChunks.length === 0) {
          return [];
        }

        // Average embeddings for case representation
        const caseEmbedding = computeAverageEmbedding(caseChunks.map(c => c.embedding || []));

        // Find similar cases using cosine similarity
        const similarCases = await db.execute(
          sql`
            SELECT DISTINCT
              c.id,
              c.title,
              AVG(1 - (dc.embedding <=> ${caseEmbedding})) as similarity_score
            FROM ${cases} c
            INNER JOIN ${evidence} e ON e.case_id = c.id
            INNER JOIN ${documentChunks} dc ON dc.document_id = e.id
            WHERE c.id != ${input.caseId}
              AND dc.embedding IS NOT NULL
            GROUP BY c.id, c.title
            HAVING AVG(1 - (dc.embedding <=> ${caseEmbedding})) > ${input.threshold}
            ORDER BY similarity_score DESC
            LIMIT 10
          `
        );

        return similarCases;
      } catch (error: any) {
        throw new Error(`Similarity search failed: ${(error as any).message}`);
      }
    }),

    // Sync database state
    syncDatabase: fromPromise(async () => {
      try {
        // Perform health checks and sync operations
        await db.select().from(users).limit(1);
        return { status: 'synced', timestamp: new Date() };
      } catch (error: any) {
        throw new Error(`Database sync failed: ${(error as any).message}`);
      }
    })
  }
});

// Helper functions
function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    start = end - overlap;
  }

  return chunks;
}

function buildAnalysisPrompt(context: any, analysisType: string): string {
  const basePrompt = `Analyze the following legal case:

Case Title: ${context.case.title}
Description: ${context.case.description}
Status: ${context.case.status}

Evidence Count: ${context.evidence.length}
Content: ${context.content.slice(0, 2000)}...

Please provide a ${analysisType} analysis focusing on:`;

  switch (analysisType) {
    case 'summary':
      return basePrompt + '\n- Key facts and timeline\n- Main legal issues\n- Current status assessment';
    case 'recommendation':
      return basePrompt + '\n- Next steps\n- Potential risks\n- Strategic recommendations';
    case 'similarity':
      return basePrompt + '\n- Legal precedents\n- Similar case patterns\n- Jurisdictional considerations';
    default:
      return basePrompt + '\n- Comprehensive case analysis\n- Risk assessment\n- Recommendations';
  }
}

function parseAnalysisResults(response: string, analysisType: string) {
  // Parse AI response into structured format
  return {
    summary: response.slice(0, 500),
    keyFindings: response.split('\n').filter(line => line.includes('•')),
    recommendations: response.split('\n').filter(line => line.toLowerCase().includes('recommend')),
    confidence: 0.85 // Placeholder - could be computed from response certainty
  };
}

function computeAverageEmbedding(embeddings: any[]): string {
  // Compute average embedding vector for case representation
  if (embeddings.length === 0) return '[]';

  // Simple averaging (you might want more sophisticated weighting)
  const dim = JSON.parse(embeddings[0]).length;
  const avgEmbedding = new Array(dim).fill(0);

  embeddings.forEach(emb => {
    const vec = JSON.parse(emb);
    vec.forEach((val: number, i: number) => {
      avgEmbedding[i] += val / embeddings.length;
    });
  });

  return JSON.stringify(avgEmbedding);
}

export default enhancedLegalCaseMachine;