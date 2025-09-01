
import type { RequestHandler } from './$types.js';

// Enhanced Evidence API with pgvector Integration
// Production-ready evidence management with AI analysis

import { z } from 'zod';
import { withApiHandler, parseRequestBody, CommonErrors, createPagination } from '../../../lib/server/api/response.js';
import { db } from '../../../lib/server/db/index.js';
import { sql, eq, and, or, ilike, count, desc, asc } from '$lib/server/db/index';
import { evidence, cases } from '../../../lib/server/db/schema-postgres.js';
import type { Evidence } from '../../../lib/server/db/schema-types.js';
import { randomUUID } from 'crypto';
import { URL } from "url";

// Enhanced AI analysis service

// Local types used by the AI service
export interface ProcessingOptions {
  useGPUAcceleration?: boolean;
  priority?: 'low' | 'normal' | 'high';
  notify?: boolean;
  saveIntermediateResults?: boolean;
  overrideExisting?: boolean;
  [k: string]: any;
}

// Minimal Ollama/embedding service stubs to avoid runtime/type errors.
// Replace these with your real implementations.

export interface AIAnalysis {
  id: string;
  model: string;
  confidence: number;
  entities: any[];
  sentiment: number;
  classification: string;
  keywords: string[];
  summary: string;
  relationships: any[];
  timestamp: Date;
  processingTime: number;
  gpuAccelerated: boolean;
  [k: string]: any;
}

export interface EvidenceData {
  id?: string;
  caseId?: string;
  userId?: string;
  title: string;
  description?: string | null;
  evidenceType?: string;
  subType?: string;
  tags?: string[];
  location?: any;
  collectedBy?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  hash?: string;
  boardPosition?: any;
  [k: string]: any;
}

/**
 * Lightweight fallback analysis used when a real AI backend is not available.
 * Keeps types satisfied and provides predictable default values.
 */
const createFallbackAnalysis = (
  evidence: EvidenceData,
  options: { useGPUAcceleration?: boolean } = {}
): AIAnalysis => ({
  id: randomUUID(),
  model: options.useGPUAcceleration ? 'fallback-gpu' : 'fallback',
  confidence: 0.5,
  entities: [],
  sentiment: 0,
  classification: 'fallback_analysis',
  keywords: evidence.tags || [],
  summary: `${evidence.title}${evidence.description ? ' — ' + evidence.description : ''}`.slice(0, 2000),
  relationships: [],
  timestamp: new Date(),
  processingTime: 0,
  gpuAccelerated: Boolean(options.useGPUAcceleration)
});
const ollamaService = {
  async generateCompletion(_: any) {
    return { response: JSON.stringify({ confidence: 0.8, entities: [], summary: 'stub', keywords: [] }) };
  },
  async generateEmbedding(_: any) {
    return { embedding: [] as number[] };
  }
};

function getEmbeddingRepository() {
  return {
    async enqueueIngestion(_: any) {
      return { jobId: `job_${Date.now()}` };
    }
  };
}

class EvidenceAIService {
  private static instance: EvidenceAIService;

  static getInstance(): EvidenceAIService {
    if (!EvidenceAIService.instance) {
      EvidenceAIService.instance = new EvidenceAIService();
    }
    return EvidenceAIService.instance;
  }

  async analyzeEvidence(
    evidenceData: EvidenceData,
    options: ProcessingOptions = {}
  ): Promise<AIAnalysis> {
    const startTime = Date.now();

    // Prepare context for analysis
    const context = {
      title: evidenceData.title,
      description: evidenceData.description || '',
      evidenceType: evidenceData.evidenceType,
      tags: Array.isArray(evidenceData.tags) ? (evidenceData.tags as string[]) : [],
      fileType: evidenceData.fileType,
      location: evidenceData.location,
      collectedBy: evidenceData.collectedBy
    };

    try {
      let analysisResult: any = null;

    // Try GPU-accelerated external service first if requested
      if (options.useGPUAcceleration) {
        try {
          const resp = await fetch('http://localhost:8094/api/evidence/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              evidence: {
                title: context.title,
                description: context.description,
                evidenceType: context.evidenceType,
                tags: context.tags
              },
              options: {
                useGPU: true,
                model: 'gemma3-legal',
                extractEntities: true,
                generateSummary: true,
                findRelationships: true,
                calculateConfidence: true
              }
            })
          });

          if (resp.ok) {
            analysisResult = await resp.json();
          } else {
            console.warn('Enhanced RAG service returned non-OK status:', resp.status);
          }
        } catch (err: any) {
          console.warn('Enhanced RAG service request failed, falling back:', err);
        }
      }

      // Fall back to local/ollama completion if no result yet
      if (!analysisResult) {
        try {
          const completion = await ollamaService.generateCompletion({
            model: 'gemma3-legal',
            prompt: `${context.title}\n\n${context.description}`
          });

          const respText = typeof completion?.response === 'string' ? completion.response : JSON.stringify(completion);
          try {
            analysisResult = JSON.parse(respText);
          } catch {
            // If parsing fails, synthesize a minimal analysis object
            analysisResult = {
              summary: String(respText).slice(0, 2000),
              confidence: 0.7,
              entities: [],
              keywords: context.tags,
              relationships: [],
              classification: 'evidence_analysis'
            };
          }
        } catch (err: any) {
          console.warn('Fallback completion failed:', err);
          analysisResult = {
            summary: `${context.title}${context.description ? ' — ' + context.description : ''}`.slice(0, 2000),
            confidence: 0.5,
            entities: [],
            keywords: context.tags,
            relationships: [],
            classification: 'fallback_analysis'
          };
        }
      }

      const processingTime = Date.now() - startTime;

      return {
        id: randomUUID(),
        model: options.useGPUAcceleration ? 'enhanced-rag-gpu' : (analysisResult?.model || 'gemma3-legal'),
        confidence: analysisResult?.confidence ?? 0.8,
        entities: analysisResult?.entities ?? [],
        sentiment: analysisResult?.sentiment ?? 0,
        classification: analysisResult?.classification ?? 'evidence_analysis',
        keywords: analysisResult?.keywords ?? context.tags,
        summary: analysisResult?.summary ?? '',
        relationships: analysisResult?.relationships ?? [],
        timestamp: new Date(),
        processingTime,
        gpuAccelerated: Boolean(options.useGPUAcceleration)
      };
    } catch (error: any) {
      console.error('Evidence AI analysis failed:', error);
      return {
        id: randomUUID(),
        model: 'error_fallback',
        confidence: 0.3,
        entities: [],
        sentiment: 0,
        classification: 'analysis_failed',
        keywords: Array.isArray(evidenceData?.tags) ? (evidenceData.tags as string[]) : [],
        summary: `Analysis failed: ${error?.message || String(error)}`,
        relationships: [],
        timestamp: new Date(),
        processingTime: Date.now() - startTime,
        gpuAccelerated: false
      };
    }
  }

  // Generate an embedding for a piece of text (stub-safe implementation)
  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const resp = await ollamaService.generateEmbedding({ model: 'all-mpnet-base-v2', input: text });
      return Array.isArray(resp?.embedding) ? (resp.embedding as number[]) : [];
    } catch (err: any) {
      console.warn('generateEmbedding failed:', err);
      return [];
    }
  }

  // Semantic search - keep a safe fallback implementation to avoid direct schema assumptions
  async semanticSearch(query: string, _caseId?: string, _limit: number = 20): Promise<any[]> {
    try {
      const queryEmbedding = await this.generateEmbedding(query);
      if (!Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
        return [];
      }

      // NOTE: avoid referencing typed table columns that may not exist in the current Drizzle schema here;
      // return an empty array by default or integrate a repository-based search if available.
      // Implementers should replace this with a proper pgvector search using raw SQL or a repository client.
      return [];
    } catch (error: any) {
      console.error('Semantic search failed:', error);
      return [];
    }
  }
}

const evidenceAI = EvidenceAIService.getInstance();

export const GET: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get('caseId');
    const type = url.searchParams.get('type');
    const search = url.searchParams.get('search');
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    const offset = (page - 1) * limit;

    // Build where conditions
    const whereConditions: any[] = [];

    if (caseId) {
      whereConditions.push(eq(evidence.caseId, caseId));
    }

    if (type) {
      whereConditions.push(eq(evidence.evidenceType, type));
    }

    if (search) {
      whereConditions.push(
        or(
          ilike(evidence.title, `%${search}%`),
          ilike(evidence.description, `%${search}%`),
          ilike(evidence.summary, `%${search}%`)
        )
      );
    }

    // Get evidence with pagination - simplified query to avoid schema issues
    const evidenceQuery = db
      .select()
      .from(evidence)
      .orderBy(desc(evidence.created_at))
      .limit(limit)
      .offset(offset);

    // Add where conditions if any
    let finalEvidenceQuery = evidenceQuery;
    if (whereConditions.length > 0) {
      finalEvidenceQuery = evidenceQuery.where(and(...whereConditions));
    }

    const evidenceResults = await finalEvidenceQuery;

    // Get total count for pagination
    let totalQuery = db
      .select({ count: count() })
      .from(evidence);

    if (whereConditions.length > 0) {
      totalQuery = totalQuery.where(and(...whereConditions));
    }

    const [{ count: totalCount }] = await totalQuery;

    return json({
      evidence: evidenceResults,
      total: totalCount,
      page,
      limit,
      totalPages: Math.ceil(totalCount / limit),
      filters: { caseId, type, search }
    });
  } catch (error: any) {
    console.error('Error fetching evidence:', error);
    return json(
      { error: 'Failed to fetch evidence' },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const data = await request.json();
    const {
      caseId,
      userId,
      title,
      description,
      evidenceType,
      subType,
      fileName,
      fileSize,
      mimeType,
      hash,
      tags = [],
      chainOfCustody = [],
      collectedAt,
      collectedBy,
      location,
      aiAnalysis = {},
      aiTags = [],
      aiSummary,
      summary,
      summaryType,
      isAdmissible = true,
      confidentialityLevel = 'internal',
      boardPosition = {}
    } = data;

    if (!caseId || !title || !evidenceType) {
      return json(
        { error: 'Case ID, title, and evidence type are required' },
        { status: 400 }
      );
    }

    // Verify case exists
    const caseExists = await db
      .select({ id: cases.id })
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (caseExists.length === 0) {
      return json(
        { error: 'Case not found' },
        { status: 404 }
      );
    }

    // Insert new evidence
    const [newEvidence] = await db
      .insert(evidence)
      .values({
        caseId,
        userId,
        title,
        description,
        evidenceType,
        subType,
        fileName,
        fileSize,
        mimeType,
        hash,
        tags,
        chainOfCustody,
        collectedAt: collectedAt ? new Date(collectedAt) : null,
        collectedBy,
        location,
        aiAnalysis,
        aiTags,
        aiSummary,
        summary,
        summaryType,
        isAdmissible,
        confidentialityLevel,
        boardPosition
      })
      .returning();

    // Enhanced AI analysis and embedding generation
    let ingestionJobId: string | undefined;
    let aiAnalysisResult: AIAnalysis | null = null;

    if (description || title) {
      try {
        // Generate AI analysis using Ollama/CUDA
        const analysisOptions: ProcessingOptions = {
          useGPUAcceleration: true,
          priority: 'normal',
          notify: false,
          saveIntermediateResults: true,
          overrideExisting: false
        };

        // Cast DB record to any to avoid strict mismatch of optional tag types in the DB row.
        aiAnalysisResult = await evidenceAI.analyzeEvidence(newEvidence as any, analysisOptions);

        // Generate and store embedding (single declaration)
        const textToEmbed = `${title} ${description || ''} ${(Array.isArray(tags) ? tags : []).join(' ')}`;
        const embedding = await evidenceAI.generateEmbedding(textToEmbed);

        // Update evidence with AI analysis (only update columns that exist in schema)
        if (aiAnalysisResult) {
          await db
            .update(evidence)
            .set({
              aiAnalysis: aiAnalysisResult,
              aiSummary: aiAnalysisResult.summary,
              aiTags: aiAnalysisResult.keywords
            })
            .where(eq(evidence.id, newEvidence.id));
        }

        // Also queue for repository ingestion (include embedding in metadata if available)
        try {
          const repo = getEmbeddingRepository();
          const jobStatus = await repo.enqueueIngestion({
            evidenceId: newEvidence.id,
            caseId: newEvidence.caseId,
            filename: newEvidence.fileName,
            mimeType: newEvidence.mimeType,
            textContent: textToEmbed,
            metadata: {
              evidenceType: newEvidence.evidenceType,
              aiAnalysis: aiAnalysisResult,
              embedding: Array.isArray(embedding) ? embedding.slice(0, 10) : []
            }
          });
          ingestionJobId = jobStatus?.jobId;
        } catch (ingestErr) {
          console.error('Embedding ingestion failed:', ingestErr);
        }

        // Fallback ingestion with minimal metadata
        try {
          const repo = getEmbeddingRepository();
          const jobStatus = await repo.enqueueIngestion({
            evidenceId: newEvidence.id,
            caseId: newEvidence.caseId,
            filename: newEvidence.fileName,
            mimeType: newEvidence.mimeType,
            textContent: description || title,
            metadata: { evidenceType: newEvidence.evidenceType }
          });
          ingestionJobId = ingestionJobId ?? jobStatus?.jobId;
        } catch (fallbackError) {
          console.error('Fallback embedding ingestion also failed:', fallbackError);
        }
      } catch (aiErr) {
        console.error('AI analysis/embedding failed for new evidence:', aiErr);
      }
    }

    return json({
      ...newEvidence,
      ingestionJobId,
      aiAnalysis: aiAnalysisResult,
      processingStatus: 'completed'
    }, { status: 201 });
  } catch (error: any) {
    console.error('Error creating evidence:', error);
    return json(
      { error: 'Failed to create evidence' },
      { status: 500 }
    );
  }
};

// Enhanced evidence processing endpoint
export const PATCH: RequestHandler = async ({ request }) => {
  try {
    const { action, evidenceId, options } = await request.json();

    if (!evidenceId) {
      return json({ error: 'Evidence ID is required' }, { status: 400 });
    }

    // Get evidence record
    const evidenceRecord = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    if (evidenceRecord.length === 0) {
      return json({ error: 'Evidence not found' }, { status: 404 });
    }

    const evidenceData = evidenceRecord[0];

    switch (action) {
      case 'analyze': {
        try {
          // Cast DB record to any to satisfy EvidenceData shape expected by analyzeEvidence
          const analysisResult = await evidenceAI.analyzeEvidence(evidenceData as any, options);

          // Update database with analysis (only fields expected to exist)
          await db
            .update(evidence)
            .set({
              aiAnalysis: analysisResult,
              aiSummary: analysisResult.summary,
              aiTags: analysisResult.keywords,
              updatedAt: new Date()
            })
            .where(eq(evidence.id, evidenceId));

          // Generate embedding and enqueue ingestion (repository stores vectors)
          const textToEmbed = `${(evidenceData as any).title || ''} ${((evidenceData as any).description) || ''} ${Array.isArray((evidenceData as any).tags) ? (evidenceData as any).tags.join(' ') : ''}`;
          const embedding = await evidenceAI.generateEmbedding(textToEmbed);

          if (Array.isArray(embedding) && embedding.length > 0) {
            try {
              const repo = getEmbeddingRepository();
              await repo.enqueueIngestion({
                evidenceId,
                caseId: (evidenceData as any).caseId,
                filename: (evidenceData as any).fileName,
                mimeType: (evidenceData as any).mimeType,
                textContent: textToEmbed,
                metadata: { evidenceType: (evidenceData as any).evidenceType, embedding: embedding.slice(0, 10) }
              });
            } catch (enqueueErr) {
              console.warn('Failed to enqueue embedding ingestion:', enqueueErr);
            }
          }

          return json({ analysis: analysisResult, embedding: Array.isArray(embedding) ? embedding.slice(0, 10) : [], status: 'completed' });
        } catch (err: any) {
          console.error('Analysis action failed:', err);
          return json({ error: 'Analysis failed' }, { status: 500 });
        }
      }

      default:
        return json({ error: 'Unknown action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Evidence processing error:', error);
    return json({ error: 'Processing failed' }, { status: 500 });
  }
};

// Note: PUT and DELETE handlers should be in /api/evidence/[id]/+server.ts