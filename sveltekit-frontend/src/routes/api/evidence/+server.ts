// @ts-nocheck
import { type RequestHandler, json } from '@sveltejs/kit';
import { getEmbeddingRepository } from '$lib/server/embedding/embedding-repository';
import { db } from "$lib/server/db/drizzle";
import { eq, and, or, ilike, count, desc, asc, sql } from "drizzle-orm";
import { evidence, cases } from '$lib/server/db/schema-postgres-enhanced';
import { ollamaService } from '$lib/services/ollama-service';
import type { 
  EvidenceData, 
  AIAnalysis, 
  ProcessingRequest,
  ProcessingOptions 
} from '$lib/types/evidence';

// Enhanced AI analysis service
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
    
    try {
      // Prepare context for analysis
      const analysisContext = {
        title: evidenceData.title,
        description: evidenceData.description || '',
        evidenceType: evidenceData.evidenceType,
        tags: evidenceData.tags || [],
        fileType: evidenceData.fileType,
        location: evidenceData.location,
        collectedBy: evidenceData.collectedBy
      };

      // Use enhanced RAG service for analysis
      let analysisResult;
      if (options.useGPUAcceleration) {
        // Use Go-based enhanced RAG service with GPU acceleration
        const response = await fetch('http://localhost:8094/api/evidence/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            evidence: analysisContext,
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
        
        if (response.ok) {
          analysisResult = await response.json();
        } else {
          throw new Error(`Enhanced RAG service error: ${response.statusText}`);
        }
      } else {
        // Fallback to Ollama direct analysis
        const prompt = `Analyze this legal evidence and provide a comprehensive analysis:

Title: ${analysisContext.title}
Type: ${analysisContext.evidenceType}
Description: ${analysisContext.description}
Tags: ${analysisContext.tags.join(', ')}

Provide:
1. Key entities (people, locations, organizations, dates, legal terms)
2. Evidence classification and significance
3. Potential relationships to other evidence
4. Legal implications and admissibility concerns
5. Summary and key findings

Respond in JSON format with structured analysis.`;
        
        const response = await ollamaService.generateCompletion({
          model: 'gemma3-legal',
          prompt,
          options: {
            temperature: 0.1,
            top_p: 0.9,
            max_tokens: 1000
          }
        });
        
        try {
          analysisResult = JSON.parse(response.response);
        } catch {
          // Fallback structured analysis if JSON parsing fails
          analysisResult = {
            entities: [],
            classification: 'evidence_analysis',
            summary: response.response.substring(0, 500),
            confidence: 0.7,
            relationships: [],
            keywords: analysisContext.tags
          };
        }
      }

      const processingTime = Date.now() - startTime;
      
      return {
        id: crypto.randomUUID(),
        model: options.useGPUAcceleration ? 'enhanced-rag-gpu' : 'gemma3-legal',
        confidence: analysisResult.confidence || 0.8,
        entities: analysisResult.entities || [],
        sentiment: analysisResult.sentiment || 0,
        classification: analysisResult.classification || 'evidence_analysis',
        keywords: analysisResult.keywords || analysisContext.tags,
        summary: analysisResult.summary || '',
        relationships: analysisResult.relationships || [],
        timestamp: new Date(),
        processingTime,
        gpuAccelerated: options.useGPUAcceleration || false
      };
    } catch (error) {
      console.error('Evidence AI analysis failed:', error);
      
      // Return minimal analysis on error
      return {
        id: crypto.randomUUID(),
        model: 'error_fallback',
        confidence: 0.3,
        entities: [],
        sentiment: 0,
        classification: 'analysis_failed',
        keywords: evidenceData.tags || [],
        summary: `Analysis failed: ${error.message}`,
        relationships: [],
        timestamp: new Date(),
        processingTime: Date.now() - startTime,
        gpuAccelerated: false
      };
    }
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await ollamaService.generateEmbedding({
        model: 'nomic-embed-text',
        prompt: text
      });
      return response.embedding;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      return [];
    }
  }

  async semanticSearch(
    query: string, 
    caseId?: string, 
    limit: number = 20
  ): Promise<any[]> {
    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);
      
      if (queryEmbedding.length === 0) {
        return [];
      }

      // Build WHERE clause
      const whereConditions = [];
      if (caseId) {
        whereConditions.push(sql`${evidence.caseId} = ${caseId}`);
      }

      // PostgreSQL pgvector similarity search
      const similarityThreshold = 0.7;
      const results = await db
        .select({
          id: evidence.id,
          caseId: evidence.caseId,
          title: evidence.title,
          description: evidence.description,
          evidenceType: evidence.evidenceType,
          fileName: evidence.fileName,
          fileUrl: evidence.fileUrl,
          tags: evidence.tags,
          summary: evidence.summary,
          aiAnalysis: evidence.aiAnalysis,
          uploadedAt: evidence.uploadedAt,
          similarity: sql<number>`1 - (${evidence.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector) as similarity`
        })
        .from(evidence)
        .where(
          and(
            ...whereConditions,
            sql`${evidence.embedding} IS NOT NULL`,
            sql`1 - (${evidence.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector) > ${similarityThreshold}`
          )
        )
        .orderBy(sql`similarity DESC`)
        .limit(limit);
        
      return results;
    } catch (error) {
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
    const whereConditions = [];

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

    // Get evidence with pagination
    const evidenceQuery = db
      .select({
        id: evidence.id,
        caseId: evidence.caseId,
        criminalId: evidence.criminalId,
        title: evidence.title,
        description: evidence.description,
        evidenceType: evidence.evidenceType,
        fileType: evidence.fileType,
        subType: evidence.subType,
        fileUrl: evidence.fileUrl,
        fileName: evidence.fileName,
        fileSize: evidence.fileSize,
        mimeType: evidence.mimeType,
        hash: evidence.hash,
        tags: evidence.tags,
        chainOfCustody: evidence.chainOfCustody,
        collectedAt: evidence.collectedAt,
        collectedBy: evidence.collectedBy,
        location: evidence.location,
        labAnalysis: evidence.labAnalysis,
        aiAnalysis: evidence.aiAnalysis,
        aiTags: evidence.aiTags,
        aiSummary: evidence.aiSummary,
        summary: evidence.summary,
        isAdmissible: evidence.isAdmissible,
        confidentialityLevel: evidence.confidentialityLevel,
        canvasPosition: evidence.canvasPosition,
        uploadedBy: evidence.uploadedBy,
        uploadedAt: evidence.uploadedAt,
        updatedAt: evidence.updatedAt
      })
      .from(evidence)
      .orderBy(desc(evidence.uploadedAt))
      .limit(limit)
      .offset(offset);

    // Add where conditions if any
    if (whereConditions.length > 0) {
      evidenceQuery.where(and(...whereConditions));
    }

    const evidenceResults = await evidenceQuery;

    // Get total count for pagination
    const totalQuery = db
      .select({ count: count() })
      .from(evidence);

    if (whereConditions.length > 0) {
      totalQuery.where(and(...whereConditions));
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
  } catch (error) {
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
      criminalId,
      title,
      description,
      evidenceType,
      fileType,
      subType,
      fileUrl,
      fileName,
      fileSize,
      mimeType,
      hash,
      tags = [],
      chainOfCustody = [],
      collectedAt,
      collectedBy,
      location,
      labAnalysis = {},
      aiAnalysis = {},
      aiTags = [],
      aiSummary,
      summary,
      isAdmissible = true,
      confidentialityLevel = 'standard',
      canvasPosition = {},
      uploadedBy
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
        criminalId,
        title,
        description,
        evidenceType,
        fileType,
        subType,
        fileUrl,
        fileName,
        fileSize,
        mimeType,
        hash,
        tags,
        chainOfCustody,
        collectedAt: collectedAt ? new Date(collectedAt) : null,
        collectedBy,
        location,
        labAnalysis,
        aiAnalysis,
        aiTags,
        aiSummary,
        summary,
        isAdmissible,
        confidentialityLevel,
        canvasPosition,
        uploadedBy
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
        
        aiAnalysisResult = await evidenceAI.analyzeEvidence(newEvidence, analysisOptions);
        
        // Generate and store embedding
        const textToEmbed = `${title} ${description || ''} ${(tags || []).join(' ')}`;
        const embedding = await evidenceAI.generateEmbedding(textToEmbed);
        
        // Update evidence with AI analysis and embedding
        if (embedding.length > 0) {
          await db
            .update(evidence)
            .set({ 
              aiAnalysis: aiAnalysisResult,
              embedding: JSON.stringify(embedding),
              aiSummary: aiAnalysisResult.summary,
              aiTags: aiAnalysisResult.keywords
            })
            .where(eq(evidence.id, newEvidence.id));
        }
        
        // Also queue for repository ingestion
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
            embedding: embedding.slice(0, 10) // Sample for metadata
          }
        });
        ingestionJobId = jobStatus?.jobId;
        
      } catch (e) {
        console.warn('Failed to perform AI analysis/embedding:', e);
        
        // Fallback to basic embedding repository
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
          ingestionJobId = jobStatus?.jobId;
        } catch (fallbackError) {
          console.error('Fallback embedding ingestion also failed:', fallbackError);
        }
      }
    }

    return json({ 
      ...newEvidence, 
      ingestionJobId,
      aiAnalysis: aiAnalysisResult,
      processingStatus: 'completed'
    }, { status: 201 });
  } catch (error) {
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
        const analysisResult = await evidenceAI.analyzeEvidence(evidenceData, options);
        
        // Update database with analysis
        await db
          .update(evidence)
          .set({
            aiAnalysis: analysisResult,
            aiSummary: analysisResult.summary,
            aiTags: analysisResult.keywords,
            updatedAt: new Date()
          })
          .where(eq(evidence.id, evidenceId));

        return json({ analysis: analysisResult, status: 'completed' });
      }

      case 'reembed': {
        const textToEmbed = `${evidenceData.title} ${evidenceData.description || ''} ${(evidenceData.tags || []).join(' ')}`;
        const embedding = await evidenceAI.generateEmbedding(textToEmbed);
        
        if (embedding.length > 0) {
          await db
            .update(evidence)
            .set({
              embedding: JSON.stringify(embedding),
              updatedAt: new Date()
            })
            .where(eq(evidence.id, evidenceId));
        }

        return json({ embedding: embedding.slice(0, 10), status: 'completed' });
      }

      case 'semantic_search': {
        const { query, limit = 20 } = options;
        const results = await evidenceAI.semanticSearch(
          query, 
          evidenceData.caseId, 
          limit
        );
        
        return json({ results, query, total: results.length });
      }

      default:
        return json({ error: 'Unknown action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Evidence processing error:', error);
    return json({ error: 'Processing failed' }, { status: 500 });
  }
};

// Note: PUT and DELETE handlers should be in /api/evidence/[id]/+server.ts