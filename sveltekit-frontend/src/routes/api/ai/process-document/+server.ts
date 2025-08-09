// @ts-nocheck
// AI Document Processing API - Summarization, Entity Extraction, Embeddings
// Production-ready endpoint with LangChain + Ollama integration

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { z } from 'zod';
import { db } from '$lib/server/database';
import { documents, aiProcessingJobs } from '$lib/database/enhanced-schema';
import { eq, and, sql } from 'drizzle-orm';
import { langChainService } from '$lib/ai/langchain-ollama-service';
import { RateLimiter } from '$lib/utils/rate-limit';
import { authenticateUser } from '$lib/server/auth';

// ============================================================================
// RATE LIMITERS
// ============================================================================

const aiProcessingLimiter = new RateLimiter({
  maxRequests: 10,
  windowMs: 60 * 1000, // 1 minute
  message: 'Too many AI processing requests'
});

const aiBatchProcessingLimiter = new RateLimiter({
  maxRequests: 3,
  windowMs: 60 * 1000, // 1 minute
  message: 'Too many AI batch processing requests'
});

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

const ProcessDocumentSchema = z.object({
  documentId: z.string().uuid(),
  generateSummary: z.boolean().default(true),
  extractEntities: z.boolean().default(true),
  riskAssessment: z.boolean().default(true),
  generateRecommendations: z.boolean().default(false),
  chunkForEmbeddings: z.boolean().default(true),
  customPrompt: z.string().optional(),
  model: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  maxTokens: z.number().min(100).max(8000).optional()
});

type ProcessDocumentRequest = z.infer<typeof ProcessDocumentSchema>;

interface ProcessingResponse {
  success: boolean;
  documentId: string;
  jobId: string;
  summary?: string;
  keyPoints?: string[];
  entities?: Array<{
    text: string;
    type: string;
    confidence: number;
  }>;
  riskAssessment?: string;
  recommendations?: string[];
  embeddings?: {
    chunksCreated: number;
    vectorDimensions: number;
    model: string;
  };
  metadata: {
    processingTime: number;
    tokensUsed: number;
    confidence: number;
    model: string;
    usedCuda: boolean;
  };
}

// ============================================================================
// MAIN PROCESSING ENDPOINT
// ============================================================================

export const POST: RequestHandler = async ({ request, getClientAddress, cookies }) => {
  const startTime = Date.now();
  const clientIP = getClientAddress();

  try {
    // Rate limiting
    const isAllowed = aiProcessingLimiter.isAllowed(clientIP);

    if (!isAllowed) {
      return json(
        {
          error: 'Rate limit exceeded',
          retryAfter: 60000, // 1 minute
        },
        { status: 429 }
      );
    }

    // Authentication
    const user = await authenticateUser(request);
    if (!user) {
      return json({ error: 'Authentication required' }, { status: 401 });
    }

    // Parse and validate request
    const rawData = await request.json();
    const validationResult = ProcessDocumentSchema.safeParse(rawData);

    if (!validationResult.success) {
      return json(
        {
          error: 'Invalid request data',
          details: validationResult.error.flatten(),
        },
        { status: 400 }
      );
    }

    const data = validationResult.data;

    // Verify document exists and user has access
    const document = await db
      .select()
      .from(documents)
      .where(
        and(
          eq(documents.id, data.documentId),
          eq(documents.createdBy, user.id)
        )
      )
      .limit(1);

    if (document.length === 0) {
      return json(
        { error: 'Document not found or access denied' },
        { status: 404 }
      );
    }

    const doc = document[0];

    if (!doc.extractedText || doc.extractedText.length === 0) {
      return json(
        { error: 'Document has no extractable text content' },
        { status: 400 }
      );
    }

    // Create processing job
    const [job] = await db
      .insert(aiProcessingJobs)
      .values({
        type: 'comprehensive_analysis',
        status: 'processing',
        input: {
          documentId: data.documentId,
          entityType: 'document',
          entityId: data.documentId,
          model: data.model || 'llama3.2',
          generateSummary: data.generateSummary,
          extractEntities: data.extractEntities,
          riskAssessment: data.riskAssessment,
          generateRecommendations: data.generateRecommendations,
          chunkForEmbeddings: data.chunkForEmbeddings,
          startedAt: new Date().toISOString(),
          userId: user.id,
        },
      })
      .returning();

    try {
      // Process the document with LangChain service
      const response: ProcessingResponse = {
        success: true,
        documentId: data.documentId,
        jobId: job.id,
        metadata: {
          processingTime: 0,
          tokensUsed: 0,
          confidence: 0,
          model: data.model || 'llama3.2',
          usedCuda: false,
        },
      };

      // 1. Generate embeddings and chunks if requested
      if (data.chunkForEmbeddings) {
        console.log('Processing document for embeddings...');
        const embeddingResult = await langChainService.processDocument(
          data.documentId,
          doc.extractedText,
          {
            title: doc.title,
            filename: doc.filename,
            documentType: doc.fileType,
          }
        );

        response.embeddings = {
          chunksCreated: embeddingResult.chunksCreated,
          vectorDimensions: 768, // nomic-embed-text
          model: embeddingResult.metadata.model,
        };

        response.metadata.usedCuda = embeddingResult.metadata.usedCuda;
      }

      // 2. Generate AI analysis
      if (data.generateSummary || data.extractEntities || data.riskAssessment) {
        console.log('Generating AI analysis...');
        const analysisResult = await langChainService.summarizeDocument(
          data.documentId,
          doc.extractedText,
          {
            extractEntities: data.extractEntities,
            riskAssessment: data.riskAssessment,
            generateRecommendations: data.generateRecommendations,
          }
        );

        if (data.generateSummary) {
          response.summary = analysisResult.summary;
          response.keyPoints = analysisResult.keyPoints;
        }

        if (data.extractEntities) {
          response.entities = analysisResult.entities;
        }

        if (data.riskAssessment) {
          response.riskAssessment = analysisResult.riskAssessment;
        }

        if (data.generateRecommendations) {
          response.recommendations = analysisResult.recommendations;
        }

        response.metadata.confidence = analysisResult.confidence;
      }

      // 3. Update document with AI analysis results
      const aiAnalysis = {
        summary: response.summary,
        keyPoints: response.keyPoints,
        entities: response.entities,
        riskAssessment: response.riskAssessment,
        recommendations: response.recommendations,
        confidence: response.metadata.confidence,
        processedAt: new Date().toISOString(),
        model: response.metadata.model,
      };

      await db
        .update(documents)
        .set({
          metadata: {
            ...(doc.metadata || {}),
            aiAnalysis,
            lastProcessed: new Date().toISOString(),
          },
        })
        .where(eq(documents.id, data.documentId));

      // 4. Complete the processing job
      const processingTime = Date.now() - startTime;
      response.metadata.processingTime = processingTime;
      response.metadata.tokensUsed = Math.ceil(doc.extractedText.length / 4); // Rough estimation

      await db
        .update(aiProcessingJobs)
        .set({
          status: 'completed',
          progress: 100,
          output: response,
          completedAt: new Date(),
        })
        .where(eq(aiProcessingJobs.id, job.id));

      console.log(`Document processing completed in ${processingTime}ms`);
      return json(response);

    } catch (processingError) {
      console.error('Document processing error:', processingError);

      // Update job with error status
      await db
        .update(aiProcessingJobs)
        .set({
          status: 'failed',
          error: String(processingError),
        })
        .where(eq(aiProcessingJobs.id, job.id));

      throw processingError;
    }

  } catch (error) {
    console.error('API error:', error);
    return json(
      {
        error: 'Document processing failed',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
};

// ============================================================================
// GET PROCESSING STATUS
// ============================================================================

export const GET: RequestHandler = async ({ url, cookies }) => {
  try {
    // Authentication
    const user = await authenticateUser(request);
    if (!user) {
      return json({ error: 'Authentication required' }, { status: 401 });
    }

    const documentId = url.searchParams.get('documentId');
    const jobId = url.searchParams.get('jobId');

    if (!documentId && !jobId) {
      return json(
        { error: 'documentId or jobId parameter required' },
        { status: 400 }
      );
    }

    let jobs;

    if (jobId) {
      jobs = await db
        .select()
        .from(aiProcessingJobs)
        .where(eq(aiProcessingJobs.id, jobId))
        .limit(1);
    } else {
      jobs = await db
        .select()
        .from(aiProcessingJobs)
        .where(
          sql`${aiProcessingJobs.input}->>'documentId' = ${documentId}`
        )
        .orderBy(aiProcessingJobs.createdAt)
        .limit(10);
    }

    return json({
      success: true,
      jobs: jobs.map(job: any => ({
        id: job.id,
        documentId: (job.input as any)?.documentId,
        status: job.status,
        progress: job.progress,
        jobType: job.jobType,
        model: job.model,
        error: job.error,
        createdAt: job.createdAt,
        metadata: job.metadata,
        outputData: job.outputData,
      })),
    });

  } catch (error) {
    console.error('Status check error:', error);
    return json(
      {
        error: 'Failed to get processing status',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
};

// ============================================================================
// BATCH PROCESSING ENDPOINT
// ============================================================================

export const PUT: RequestHandler = async ({ request, getClientAddress, cookies }) => {
  const clientIP = getClientAddress();

  try {
    // Enhanced rate limiting for batch operations
    const isBatchAllowed = aiBatchProcessingLimiter.isAllowed(clientIP);

    if (!isBatchAllowed) {
      return json(
        {
          error: 'Rate limit exceeded for batch operations',
          retryAfter: 60000, // 1 minute
        },
        { status: 429 }
      );
    }

    // Authentication
    const user = await authenticateUser(request);
    if (!user) {
      return json({ error: 'Authentication required' }, { status: 401 });
    }

    // Parse batch request
    const { documentIds, options } = await request.json();

    if (!Array.isArray(documentIds) || documentIds.length === 0) {
      return json(
        { error: 'documentIds array is required' },
        { status: 400 }
      );
    }

    if (documentIds.length > 10) {
      return json(
        { error: 'Maximum 10 documents per batch' },
        { status: 400 }
      );
    }

    // Verify all documents exist and user has access
    const userDocuments = await db
      .select()
      .from(documents)
      .where(
        and(
          eq(documents.createdBy, user.id)
        )
      );

    const validDocumentIds = userDocuments
      .filter(doc: any => documentIds.includes(doc.id))
      .map(doc: any => doc.id);

    if (validDocumentIds.length !== documentIds.length) {
      return json(
        { error: 'Some documents not found or access denied' },
        { status: 404 }
      );
    }

    // Create batch processing jobs
    const batchId = crypto.randomUUID();
    const jobs = [];

    for (const documentId of validDocumentIds) {
      const [job] = await db
        .insert(aiProcessingJobs)
        .values({
          type: 'batch_analysis',
          status: 'queued',
          input: {
            documentId,
            entityType: 'document',
            batchId,
            ...options,
            queuedAt: new Date().toISOString(),
          },
        })
        .returning();

      jobs.push(job);
    }

    // Start background processing (would typically use a job queue like Bull/BullMQ)
    // For now, we'll process sequentially with a timeout
    setTimeout(async () => {
      for (const job of jobs) {
        try {
          await db
            .update(aiProcessingJobs)
            .set({ status: 'processing' })
            .where(eq(aiProcessingJobs.id, job.id));

          // Process document (simplified for batch)
          const doc = userDocuments.find(d: any => d.id === job.entityId);
          if (doc && doc.extractedText) {
            const result = await langChainService.summarizeDocument(
              job.entityId,
              doc.extractedText,
              {
                extractEntities: options.extractEntities ?? true,
                riskAssessment: options.riskAssessment ?? true,
              }
            );

            await db
              .update(aiProcessingJobs)
              .set({
                status: 'completed',
                progress: 100,
                output: result,
                completedAt: new Date(),
              })
              .where(eq(aiProcessingJobs.id, job.id));
          }

        } catch (error) {
          await db
            .update(aiProcessingJobs)
            .set({
              status: 'failed',
              error: String(error),
            })
            .where(eq(aiProcessingJobs.id, job.id));
        }
      }
    }, 1000); // Start processing after 1 second

    return json({
      success: true,
      batchId,
      jobIds: jobs.map(job: any => job.id),
      message: `Batch processing started for ${validDocumentIds.length} documents`,
    });

  } catch (error) {
    console.error('Batch processing error:', error);
    return json(
      {
        error: 'Batch processing failed',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
};