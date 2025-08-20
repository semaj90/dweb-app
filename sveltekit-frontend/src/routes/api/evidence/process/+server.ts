// Enhanced evidence processing endpoint with WebAssembly + WebGPU integration
// POST: Start comprehensive evidence processing with AI analysis
// GET:  ?jobId=... returns detailed processing status
// DELETE: Cancel active processing job

import { json, type RequestHandler } from '@sveltejs/kit';
import { processNextJob, getJobStatus } from '$lib/server/embedding/pgvector-embedding-repository';
import { db } from '$lib/server/db/drizzle';
import { evidence } from '$lib/server/db/schema-postgres-enhanced';
import { eq } from 'drizzle-orm';
import type { 
  ProcessingRequest, 
  ProcessingResult,
  ProcessingStep,
  ProcessingOptions,
  EvidenceData,
  AIAnalysis
} from '$lib/types/evidence';

// Enhanced evidence processing service with WebAssembly + WebGPU integration
class EvidenceProcessingService {
  private static instance: EvidenceProcessingService;
  private processingJobs: Map<string, ProcessingResult> = new Map();
  private wsConnections: Map<string, WebSocket> = new Map();

  static getInstance(): EvidenceProcessingService {
    if (!EvidenceProcessingService.instance) {
      EvidenceProcessingService.instance = new EvidenceProcessingService();
    }
    return EvidenceProcessingService.instance;
  }

  async startProcessing(request: ProcessingRequest): Promise<{ sessionId: string; jobId: string }> {
    const sessionId = crypto.randomUUID();
    const jobId = crypto.randomUUID();

    // Get evidence data
    const evidenceRecord = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, request.evidenceId))
      .limit(1);

    if (evidenceRecord.length === 0) {
      throw new Error('Evidence not found');
    }

    const evidenceData = evidenceRecord[0];

    // Initialize processing result
    const processingResult: ProcessingResult = {
      jobId,
      status: 'processing',
      progress: 0,
      step: request.steps[0] as ProcessingStep,
      stepProgress: 0,
      results: null,
      error: null,
      startTime: new Date(),
      processingTime: 0,
      gpuAccelerated: request.options.useGPUAcceleration || false
    };

    this.processingJobs.set(jobId, processingResult);

    // Start processing in background
    this.processEvidence(sessionId, jobId, evidenceData, request).catch(error => {
      console.error('Processing failed:', error);
      const failedResult = this.processingJobs.get(jobId);
      if (failedResult) {
        failedResult.status = 'error';
        failedResult.error = error.message;
        failedResult.endTime = new Date();
        failedResult.processingTime = Date.now() - failedResult.startTime.getTime();
        this.processingJobs.set(jobId, failedResult);
      }
    });

    return { sessionId, jobId };
  }

  private async processEvidence(
    sessionId: string, 
    jobId: string, 
    evidenceData: EvidenceData, 
    request: ProcessingRequest
  ): Promise<void> {
    const result = this.processingJobs.get(jobId);
    if (!result) return;

    try {
      const totalSteps = request.steps.length;
      let currentStepIndex = 0;
      const allResults: any = {};

      for (const step of request.steps) {
        result.step = step;
        result.stepProgress = 0;
        result.progress = (currentStepIndex / totalSteps) * 100;
        this.processingJobs.set(jobId, result);
        
        let stepResult: any;
        
        switch (step) {
          case 'ocr':
            stepResult = await this.performOCR(evidenceData, request.options);
            break;
          case 'embedding':
            stepResult = await this.generateEmbedding(evidenceData, request.options);
            break;
          case 'analysis':
            stepResult = await this.performAnalysis(evidenceData, request.options);
            break;
          case 'classification':
            stepResult = await this.performClassification(evidenceData, request.options);
            break;
          case 'entity_extraction':
            stepResult = await this.extractEntities(evidenceData, request.options);
            break;
          case 'similarity':
            stepResult = await this.findSimilarEvidence(evidenceData, request.options);
            break;
          case 'indexing':
            stepResult = await this.indexEvidence(evidenceData, request.options);
            break;
          default:
            throw new Error(`Unknown processing step: ${step}`);
        }

        allResults[step] = stepResult;
        result.stepProgress = 100;
        currentStepIndex++;
        
        result.progress = (currentStepIndex / totalSteps) * 100;
        this.processingJobs.set(jobId, result);
      }

      // Complete processing
      result.status = 'completed';
      result.progress = 100;
      result.results = allResults;
      result.endTime = new Date();
      result.processingTime = Date.now() - result.startTime.getTime();
      
      await this.updateEvidenceWithResults(evidenceData.id, allResults);
      this.processingJobs.set(jobId, result);

    } catch (error) {
      result.status = 'error';
      result.error = error.message;
      result.endTime = new Date();
      result.processingTime = Date.now() - result.startTime.getTime();
      this.processingJobs.set(jobId, result);
    }
  }

  private async performOCR(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    if (!evidenceData.fileUrl || !evidenceData.mimeType?.includes('image')) {
      return { text: evidenceData.description || '', confidence: 1.0 };
    }

    try {
      // Use WebAssembly + WebGPU middleware for OCR
      const response = await fetch('http://localhost:8090/wasm/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          module: 'ocr_processor',
          function: 'extract_text',
          data: {
            fileUrl: evidenceData.fileUrl,
            mimeType: evidenceData.mimeType,
            options: { language: 'eng', dpi: 300, preprocess: true }
          },
          gpu_accelerated: options.useGPUAcceleration
        })
      });

      if (!response.ok) {
        throw new Error(`OCR processing failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('OCR failed, using fallback:', error);
      return { 
        text: evidenceData.description || evidenceData.title || '', 
        confidence: 0.5,
        error: error.message 
      };
    }
  }

  private async generateEmbedding(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    try {
      const textToEmbed = `${evidenceData.title} ${evidenceData.description || ''} ${(evidenceData.tags || []).join(' ')}`;
      
      if (options.useGPUAcceleration) {
        // Use enhanced RAG service for embeddings
        const response = await fetch('http://localhost:8094/api/embeddings/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: textToEmbed,
            model: 'nomic-embed-text',
            options: { normalize: true, dimensions: 768 }
          })
        });

        if (response.ok) {
          const result = await response.json();
          return {
            embedding: result.embedding,
            model: 'nomic-embed-text-gpu',
            dimensions: result.embedding?.length || 0
          };
        }
      }
      
      // Fallback to Ollama
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'nomic-embed-text', prompt: textToEmbed })
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      return {
        embedding: result.embedding,
        model: 'nomic-embed-text',
        dimensions: result.embedding?.length || 0
      };
    } catch (error) {
      console.error('Embedding generation failed:', error);
      return { embedding: [], model: 'error', dimensions: 0, error: error.message };
    }
  }

  private async performAnalysis(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    try {
      const analysisPayload = {
        evidence: {
          title: evidenceData.title,
          description: evidenceData.description,
          evidenceType: evidenceData.evidenceType,
          tags: evidenceData.tags,
          fileType: evidenceData.fileType,
          location: evidenceData.location
        },
        options: {
          extractEntities: true,
          generateSummary: true,
          findRelationships: true,
          calculateConfidence: true,
          useGPU: options.useGPUAcceleration
        }
      };

      if (options.useGPUAcceleration) {
        // Use WebAssembly + WebGPU middleware for analysis
        const response = await fetch('http://localhost:8090/wasm/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            module: 'evidence_analyzer',
            function: 'analyze_evidence',
            data: analysisPayload,
            gpu_accelerated: true
          })
        });

        if (response.ok) {
          return await response.json();
        }
      }

      // Fallback to enhanced RAG service
      const response = await fetch('http://localhost:8094/api/evidence/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisPayload)
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis failed:', error);
      return {
        entities: [],
        summary: `Analysis failed: ${error.message}`,
        confidence: 0.3,
        relationships: [],
        keywords: evidenceData.tags || [],
        error: error.message
      };
    }
  }

  private async performClassification(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    try {
      const classificationPrompt = `Classify this evidence into detailed categories:
      
Title: ${evidenceData.title}
Type: ${evidenceData.evidenceType}
Description: ${evidenceData.description || 'N/A'}

Provide classification for:
1. Legal significance (high/medium/low)
2. Evidential weight (direct/circumstantial/corroborative)
3. Admissibility likelihood (admissible/questionable/inadmissible)
4. Priority level (critical/important/routine)
5. Content categories (factual, testimonial, documentary, physical, digital)

Return JSON format with detailed reasoning.`;

      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: classificationPrompt,
          options: { temperature: 0.1, top_p: 0.9 },
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Classification failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      try {
        return JSON.parse(result.response);
      } catch {
        return {
          classification: result.response.substring(0, 200),
          significance: 'medium',
          weight: 'circumstantial',
          admissibility: 'questionable',
          priority: 'routine',
          categories: [evidenceData.evidenceType]
        };
      }
    } catch (error) {
      console.error('Classification failed:', error);
      return {
        classification: `Classification failed: ${error.message}`,
        significance: 'unknown',
        weight: 'unknown',
        admissibility: 'unknown',
        priority: 'unknown',
        categories: [evidenceData.evidenceType],
        error: error.message
      };
    }
  }

  private async extractEntities(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    try {
      const textContent = `${evidenceData.title} ${evidenceData.description || ''}`;
      
      // Use WebAssembly + WebGPU middleware for NER
      const response = await fetch('http://localhost:8090/wasm/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          module: 'ner_processor',
          function: 'extract_entities',
          data: {
            text: textContent,
            options: {
              extractPeople: true,
              extractLocations: true,
              extractOrganizations: true,
              extractDates: true,
              extractLegalTerms: true,
              extractVehicles: true,
              extractWeapons: true
            }
          },
          gpu_accelerated: options.useGPUAcceleration
        })
      });

      if (response.ok) {
        return await response.json();
      }

      // Fallback entity extraction using regex patterns
      const entities = [];
      const patterns = {
        person: /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g,
        date: /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b/g,
        money: /\$[\d,]+(?:\.\d{2})?\b/g,
        location: /\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)\b/gi
      };

      for (const [type, pattern] of Object.entries(patterns)) {
        const matches = textContent.match(pattern) || [];
        for (const match of matches) {
          entities.push({
            text: match,
            type: type as any,
            confidence: 0.6,
            position: { start: 0, end: match.length }
          });
        }
      }

      return { entities, method: 'regex_fallback' };
    } catch (error) {
      console.error('Entity extraction failed:', error);
      return { entities: [], method: 'error', error: error.message };
    }
  }

  private async findSimilarEvidence(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    try {
      const textToSearch = `${evidenceData.title} ${evidenceData.description || ''}`;
      
      // Use semantic search to find similar evidence
      const searchResponse = await fetch('/api/evidence', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'semantic_search',
          evidenceId: evidenceData.id,
          options: { query: textToSearch, limit: 10 }
        })
      });

      if (searchResponse.ok) {
        const searchResult = await searchResponse.json();
        return {
          similarEvidence: searchResult.results,
          totalFound: searchResult.total,
          query: textToSearch
        };
      }

      return { similarEvidence: [], totalFound: 0, query: textToSearch };
    } catch (error) {
      console.error('Similarity search failed:', error);
      return { similarEvidence: [], totalFound: 0, error: error.message };
    }
  }

  private async indexEvidence(evidenceData: EvidenceData, options: ProcessingOptions): Promise<any> {
    try {
      // Index in vector database (Qdrant)
      const indexingPayload = {
        evidence_id: evidenceData.id,
        case_id: evidenceData.caseId,
        content: `${evidenceData.title} ${evidenceData.description || ''}`,
        metadata: {
          evidence_type: evidenceData.evidenceType,
          file_type: evidenceData.fileType,
          tags: evidenceData.tags,
          uploaded_at: evidenceData.uploadedAt
        }
      };

      const response = await fetch('http://localhost:8094/api/vector/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(indexingPayload)
      });

      if (response.ok) {
        const result = await response.json();
        return {
          indexed: true,
          vectorId: result.vector_id,
          collection: result.collection || 'evidence',
          similarity_threshold: 0.7
        };
      }

      throw new Error(`Indexing failed: ${response.statusText}`);
    } catch (error) {
      console.error('Indexing failed:', error);
      return {
        indexed: false,
        vectorId: null,
        collection: null,
        error: error.message
      };
    }
  }

  private async updateEvidenceWithResults(evidenceId: string, results: any): Promise<void> {
    try {
      const updateData: any = { updatedAt: new Date() };

      if (results.analysis) {
        updateData.aiAnalysis = results.analysis;
        updateData.aiSummary = results.analysis.summary;
        updateData.aiTags = results.analysis.keywords;
      }

      if (results.embedding?.embedding) {
        updateData.embedding = JSON.stringify(results.embedding.embedding);
      }

      if (results.classification) {
        updateData.labAnalysis = {
          ...updateData.labAnalysis,
          classification: results.classification
        };
      }

      await db
        .update(evidence)
        .set(updateData)
        .where(eq(evidence.id, evidenceId));
    } catch (error) {
      console.error('Failed to update evidence with results:', error);
    }
  }

  getProcessingStatus(jobId: string): ProcessingResult | null {
    return this.processingJobs.get(jobId) || null;
  }

  cancelProcessing(jobId: string): boolean {
    const result = this.processingJobs.get(jobId);
    if (result && result.status === 'processing') {
      result.status = 'error';
      result.error = 'Cancelled by user';
      result.endTime = new Date();
      result.processingTime = Date.now() - result.startTime.getTime();
      this.processingJobs.set(jobId, result);
      return true;
    }
    return false;
  }
}

const processingService = EvidenceProcessingService.getInstance();

// Enhanced POST endpoint for comprehensive processing
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    
    // Check if this is the old simple processing call
    if (!body.evidenceId && !body.steps) {
      // Fallback to original simple processing
      const status = await processNextJob();
      return json({ processed: !!status, status });
    }

    const { evidenceId, steps, options = {} } = body;

    if (!evidenceId || !steps || !Array.isArray(steps)) {
      return json({ 
        error: 'evidenceId and steps array are required' 
      }, { status: 400 });
    }

    const processingRequest: ProcessingRequest = {
      evidenceId,
      steps: steps as ProcessingStep[],
      options: {
        useGPUAcceleration: true,
        priority: 'normal',
        notify: false,
        saveIntermediateResults: true,
        overrideExisting: false,
        ...options
      }
    };

    const { sessionId, jobId } = await processingService.startProcessing(processingRequest);

    return json({ 
      sessionId, 
      jobId, 
      status: 'started',
      steps: processingRequest.steps,
      options: processingRequest.options
    });
  } catch (error) {
    console.error('Evidence processing request failed:', error);
    return json({ 
      error: error.message || 'Processing request failed' 
    }, { status: 500 });
  }
};

// Enhanced GET endpoint with detailed status
export const GET: RequestHandler = async ({ url }) => {
  try {
    const jobId = url.searchParams.get('jobId');
    
    if (!jobId) {
      return json({ error: 'jobId is required' }, { status: 400 });
    }

    const status = processingService.getProcessingStatus(jobId);
    
    if (!status) {
      // Try original status check as fallback
      const originalStatus = getJobStatus(jobId);
      if (!originalStatus) {
        return json({ error: 'Job not found' }, { status: 404 });
      }
      return json(originalStatus);
    }

    return json(status);
  } catch (error) {
    console.error('Failed to get processing status:', error);
    return json({ error: 'Failed to get status' }, { status: 500 });
  }
};

// Enhanced DELETE endpoint with cancellation
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const jobId = url.searchParams.get('jobId');
    
    if (!jobId) {
      return json({ error: 'jobId is required' }, { status: 400 });
    }

    const cancelled = processingService.cancelProcessing(jobId);
    
    return json({ 
      cancelled, 
      jobId,
      message: cancelled ? 'Processing cancelled' : 'Job not found or not cancellable'
    });
  } catch (error) {
    console.error('Failed to cancel processing:', error);
    return json({ error: 'Failed to cancel processing' }, { status: 500 });
  }
};