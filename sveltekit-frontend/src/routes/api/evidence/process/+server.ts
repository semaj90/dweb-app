import type { RequestHandler } from './$types.js';

// Simplified and type-safe evidence processing endpoint with an in-memory processing service
// POST: start processing -> returns sessionId and jobId
// GET:  ?jobId=... -> returns processing status
// DELETE: ?jobId=... -> cancels job

import { db } from '$lib/server/db';
import { evidence } from '$lib/server/db/schema-postgres';
import { eq } from 'drizzle-orm';
import crypto from "crypto";
import { URL } from "url";

// Type definitions for evidence processing
type StepName = 'ocr' | 'embedding' | 'analysis' | 'classification' | 'entity_extraction' | 'similarity' | 'indexing';

export interface ProcessingRequest {
  evidenceId: string;
  steps: StepName[];
  options?: ProcessingOptions;
}

export interface ProcessingResult {
  sessionId?: string;
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'error';
  progress: number;
  steps?: StepName[];
  currentStep?: StepName | null;
  stepProgress?: number;
  results?: Record<string, any> | null;
  error?: string | null;
  startTime?: Date;
  endTime?: Date;
  processingTime?: number;
  gpuAccelerated?: boolean;
}

export interface ProcessingOptions {
  useGPUAcceleration?: boolean;
  priority?: 'low' | 'medium' | 'high' | 'normal';
  notify?: boolean;
  saveIntermediateResults?: boolean;
  overrideExisting?: boolean;
}

export interface EvidenceData {
  id: string;
  caseId?: string;
  title?: string;
  description?: string | null;
  tags?: string[];
  fileUrl?: string | null;
  mimeType?: string | null;
  evidenceType?: string | null;
  fileType?: string | null;
  location?: string | null;
  uploadedAt?: string | Date | null;
}

class EvidenceProcessingService {
  private static instance: EvidenceProcessingService | null = null;
  private processingJobs: Map<string, ProcessingResult> = new Map();

  private constructor() { }

  static getInstance(): EvidenceProcessingService {
    if (!EvidenceProcessingService.instance) {
      EvidenceProcessingService.instance = new EvidenceProcessingService();
    }
    return EvidenceProcessingService.instance;
  }

  async startProcessing(request: ProcessingRequest): Promise<{ sessionId: string; jobId: string }> {
    const sessionId = crypto.randomUUID();
    const jobId = crypto.randomUUID();

    const processingResult: ProcessingResult = {
      sessionId,
      jobId,
      status: 'processing',
      progress: 0,
      steps: request.steps,
      currentStep: request.steps.length > 0 ? request.steps[0] : null,
      stepProgress: 0,
      results: {},
      error: null,
      startTime: new Date(),
      processingTime: 0,
      gpuAccelerated: !!request.options?.useGPUAcceleration
    };

    this.processingJobs.set(jobId, processingResult);

    // Background processing (non-blocking)
    this.processEvidence(sessionId, jobId, request).catch((err) => {
      console.error('Processing background error:', err);
      const r = this.processingJobs.get(jobId);
      if (r) {
        r.status = 'error';
        r.error = err?.message ?? String(err);
        r.endTime = new Date();
        r.processingTime = r.startTime ? Date.now() - r.startTime.getTime() : 0;
        this.processingJobs.set(jobId, r);
      }
    });

    return { sessionId, jobId };
  }

  private async processEvidence(sessionId: string, jobId: string, request: ProcessingRequest): Promise<void> {
    const result = this.processingJobs.get(jobId);
    if (!result) return;

    try {
      // Attempt to load evidence (best-effort; if not found, proceed with minimal metadata)
      let evidenceData: EvidenceData | null = null;
      try {
        const rows = await db.select().from(evidence).where(eq(evidence.id, request.evidenceId)).limit(1);
        if (rows && rows.length > 0) {
          evidenceData = rows[0] as EvidenceData;
        }
      } catch (e: any) {
        console.warn('Failed to load evidence from DB, continuing with provided id.');
      }
      if (!evidenceData) {
        evidenceData = { id: request.evidenceId, title: undefined, description: undefined };
      }

      const totalSteps = Math.max(1, request.steps.length);
      const results: Record<string, any> = {};
      for (let i = 0; i < request.steps.length; i++) {
        const step = request.steps[i];
        result.currentStep = step;
        result.stepProgress = 0;
        result.progress = Math.floor((i / totalSteps) * 100);
        this.processingJobs.set(jobId, result);

        // simple simulated work per step
        let stepResult: any = null;
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
            stepResult = { error: 'unknown_step' };
        }

        results[step] = stepResult;
        result.stepProgress = 100;
        result.progress = Math.floor(((i + 1) / totalSteps) * 100);
        this.processingJobs.set(jobId, result);
      }

      result.status = 'completed';
      result.results = results;
      result.endTime = new Date();
      result.processingTime = result.startTime ? Date.now() - result.startTime.getTime() : 0;
      this.processingJobs.set(jobId, result);

      // Best-effort persist results
      await this.updateEvidenceWithResults(request.evidenceId, results).catch((e: any) => {
        console.warn('Failed to persist results:', e);
      });
    } catch (err: any) {
      result.status = 'error';
      result.error = err?.message ?? String(err);
      result.endTime = new Date();
      result.processingTime = result.startTime ? Date.now() - result.startTime.getTime() : 0;
      this.processingJobs.set(jobId, result);
    }
  }

  private async performOCR(evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    // stubbed OCR result
    await new Promise((r) => setTimeout(r, 100));
    return { text: evidenceData.description || evidenceData.title || '', confidence: 0.9 };
  }

  private async generateEmbedding(evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    await new Promise((r) => setTimeout(r, 100));
    const text = `${evidenceData.title || ''} ${evidenceData.description || ''}`.trim();
    const embedding = Array.from({ length: 8 }, (_, i) => (text.length + i) % 10 / 10);
    return { embedding, model: 'stub-embed', dimensions: embedding.length };
  }

  private async performAnalysis(evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    await new Promise((r) => setTimeout(r, 120));
    return {
      summary: (evidenceData.description || evidenceData.title || '').slice(0, 200),
      keywords: evidenceData.tags || [],
      confidence: 0.8
    };
  }

  private async performClassification(_evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    await new Promise((r) => setTimeout(r, 80));
    return {
      significance: 'medium',
      weight: 'circumstantial',
      admissibility: 'questionable',
      priority: 'routine',
      categories: []
    };
  }

  private async extractEntities(evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    await new Promise((r) => setTimeout(r, 60));
    const text = `${evidenceData.title || ''} ${evidenceData.description || ''}`;
    return { entities: text ? [{ text: text.slice(0, 30), type: 'text', confidence: 0.5 }] : [], method: 'stub' };
  }

  private async findSimilarEvidence(_evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    await new Promise((r) => setTimeout(r, 60));
    return { similarEvidence: [], totalFound: 0 };
  }

  private async indexEvidence(_evidenceData: EvidenceData, _options?: ProcessingOptions): Promise<any> {
    await new Promise((r) => setTimeout(r, 60));
    return { indexed: true, vectorId: crypto.randomUUID(), collection: 'evidence' };
  }

  private async updateEvidenceWithResults(evidenceId: string, results: any): Promise<void> {
    const updateData: any = { updatedAt: new Date() };
    if (results.analysis) {
      updateData.aiAnalysis = results.analysis;
      updateData.aiSummary = results.analysis.summary;
      updateData.aiTags = results.analysis.keywords;
    }
    if (results.embedding?.embedding) {
      updateData.embedding = JSON.stringify(results.embedding.embedding);
    }
    try {
      await db.update(evidence).set(updateData).where(eq(evidence.id, evidenceId));
    } catch (e: any) {
      // ignore persistence errors (best-effort)
      console.warn('DB update failed:', e);
    }
  }

  getProcessingStatus(jobId: string): ProcessingResult | null {
    return this.processingJobs.get(jobId) ?? null;
  }

  cancelProcessing(jobId: string): boolean {
    const r = this.processingJobs.get(jobId);
    if (r && r.status === 'processing') {
      r.status = 'error';
      r.error = 'Cancelled by user';
      r.endTime = new Date();
      r.processingTime = r.startTime ? Date.now() - r.startTime.getTime() : 0;
      this.processingJobs.set(jobId, r);
      return true;
    }
    return false;
  }
}

const processingService = EvidenceProcessingService.getInstance();

// POST endpoint: start processing
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const { evidenceId, steps, options = {} } = body ?? {};

    if (!evidenceId || !steps || !Array.isArray(steps)) {
      return json({ error: 'evidenceId and steps array are required' }, { status: 400 });
    }

    const processingRequest: ProcessingRequest = {
      evidenceId,
      steps: steps as StepName[],
      options: {
        useGPUAcceleration: !!options.useGPUAcceleration,
        priority: options.priority ?? 'normal',
        notify: !!options.notify,
        saveIntermediateResults: !!options.saveIntermediateResults,
        overrideExisting: !!options.overrideExisting
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
  } catch (err: any) {
    console.error('POST processing error:', err);
    return json({ error: (err as any)?.message ?? 'Processing request failed' }, { status: 500 });
  }
};

// GET endpoint: get status
export const GET: RequestHandler = async ({ url }) => {
  try {
    const jobId = url.searchParams.get('jobId');
    if (!jobId) {
      return json({ error: 'jobId is required' }, { status: 400 });
    }

    const status = processingService.getProcessingStatus(jobId);
    if (!status) {
      return json({ error: 'Job not found' }, { status: 404 });
    }
    return json(status);
  } catch (err: any) {
    console.error('GET status error:', err);
    return json({ error: 'Failed to get status' }, { status: 500 });
  }
};

// DELETE endpoint: cancel job
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
  } catch (err: any) {
    console.error('DELETE error:', err);
    return json({ error: 'Failed to cancel processing' }, { status: 500 });
  }
};