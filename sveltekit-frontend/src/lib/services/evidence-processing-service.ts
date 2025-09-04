// Evidence Processing & Caching Service
// Extracted from AiAssistant.svelte for testability and reuse.
// Provides summary synthesis with TensorFlowSynthesizer + multi-layer cache + RabbitMQ queuing fallback.

import { TensorFlowSynthesizer, type SynthesizedAnalysis } from '$lib/middleware/tfjs-synthesizer';
import { MultiLayerCache } from '$lib/services/multi-layer-cache';
import { rabbitMQQueue, type DocumentMessage } from '$lib/services/rabbitmq-queue-service';

export interface EvidenceProcessingResult {
  summary: string;
  sources: any[];
  confidence: number;
  processingTime?: number;
  aiInsights?: SynthesizedAnalysis['synthesizedInsights'];
  processingMode?: 'direct' | 'queued';
  jobIds?: string[];
  estimatedCompletionTime?: number; // seconds
}

export interface EvidenceProcessingServiceOptions {
  synthesizer?: TensorFlowSynthesizer | null;
  cache?: MultiLayerCache | null;
  queue?: typeof rabbitMQQueue;
  userId?: string | null;
}

export class EvidenceProcessingService {
  private synthesizer: TensorFlowSynthesizer | null;
  private cache: MultiLayerCache | null;
  private queue: typeof rabbitMQQueue | null;
  private userId?: string | null;

  constructor(opts: EvidenceProcessingServiceOptions) {
    this.synthesizer = opts.synthesizer || null;
    this.cache = opts.cache || null;
    this.queue = opts.queue || null;
    this.userId = opts.userId;
  }

  updateRuntimeDeps(opts: Partial<EvidenceProcessingServiceOptions>) {
    if (opts.synthesizer !== undefined) this.synthesizer = opts.synthesizer;
    if (opts.cache !== undefined) this.cache = opts.cache;
    if (opts.queue !== undefined) this.queue = opts.queue;
    if (opts.userId !== undefined) this.userId = opts.userId;
  }

  async processWithQueuing(evidence: any[], caseId: string, userId?: string | null): Promise<EvidenceProcessingResult> {
    // Small batch: direct processing
    if (evidence.length <= 3) {
      return this.fetchSummary({ caseId, evidence, userId: userId || this.userId || undefined });
    }

    // Large batch: try queue
    if (this.queue?.connected) {
      try {
        const jobIds = await this.queueDocuments(evidence, caseId, userId);
        return {
          summary: `Processing ${evidence.length} evidence items in background. Job IDs: ${jobIds.slice(0, 3).join(', ')}${jobIds.length > 3 ? '...' : ''}`,
          sources: [],
          confidence: 0,
          processingMode: 'queued',
          jobIds,
          estimatedCompletionTime: evidence.length * 30
        };
      } catch (e) {
        console.warn('Queue failed, falling back to direct processing', e);
      }
    }

    return this.fetchSummary({ caseId, evidence, userId: userId || this.userId || undefined });
  }

  async fetchSummary(payload: { caseId: string; evidence: any[]; userId?: string }): Promise<EvidenceProcessingResult> {
    const evidenceText = payload.evidence
      .map(item => `${item.title || 'Evidence'}: ${item.content || item.description || ''}`)
      .join('\n\n');

    // Attempt synthesizer + cache first
    if (this.synthesizer && this.cache) {
      try {
        const cacheKey = `evidence_analysis_${payload.caseId}`;
        const cachedStructured = await this.cache.getSynthesizedAnalysis(cacheKey);
        const cachedSummary = await this.cache.getSummary(cacheKey);
        if (cachedSummary || cachedStructured) {
          return {
            summary: cachedSummary || cachedStructured?.enhancedResponse.primaryResponse || 'Cached summary',
            sources: cachedStructured?.enhancedResponse.sources || [],
            confidence: cachedStructured ? (cachedStructured.qualityMetrics.overallQuality || 0.85) : 0.8,
            aiInsights: cachedStructured?.synthesizedInsights,
            processingMode: 'direct'
          };
        }

        const analysisResult = await this.synthesizer.synthesizeAnalysis(
          evidenceText,
            `Analyze and summarize the following legal evidence for case ${payload.caseId}`,
          {
            caseId: payload.caseId,
            userId: payload.userId,
            evidenceCount: payload.evidence.length,
            requestType: 'evidence_summary'
          }
        );

        await this.cache.setSynthesizedAnalysis(cacheKey, analysisResult);
        await this.cache.setSummary(cacheKey, analysisResult.enhancedResponse.primaryResponse);

        return {
          summary: analysisResult.enhancedResponse.primaryResponse,
            sources: analysisResult.enhancedResponse.sources || [],
          confidence: analysisResult.qualityMetrics.overallQuality || 0.9,
          processingTime: analysisResult.processingPipeline.totalProcessingTime,
          aiInsights: analysisResult.synthesizedInsights,
          processingMode: 'direct'
        };
      } catch (e) {
        console.warn('Synthesizer failed, attempting backend fallback', e);
      }
    }

    // Backend fallback chain
    let ragResult: any = null;
    try {
      const healthRes = await fetch('http://localhost:8094/health');
      if (!healthRes.ok) throw new Error('Enhanced RAG health failed');
    } catch {
      // continue to fallback logic
    }

    // Primary proxy endpoint
    const primary = await fetch('/api/ai/analyze-evidence', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        evidenceId: payload.caseId + '-batch',
        content: JSON.stringify(payload.evidence),
        forceReanalyze: true
      })
    });

    if (primary.ok) {
      ragResult = await primary.json();
    } else {
      const fallback = await fetch('/api/ai/process-enhanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidence: payload.evidence,
          options: {
            analysisType: 'summary',
            model: 'gemma3-legal:latest',
            caseId: payload.caseId,
            userId: payload.userId
          }
        })
      });
      if (!fallback.ok) throw new Error('All AI services unavailable');
      ragResult = await fallback.json();
    }

    return {
      summary: ragResult.summary || ragResult.analysis || ragResult.result || 'Analysis completed',
      sources: ragResult.sources || [],
      confidence: ragResult.confidence || 0.85,
      processingMode: 'direct'
    };
  }

  async queueDocuments(evidence: any[], caseId: string, userId?: string | null): Promise<string[]> {
    if (!this.queue?.connected) throw new Error('Queue not connected');
    const jobIds: string[] = [];
    for (const item of evidence) {
      const documentMessage: DocumentMessage = {
        document_id: `doc_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`,
        case_id: caseId,
        source_location: item.url || item.path || 'memory://evidence',
        metadata: {
          title: item.title || item.name || 'Evidence Item',
          file_type: item.type || 'text',
          upload_date: new Date().toISOString(),
          user_id: userId || this.userId || undefined
        }
      };
      const jobId = await this.queue.publishDocument(documentMessage);
      jobIds.push(jobId);
    }
    return jobIds;
  }

  async fetchRelatedEvidence(currentEmbedding: number[], k = 5) {
    try {
      const res = await fetch('/api/ai/evidence-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ embedding: currentEmbedding, limit: k })
      });
      if (!res.ok) throw new Error('related evidence search failed');
      const data = await res.json();
      return data.results?.rows || data.results || [];
    } catch (e) {
      console.warn('Related evidence fetch error:', e);
      return [];
    }
  }

  async generateEmbeddingFromSummary(text: string): Promise<number[] | null> {
    try {
      const res = await fetch('/api/ai/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!res.ok) return null;
      const data = await res.json();
      return Array.isArray(data.embedding) ? data.embedding : null;
    } catch (e) {
      console.warn('Embedding generation failed:', e);
      return null;
    }
  }
}
