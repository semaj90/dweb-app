/**
 * GPU Integration Bridge - Connects Your Existing Architecture
 * Links GPU caching system with JSONB schema, Qdrant sync, and WASM workers
 * Provides unified interface for legal AI GPU operations
 */

import { legalAIGPUQueue } from './gpu-job-queue';
import { legalAIResultCache } from './advanced-result-cache';
import { legalAIKernelManager } from './gpu-kernel-manager';
import { postgresqlQdrantSync } from './postgresql-qdrant-sync';
import { db } from '$lib/server/db/index';
import { aiSummarizedDocuments, documentEmbeddings } from '$lib/db/schema-jsonb';
import { eq, sql } from 'drizzle-orm';
import type { SummaryData, DocumentMetadata } from '$lib/db/schema-jsonb';

export interface GPUProcessingResult {
  success: boolean;
  data?: any;
  metadata: {
    processingTimeMs: number;
    gpuUtilized: boolean;
    cacheHit: boolean;
    embeddingModel: string;
    qdrantSynced: boolean;
  };
}

export class GPUIntegrationBridge {
  
  /**
   * Process legal document with full GPU optimization + JSONB storage + Qdrant sync
   */
  async processLegalDocumentComplete(
    documentText: string,
    documentType: 'contract' | 'brief' | 'case_study' | 'memo' | 'agreement' | 'policy' | 'other',
    metadata: Partial<DocumentMetadata> = {},
    userId?: string
  ): Promise<GPUProcessingResult & { documentId: string }> {
    const startTime = Date.now();
    let documentId: string;
    let cacheHit = false;
    let qdrantSynced = false;

    try {
      // 1. Generate embeddings using GPU queue with caching
      const embeddingResult = await legalAIGPUQueue.embedDocument(documentText, {
        priority: 'high',
        userId
      });
      
      // Check if result was from cache
      cacheHit = embeddingResult.fromCache || false;

      // 2. Extract legal entities with GPU acceleration
      const entitiesResult = await legalAIGPUQueue.extractEntities(documentText, {
        priority: 'medium',
        userId
      });

      // 3. Classify document using optimized GPU kernel
      const classificationResult = await legalAIGPUQueue.classifyDocument(documentText, {
        priority: 'medium', 
        userId
      });

      // 4. Generate legal summary with GPU acceleration
      const summaryResult = await legalAIKernelManager.summarizeLegalDocument(
        documentText, 
        500 // max length
      );

      // 5. Prepare JSONB data structures
      const summaryData: SummaryData = {
        executive_summary: summaryResult,
        key_findings: this.extractKeyFindings(documentText, entitiesResult),
        legal_issues: this.convertToLegalIssues(entitiesResult),
        recommendations: this.generateRecommendations(classificationResult),
        risk_assessment: this.assessRisk(entitiesResult, classificationResult),
        confidence_score: classificationResult.confidence,
        processing_metrics: {
          chunks_processed: Math.ceil(documentText.length / 1000),
          gpu_memory_used_mb: 1024, // Estimated
          model_temperature: 0.2,
          inference_time_ms: Date.now() - startTime,
          queue_time_ms: 0 // Would be tracked by queue
        }
      };

      const docMetadata: DocumentMetadata = {
        source: 'api',
        file_size_bytes: Buffer.byteLength(documentText, 'utf8'),
        pages: Math.ceil(documentText.length / 2000), // Estimated
        language: 'en',
        ocr_applied: false,
        original_format: 'text',
        tags: entitiesResult.map((entity: any) => entity.text),
        ...metadata
      };

      // 6. Store in PostgreSQL with JSONB optimization
      const [insertedDoc] = await db.insert(aiSummarizedDocuments).values({
        documentName: `Legal Document - ${documentType}`,
        documentType,
        originalText: documentText,
        metadata: docMetadata,
        summaryData,
        analysis: {
          entities: entitiesResult,
          classification: classificationResult,
          embedding_model: 'legal-embedding'
        },
        entities: entitiesResult,
        status: 'completed',
        processingTimeMs: Date.now() - startTime,
        modelUsed: 'gpu-accelerated-legal-pipeline',
        gpuUtilized: true,
        createdBy: userId
      }).returning({ id: aiSummarizedDocuments.id });

      documentId = insertedDoc.id;

      // 7. Store embeddings with vector optimization
      await db.insert(documentEmbeddings).values({
        documentId,
        chunkIndex: 0,
        chunkText: documentText.substring(0, 1000), // First chunk
        embedding: embeddingResult,
        metadata: {
          model: 'legal-embedding',
          gpu_optimized: true,
          cache_hit: cacheHit
        },
        modelName: 'legal-embedding'
      });

      // 8. Sync to Qdrant for vector search optimization
      try {
        await postgresqlQdrantSync.syncDocumentById(documentId);
        qdrantSynced = true;
      } catch (qdrantError) {
        console.warn('Qdrant sync failed:', qdrantError);
      }

      const totalProcessingTime = Date.now() - startTime;

      return {
        success: true,
        documentId,
        data: {
          summary: summaryResult,
          entities: entitiesResult,
          classification: classificationResult,
          embedding: embeddingResult,
          summaryData
        },
        metadata: {
          processingTimeMs: totalProcessingTime,
          gpuUtilized: true,
          cacheHit,
          embeddingModel: 'legal-embedding',
          qdrantSynced
        }
      };

    } catch (error: any) {
      return {
        success: false,
        documentId: '',
        metadata: {
          processingTimeMs: Date.now() - startTime,
          gpuUtilized: false,
          cacheHit: false,
          embeddingModel: 'none',
          qdrantSynced: false
        }
      };
    }
  }

  /**
   * Enhanced RAG query with GPU optimization + Qdrant vector search
   */
  async performEnhancedRAGQuery(
    query: string,
    options: {
      userId?: string;
      caseId?: string;
      documentTypes?: string[];
      maxResults?: number;
      scoreThreshold?: number;
    } = {}
  ): Promise<GPUProcessingResult & { 
    answer: string;
    sources: Array<{
      documentId: string;
      content: string;
      score: number;
      metadata: any;
    }>;
  }> {
    const startTime = Date.now();
    
    try {
      // 1. Generate query embedding with GPU acceleration
      const queryEmbedding = await legalAIGPUQueue.embedDocument(query, {
        priority: 'critical',
        userId: options.userId
      });

      // 2. Perform optimized vector search via Qdrant
      const searchResults = await postgresqlQdrantSync.searchForWASMInference(
        queryEmbedding,
        options.maxResults || 5,
        options.scoreThreshold || 0.7,
        {
          ...(options.caseId && { caseId: options.caseId }),
          ...(options.documentTypes && { documentType: options.documentTypes })
        }
      );

      // 3. Generate RAG response using GPU-accelerated summarization
      const ragContext = searchResults.map(r => r.content).join('\n\n');
      const ragAnswer = await legalAIKernelManager.summarizeLegalDocument(
        `Query: ${query}\n\nContext:\n${ragContext}`,
        300
      );

      const processingTime = Date.now() - startTime;

      return {
        success: true,
        data: {
          answer: ragAnswer,
          sources: searchResults.map(result => ({
            documentId: result.metadata.documentId || '',
            content: result.content,
            score: result.score,
            metadata: result.metadata
          }))
        },
        answer: ragAnswer,
        sources: searchResults.map(result => ({
          documentId: result.metadata.documentId || '',
          content: result.content,
          score: result.score,
          metadata: result.metadata
        })),
        metadata: {
          processingTimeMs: processingTime,
          gpuUtilized: true,
          cacheHit: false, // Would be determined by individual operations
          embeddingModel: 'legal-embedding',
          qdrantSynced: true
        }
      };

    } catch (error: any) {
      return {
        success: false,
        answer: `Error processing query: ${error instanceof Error ? error.message : 'Unknown error'}`,
        sources: [],
        metadata: {
          processingTimeMs: Date.now() - startTime,
          gpuUtilized: false,
          cacheHit: false,
          embeddingModel: 'none',
          qdrantSynced: false
        }
      };
    }
  }

  /**
   * Batch process legal documents with GPU optimization
   */
  async batchProcessLegalDocuments(
    documents: Array<{
      text: string;
      type: 'contract' | 'brief' | 'case_study' | 'memo' | 'agreement' | 'policy' | 'other';
      metadata?: Partial<DocumentMetadata>;
    }>,
    userId?: string
  ): Promise<{
    results: Array<GPUProcessingResult & { documentId: string }>;
    summary: {
      successful: number;
      failed: number;
      totalProcessingTime: number;
      averageProcessingTime: number;
      gpuUtilization: number;
    };
  }> {
    const startTime = Date.now();
    const results: Array<GPUProcessingResult & { documentId: string }> = [];

    // Process documents in batches of 5 for optimal GPU utilization
    const batchSize = 5;
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      
      const batchResults = await Promise.all(
        batch.map(doc => 
          this.processLegalDocumentComplete(doc.text, doc.type, doc.metadata, userId)
        )
      );
      
      results.push(...batchResults);
    }

    const totalProcessingTime = Date.now() - startTime;
    const successful = results.filter(r => r.success).length;
    const failed = results.length - successful;
    const gpuUtilized = results.filter(r => r.metadata.gpuUtilized).length;

    return {
      results,
      summary: {
        successful,
        failed,
        totalProcessingTime,
        averageProcessingTime: totalProcessingTime / documents.length,
        gpuUtilization: (gpuUtilized / documents.length) * 100
      }
    };
  }

  /**
   * Get comprehensive processing statistics
   */
  async getProcessingStats(): Promise<any> {
    const [
      cacheStats,
      queueStats,
      kernelStats,
      qdrantHealth,
      wasmStats
    ] = await Promise.all([
      Promise.resolve(legalAIResultCache.getStats()),
      Promise.resolve(legalAIGPUQueue.getQueueStats()),
      Promise.resolve(legalAIKernelManager.getKernelStats()),
      postgresqlQdrantSync.healthCheck(),
      Promise.resolve(postgresqlQdrantSync.getWASMStats())
    ]);

    // Get database statistics
    const [documentCount] = await db
      .select({ count: sql`count(*)` })
      .from(aiSummarizedDocuments);

    const [recentDocuments] = await db
      .select({ 
        avgProcessingTime: sql`avg(processing_time_ms)`,
        gpuUtilizedCount: sql`count(*) filter (where gpu_utilized = true)`
      })
      .from(aiSummarizedDocuments)
      .where(sql`created_at > now() - interval '24 hours'`);

    return {
      cache: cacheStats,
      queue: queueStats,
      kernels: kernelStats,
      vectorDB: {
        health: qdrantHealth,
        wasmOptimization: wasmStats
      },
      database: {
        totalDocuments: documentCount.count,
        recentDocuments: {
          averageProcessingTime: recentDocuments?.avgProcessingTime || 0,
          gpuUtilizationRate: recentDocuments?.gpuUtilizedCount || 0
        }
      },
      performance: {
        overallEfficiency: this.calculateOverallEfficiency(cacheStats, queueStats),
        recommendedOptimizations: this.generateOptimizationRecommendations(cacheStats, queueStats, kernelStats)
      }
    };
  }

  // Helper methods for data processing
  private extractKeyFindings(text: string, entities: any[]): string[] {
    return entities
      .filter((entity: any) => entity.label === 'KEY_POINT' || entity.confidence > 0.8)
      .slice(0, 5)
      .map((entity: any) => entity.text);
  }

  private convertToLegalIssues(entities: any[]): any[] {
    return entities
      .filter((entity: any) => entity.label === 'LEGAL_ISSUE' || entity.label === 'VIOLATION')
      .map((entity: any) => ({
        issue: entity.text,
        severity: entity.confidence > 0.9 ? 'HIGH' : entity.confidence > 0.7 ? 'MEDIUM' : 'LOW',
        description: `Legal issue identified: ${entity.text}`,
        precedents: []
      }));
  }

  private generateRecommendations(classification: any): any[] {
    const recommendations = [];
    
    if (classification.confidence < 0.7) {
      recommendations.push({
        action: 'Manual review required due to low confidence classification',
        priority: 'HIGH' as const,
        rationale: 'Classification confidence below threshold'
      });
    }

    if (classification.category === 'contract') {
      recommendations.push({
        action: 'Review contract terms and conditions',
        priority: 'MEDIUM' as const,
        rationale: 'Standard contract review protocol'
      });
    }

    return recommendations;
  }

  private assessRisk(entities: any[], classification: any): any {
    const riskFactors = entities
      .filter((entity: any) => entity.label === 'RISK_FACTOR')
      .map((entity: any) => ({
        factor: entity.text,
        impact: 'Medium',
        probability: entity.confidence
      }));

    let overallRisk = 'LOW' as const;
    if (classification.confidence < 0.5) overallRisk = 'HIGH';
    else if (classification.confidence < 0.7) overallRisk = 'MEDIUM';

    return {
      overall_risk: overallRisk,
      risk_factors: riskFactors,
      mitigation: ['Regular review', 'Legal consultation', 'Compliance monitoring']
    };
  }

  private calculateOverallEfficiency(cacheStats: any, queueStats: any): number {
    const cacheEfficiency = cacheStats.overall.hitRate;
    const queueEfficiency = queueStats.throughputPerSecond > 0 ? Math.min(queueStats.throughputPerSecond / 10, 1) : 0;
    
    return (cacheEfficiency + queueEfficiency) / 2;
  }

  private generateOptimizationRecommendations(cacheStats: any, queueStats: any, kernelStats: any): string[] {
    const recommendations: string[] = [];

    if (cacheStats.overall.hitRate < 0.7) {
      recommendations.push('Consider increasing cache size or TTL for better hit rates');
    }

    if (queueStats.pending > 50) {
      recommendations.push('Queue backlog is high - consider adding more GPU workers');
    }

    if (kernelStats.preloadedKernels < 3) {
      recommendations.push('Preload more GPU kernels for better performance');
    }

    return recommendations;
  }
}

// Export singleton instance
export const gpuIntegrationBridge = new GPUIntegrationBridge();