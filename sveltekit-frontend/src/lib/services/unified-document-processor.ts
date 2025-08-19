/**
 * Unified Document Processing Pipeline
 * Integrates OCR + LangExtract + Legal-BERT + Nomic Embeddings + AI Summarization
 * Uses gemma3-legal:latest for document processing
 */

import type { RequestHandler } from "@sveltejs/kit";
// Orphaned content: import {

import { legalNLP } from "./sentence-transformer.js";
// Orphaned content: import {
goMicroserviceMachine, goMicroserviceServices
import { createActor } from "xstate";
import { aiSummaryMachine, , export interface DocumentProcessingConfig {,   enableOCR: boolean;,   enableLegalBERT: boolean;,   enableEmbeddings: boolean;,   enableSummarization: boolean;,   enableMinIOStorage: boolean;,   model: 'gemma3-legal:latest' | 'nomic-embed-text:latest';,   chunkSize: number;,   confidence: number; } from

export interface ProcessingResult {
  success: boolean;
  documentId: string;
  ocr: {
    extractedText: string;
    confidence: number;
    processingMethod: string;
    legal?: {
      entities: any[];
      concepts: string[];
      documentType: string;
      jurisdiction: string;
    };
  };
  embeddings: {
    chunks: string[];
    vectors: number[][];
    indexedCount: number;
  };
  analysis: {
    summary: string;
    keywords: string[];
    complexity: 'low' | 'medium' | 'high';
    legalDomain: string[];
  };
  summarization: {
    sections: any[];
    keyInsights: string[];
    confidence: number;
  };
  storage: {
    minioUrl?: string;
    databaseId?: string;
  };
  metadata: {
    processingTime: number;
    stagesCompleted: string[];
    errors: string[];
  };
}

class UnifiedDocumentProcessor {
  private static instance: UnifiedDocumentProcessor;
  private goActor: any;
  private summaryActor: any;
  
  private constructor() {
    this.goActor = createActor(goMicroserviceMachine);
    this.summaryActor = createActor(aiSummaryMachine);
    this.goActor.start();
    this.summaryActor.start();
  }

  public static getInstance(): UnifiedDocumentProcessor {
    if (!UnifiedDocumentProcessor.instance) {
      UnifiedDocumentProcessor.instance = new UnifiedDocumentProcessor();
    }
    return UnifiedDocumentProcessor.instance;
  }

  /**
   * Process document through complete pipeline
   */
  async processDocument(
    file: File,
    config: DocumentProcessingConfig,
    metadata: {
      caseId: string;
      documentType: string;
      description?: string;
      tags?: string[];
    }
  ): Promise<ProcessingResult> {
    const startTime = Date.now();
    const documentId = this.generateDocumentId();
    const stagesCompleted: string[] = [];
    const errors: string[] = [];
    
    console.log(`🚀 Starting unified document processing for ${file.name}`);
    
    const result: ProcessingResult = {
      success: false,
      documentId,
      ocr: {
        extractedText: '',
        confidence: 0,
        processingMethod: ''
      },
      embeddings: {
        chunks: [],
        vectors: [],
        indexedCount: 0
      },
      analysis: {
        summary: '',
        keywords: [],
        complexity: 'low',
        legalDomain: []
      },
      summarization: {
        sections: [],
        keyInsights: [],
        confidence: 0
      },
      storage: {},
      metadata: {
        processingTime: 0,
        stagesCompleted,
        errors
      }
    };

    try {
      // Stage 1: OCR + LangExtract
      if (config.enableOCR) {
        console.log('📄 Stage 1: OCR + Legal Entity Extraction');
        try {
          const ocrResult = await this.performOCR(file, config.enableLegalBERT);
          result.ocr = ocrResult;
          stagesCompleted.push('OCR');
          console.log(`✅ OCR completed: ${ocrResult.extractedText.length} characters extracted`);
        } catch (error) {
          errors.push(`OCR failed: ${error.message}`);
          console.error('❌ OCR stage failed:', error);
        }
      }

      // Stage 2: Legal Analysis with Sentence Transformers
      if (result.ocr.extractedText) {
        console.log('🧠 Stage 2: Legal Analysis with Sentence Transformers');
        try {
          const analysis = await legalNLP.analyzeLegalDocument(result.ocr.extractedText);
          result.analysis = analysis;
          stagesCompleted.push('Legal Analysis');
          console.log(`✅ Legal analysis completed: ${analysis.legalDomain.join(', ')} domains detected`);
        } catch (error) {
          errors.push(`Legal analysis failed: ${error.message}`);
          console.error('❌ Legal analysis stage failed:', error);
        }
      }

      // Stage 3: Vector Embeddings with Nomic
      if (config.enableEmbeddings && result.ocr.extractedText) {
        console.log('🔗 Stage 3: Vector Embeddings Generation');
        try {
          const embeddingResult = await nomicEmbeddingService.processDocument(
            result.ocr.extractedText,
            {
              source: 'upload',
              title: file.name,
              entityType: metadata.documentType,
              entityId: documentId,
              caseId: metadata.caseId,
              tags: metadata.tags,
              description: metadata.description
            }
          );
          
          result.embeddings = {
            chunks: embeddingResult.chunks.map(chunk => chunk.content),
            vectors: embeddingResult.embeddings.map(emb => emb.embedding),
            indexedCount: embeddingResult.indexedCount
          };
          stagesCompleted.push('Embeddings');
          console.log(`✅ Embeddings completed: ${embeddingResult.chunks.length} chunks, ${embeddingResult.indexedCount} indexed`);
        } catch (error) {
          errors.push(`Embeddings failed: ${error.message}`);
          console.error('❌ Embeddings stage failed:', error);
        }
      }

      // Stage 4: AI Summarization with gemma3-legal
      if (config.enableSummarization && result.ocr.extractedText) {
        console.log('📝 Stage 4: AI Summarization with gemma3-legal');
        try {
          const summaryResult = await this.generateSummary(
            result.ocr.extractedText,
            metadata.documentType as any,
            documentId
          );
          result.summarization = summaryResult;
          stagesCompleted.push('Summarization');
          console.log(`✅ Summarization completed: ${summaryResult.sections.length} sections generated`);
        } catch (error) {
          errors.push(`Summarization failed: ${error.message}`);
          console.error('❌ Summarization stage failed:', error);
        }
      }

      // Stage 5: MinIO Storage (if enabled)
      if (config.enableMinIOStorage) {
        console.log('💾 Stage 5: MinIO Storage');
        try {
          const storageResult = await this.storeInMinIO(file, documentId, metadata);
          result.storage = storageResult;
          stagesCompleted.push('Storage');
          console.log(`✅ Storage completed: ${storageResult.minioUrl}`);
        } catch (error) {
          errors.push(`Storage failed: ${error.message}`);
          console.error('❌ Storage stage failed:', error);
        }
      }

      // Final result
      result.success = stagesCompleted.length > 0 && errors.length === 0;
      result.metadata.processingTime = Date.now() - startTime;
      result.metadata.stagesCompleted = stagesCompleted;
      result.metadata.errors = errors;

      console.log(`🎉 Document processing completed: ${stagesCompleted.length} stages successful, ${errors.length} errors`);
      return result;

    } catch (error) {
      console.error('❌ Document processing pipeline failed:', error);
      result.success = false;
      result.metadata.processingTime = Date.now() - startTime;
      result.metadata.errors.push(`Pipeline error: ${error.message}`);
      return result;
    }
  }

  /**
   * Stage 1: OCR + Legal Entity Extraction
   */
  private async performOCR(file: File, enableLegalBERT: boolean): Promise<ProcessingResult['ocr']> {
    const formData = new FormData();
    formData.append('file', file);

    const headers: Record<string, string> = {};
    if (enableLegalBERT) {
      headers['X-Enable-LegalBERT'] = 'true';
    }

    const response = await fetch('http://localhost:5176/api/ocr/langextract', {
      method: 'POST',
      body: formData,
      headers
    });

    if (!response.ok) {
      throw new Error(`OCR request failed: ${response.statusText}`);
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'OCR processing failed');
    }

    return {
      extractedText: data.text,
      confidence: data.confidence,
      processingMethod: data.processingMethod,
      legal: data.legal
    };
  }

  /**
   * Stage 4: AI Summarization with XState machine
   */
  private async generateSummary(
    content: string,
    documentType: 'evidence' | 'report' | 'contract' | 'case_law' | 'general',
    documentId: string
  ): Promise<ProcessingResult['summarization']> {
    return new Promise((resolve, reject) => {
      // Subscribe to state changes
      this.summaryActor.subscribe((state: any) => {
        if (state.matches('ready') && state.context.summary) {
          resolve({
            sections: state.context.sections || [],
            keyInsights: state.context.keyInsights || [],
            confidence: state.context.confidence || 0
          });
        } else if (state.matches('error')) {
          reject(new Error(state.context.error || 'Summarization failed'));
        }
      });

      // Start summarization
      this.summaryActor.send({
        type: 'GENERATE_SUMMARY',
        content,
        documentType
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        reject(new Error('Summarization timeout'));
      }, 30000);
    });
  }

  /**
   * Stage 5: MinIO Storage
   */
  private async storeInMinIO(
    file: File,
    documentId: string,
    metadata: any
  ): Promise<ProcessingResult['storage']> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('caseId', metadata.caseId);
    formData.append('documentType', metadata.documentType);
    formData.append('description', metadata.description || '');
    formData.append('documentId', documentId);

    const response = await fetch('http://localhost:5176/api/upload/minio', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`MinIO upload failed: ${response.statusText}`);
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'MinIO upload failed');
    }

    return {
      minioUrl: data.url,
      databaseId: data.documentId
    };
  }

  /**
   * Semantic search across processed documents
   */
  async semanticSearch(
    query: string,
    options: {
      caseId?: string;
      documentType?: string;
      limit?: number;
      threshold?: number;
    } = {}
  ): Promise<{
    results: any[];
    processingTime: number;
  }> {
    const startTime = Date.now();
    
    try {
      const results = await nomicEmbeddingService.similaritySearch(query, {
        k: options.limit || 10,
        threshold: options.threshold || 0.7,
        entityType: options.documentType,
        entityId: options.caseId
      });

      return {
        results: results.map(result => ({
          content: result.document.content,
          similarity: result.similarity,
          metadata: result.metadata,
          documentType: result.document.metadata.entityType,
          caseId: result.document.metadata.entityId
        })),
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('Semantic search failed:', error);
      throw error;
    }
  }

  /**
   * Batch process multiple documents
   */
  async batchProcess(
    files: File[],
    config: DocumentProcessingConfig,
    metadata: any
  ): Promise<ProcessingResult[]> {
    console.log(`🔄 Starting batch processing for ${files.length} documents`);
    
    const results: ProcessingResult[] = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      console.log(`📁 Processing document ${i + 1}/${files.length}: ${file.name}`);
      
      try {
        const result = await this.processDocument(file, config, {
          ...metadata,
          documentId: `${metadata.caseId}-doc-${i + 1}`
        });
        results.push(result);
      } catch (error) {
        console.error(`❌ Failed to process ${file.name}:`, error);
        results.push({
          success: false,
          documentId: `${metadata.caseId}-doc-${i + 1}`,
          ocr: { extractedText: '', confidence: 0, processingMethod: '' },
          embeddings: { chunks: [], vectors: [], indexedCount: 0 },
          analysis: { summary: '', keywords: [], complexity: 'low', legalDomain: [] },
          summarization: { sections: [], keyInsights: [], confidence: 0 },
          storage: {},
          metadata: {
            processingTime: 0,
            stagesCompleted: [],
            errors: [`Processing failed: ${error.message}`]
          }
        });
      }
    }
    
    console.log(`✅ Batch processing completed: ${results.filter(r => r.success).length}/${files.length} successful`);
    return results;
  }

  /**
   * Health check for all services
   */
  async healthCheck(): Promise<{
    overall: boolean;
    services: {
      ocr: boolean;
      embeddings: boolean;
      llm: boolean;
      storage: boolean;
    };
    details: any;
  }> {
    const checks = await Promise.allSettled([
      fetch('http://localhost:5176/api/ocr/langextract').then(r => r.ok).catch(() => false),
      fetch('http://localhost:11434/api/tags').then(r => r.ok).catch(() => false),
      Promise.resolve(nomicEmbeddingService.isInitialized),
      fetch('http://localhost:5176/api/upload/health').then(r => r.ok).catch(() => false)
    ]);

    const services = {
      ocr: checks[0].status === 'fulfilled' && checks[0].value,
      llm: checks[1].status === 'fulfilled' && checks[1].value,
      embeddings: checks[2].status === 'fulfilled' && checks[2].value,
      storage: checks[3].status === 'fulfilled' && checks[3].value
    };

    return {
      overall: Object.values(services).every(Boolean),
      services,
      details: {
        timestamp: new Date().toISOString(),
        models: await this.getAvailableModels()
      }
    };
  }

  private async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      const data = await response.json();
      return data.models?.map((m: any) => m.name) || [];
    } catch {
      return [];
    }
  }

  private generateDocumentId(): string {
    return `doc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Export singleton instance
export const unifiedDocumentProcessor = UnifiedDocumentProcessor.getInstance();
export default unifiedDocumentProcessor;