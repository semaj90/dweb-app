/**
 * Enhanced Ingest Integration Service
 * Integrates with your existing ai-agent.ts store and production service architecture
 * Follows your established patterns and conventions
 */

import { aiAgentStore } from '$lib/stores/ai-agent';
import { get } from 'svelte/store';
import type { 
  DocumentIngestRequest, 
  BatchIngestRequest,
  IngestResult,
  BatchIngestResult,
  ChunkingOptions,
  ChunkedDocument,
  LegalSection,
  SimilarDocument
} from '$lib/types/ingest';

export class EnhancedIngestService {
  private baseUrl: string;
  private timeout: number;
  private retryAttempts: number;

  constructor() {
    // Use your established environment configuration pattern
    this.baseUrl = typeof window !== 'undefined' 
      ? window.location.origin 
      : 'http://localhost:5173';
    this.timeout = 30000; // 30 seconds
    this.retryAttempts = 3;
  }

  /**
   * Ingest single document with AI agent integration
   * Follows your aiAgentStore patterns for error handling and progress tracking
   */
  async ingestDocument(request: DocumentIngestRequest): Promise<IngestResult> {
    const startTime = Date.now();
    
    // Update AI agent store with processing status (following your pattern)
    aiAgentStore.update(state => ({
      ...state,
      isProcessing: true,
      currentTask: 'document_ingest'
    }));

    try {
      // Generate embedding preview using your existing embedding service
      let embeddingPreview: number[] = [];
      try {
        const aiAgent = get(aiAgentStore);
        const similarDocs = await aiAgent.searchSimilarDocuments?.(
          request.content.substring(0, 500), 1
        );
        if (similarDocs && similarDocs.length > 0) {
          embeddingPreview = similarDocs[0].embedding?.slice(0, 5) || [];
        }
      } catch (embedError) {
        console.warn('Embedding preview failed:', embedError);
      }

      // Call SvelteKit API (which proxies to Go service)
      const aiAgent = get(aiAgentStore);
      const response = await this.callIngestAPI('/api/v1/ingest', {
        ...request,
        metadata: {
          ...request.metadata,
          // Integrate with your AI agent metadata patterns
          ai_agent_session: aiAgent.activeSessionId,
          embedding_preview: embeddingPreview,
          processing_mode: 'enhanced_ai_integration'
        }
      });

      if (!response.success) {
        throw new Error(response.error || 'Ingest failed');
      }

      // Update AI agent store with success (following your pattern)
      aiAgentStore.update(state => ({
        ...state,
        isProcessing: false,
        vectorStore: {
          ...state.vectorStore,
          documentCount: state.vectorStore.documentCount + 1,
          lastIndexUpdate: new Date(),
          isIndexed: true
        },
        // Add to processing history following your job pattern
        completedJobs: [
          ...state.completedJobs,
          {
            id: response.id,
            type: 'ingest',
            status: 'completed',
            input: request,
            output: response,
            startTime: new Date(startTime),
            endTime: new Date(),
            retryCount: 0
          }
        ]
      }));

      return {
        success: true,
        documentId: response.document_id,
        embeddingId: response.embedding_id,
        processingTime: response.process_time_ms,
        metadata: response.service_info
      };

    } catch (error) {
      // Error handling following your aiAgentStore pattern
      aiAgentStore.update(state => ({
        ...state,
        isProcessing: false,
        errors: [
          ...state.errors,
          {
            id: `ingest_${Date.now()}`,
            type: 'processing',
            message: `Document ingest failed: ${error.message}`,
            timestamp: new Date(),
            context: { request },
            resolved: false,
            retryable: true
          }
        ]
      }));

      throw error;
    }
  }

  /**
   * Batch ingest with progress tracking
   * Integrates with your existing batch processing patterns
   */
  async ingestBatch(requests: DocumentIngestRequest[]): Promise<BatchIngestResult> {
    const batchId = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize batch processing state
    aiAgentStore.update(state => ({
      ...state,
      isProcessing: true,
      currentTask: 'batch_ingest',
      processingQueue: [
        ...state.processingQueue,
        {
          id: batchId,
          type: 'batch_ingest',
          status: 'pending',
          input: { documents: requests },
          startTime: new Date(),
          retryCount: 0
        }
      ]
    }));

    try {
      // Enhanced batch request with AI agent context
      const aiAgent = get(aiAgentStore);
      const batchRequest: BatchIngestRequest = {
        documents: requests.map((doc, index) => ({
          ...doc,
          metadata: {
            ...doc.metadata,
            batch_id: batchId,
            batch_index: index,
            ai_agent_session: aiAgent.activeSessionId
          }
        }))
      };

      const response = await this.callIngestAPI('/api/v1/ingest/batch', batchRequest);

      if (!response.success) {
        throw new Error(response.error || 'Batch ingest failed');
      }

      // Update AI agent store with batch completion
      aiAgentStore.update(state => ({
        ...state,
        isProcessing: false,
        vectorStore: {
          ...state.vectorStore,
          documentCount: state.vectorStore.documentCount + response.batch_summary.processed,
          lastIndexUpdate: new Date(),
          isIndexed: true
        },
        // Complete the batch job
        processingQueue: state.processingQueue.filter(job => job.id !== batchId),
        completedJobs: [
          ...state.completedJobs,
          {
            id: batchId,
            type: 'batch_ingest',
            status: 'completed',
            input: batchRequest,
            output: response,
            startTime: new Date(),
            endTime: new Date(),
            retryCount: 0
          }
        ]
      }));

      return {
        success: true,
        batchId,
        processed: response.batch_summary.processed,
        failed: response.batch_summary.failed,
        successRate: response.batch_summary.success_rate,
        results: response.results.map(r => ({
          documentId: r.document_id,
          embeddingId: r.embedding_id,
          processingTime: r.process_time_ms
        })),
        performance: response.performance
      };

    } catch (error) {
      // Fail the batch job in AI agent store
      aiAgentStore.update(state => ({
        ...state,
        isProcessing: false,
        processingQueue: state.processingQueue.filter(job => job.id !== batchId),
        completedJobs: [
          ...state.completedJobs,
          {
            id: batchId,
            type: 'batch_ingest',
            status: 'failed',
            input: { documents: requests },
            error: error.message,
            startTime: new Date(),
            endTime: new Date(),
            retryCount: 0
          }
        ]
      }));

      throw error;
    }
  }

  /**
   * Smart document chunking using your sentence-transformer service patterns
   */
  async intelligentChunking(content: string, options: ChunkingOptions = {}): Promise<ChunkedDocument[]> {
    const {
      maxChunkSize = 600,
      overlap = 60,
      preserveSentences = true,
      legalAware = true
    } = options;

    // Use your existing legal NLP patterns for smart chunking
    const chunks: ChunkedDocument[] = [];
    
    if (legalAware) {
      // Legal-aware chunking (preserve legal sections, clauses, etc.)
      const legalSections = this.extractLegalSections(content);
      
      for (const section of legalSections) {
        if (section.content.length <= maxChunkSize) {
          chunks.push({
            content: section.content,
            index: chunks.length,
            metadata: {
              section_type: section.type,
              legal_context: section.context,
              preserves_legal_structure: true
            }
          });
        } else {
          // Split large sections while preserving legal structure
          const subChunks = this.splitLegalSection(section, maxChunkSize, overlap);
          chunks.push(...subChunks.map((chunk, i) => ({
            content: chunk,
            index: chunks.length + i,
            metadata: {
              section_type: section.type,
              legal_context: section.context,
              sub_chunk: true,
              parent_section: section.title
            }
          })));
        }
      }
    } else {
      // Standard chunking
      chunks.push(...this.standardChunking(content, maxChunkSize, overlap, preserveSentences));
    }

    return chunks;
  }

  /**
   * Enhanced search integration with your existing vector search patterns
   */
  async searchSimilarToIngestedDocument(
    documentId: string, 
    query?: string, 
    limit: number = 5
  ): Promise<SimilarDocument[]> {
    // Leverage your existing searchSimilarDocuments from ai-agent store
    const aiAgent = get(aiAgentStore);
    
    try {
      // If no query provided, use the document content
      let searchQuery = query;
      if (!searchQuery) {
        // Fetch document content from the ingest service
        const docResponse = await fetch(`${this.baseUrl}/api/v1/ingest/document/${documentId}`);
        if (docResponse.ok) {
          const docData = await docResponse.json();
          searchQuery = docData.content?.substring(0, 200) || '';
        }
      }

      if (!searchQuery) {
        throw new Error('No query or document content available for search');
      }

      // Use your existing similar document search
      return await aiAgent.searchSimilarDocuments?.(searchQuery, limit) || [];
      
    } catch (error) {
      console.error('Similar document search failed:', error);
      return [];
    }
  }

  // Private helper methods following your established patterns

  private async callIngestAPI(endpoint: string, data: any): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API call failed: ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  private extractLegalSections(content: string): LegalSection[] {
    // Legal document structure detection
    const sections: LegalSection[] = [];
    
    // Common legal section patterns
    const sectionPatterns = [
      /^(\d+\.\s+[A-Z][^\.]+)/gm, // Numbered sections
      /^([A-Z][A-Z\s]+)$/gm, // All caps headers
      /^(WHEREAS[^;]*;)/gm, // WHEREAS clauses
      /^(NOW, THEREFORE[^;]*;)/gm, // NOW THEREFORE clauses
    ];

    // Implementation would follow your legal document parsing patterns
    // This is a simplified version for the example
    
    return sections;
  }

  private splitLegalSection(section: LegalSection, maxSize: number, overlap: number): string[] {
    // Smart legal section splitting that preserves clause boundaries
    const chunks: string[] = [];
    
    // Implementation would use your sentence-transformer chunking logic
    // This is simplified for the example
    
    return chunks;
  }

  private standardChunking(content: string, maxSize: number, overlap: number, preserveSentences: boolean): ChunkedDocument[] {
    // Standard text chunking following your established patterns
    const chunks: ChunkedDocument[] = [];
    
    // Implementation would use your text chunking logic from sentence-transformer.ts
    
    return chunks;
  }
}

// Type definitions following your established patterns
interface IngestResult {
  success: boolean;
  documentId: string;
  embeddingId: string;
  processingTime: number;
  metadata?: any;
}

interface BatchIngestResult {
  success: boolean;
  batchId: string;
  processed: number;
  failed: number;
  successRate: string;
  results: Array<{
    documentId: string;
    embeddingId: string;
    processingTime: number;
  }>;
  performance?: any;
}

interface ChunkingOptions {
  maxChunkSize?: number;
  overlap?: number;
  preserveSentences?: boolean;
  legalAware?: boolean;
}

interface ChunkedDocument {
  content: string;
  index: number;
  metadata?: Record<string, any>;
}

interface LegalSection {
  title: string;
  content: string;
  type: string;
  context: string;
}

interface SimilarDocument {
  id: string;
  title: string;
  content: string;
  similarity: number;
  metadata?: Record<string, any>;
  embedding?: number[];
}

// Export singleton instance following your service patterns
export const enhancedIngestService = new EnhancedIngestService();