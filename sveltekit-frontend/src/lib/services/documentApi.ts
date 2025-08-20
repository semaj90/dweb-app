/**
 * Document API Service
 * Handles document processing, upload, and management operations
 */

export interface DocumentMetadata {
  filename: string;
  fileSize: number;
  mimeType: string;
  uploadedAt: string;
  caseId?: string;
  documentType?: string;
  tags?: string[];
  isConfidential?: boolean;
}

export interface ProcessingResult {
  documentId: string;
  status: 'processing' | 'completed' | 'failed';
  extractedText?: string;
  embeddings?: number[][];
  analysis?: {
    summary: string;
    entities: any[];
    sentiment: string;
    classification: string;
  };
  error?: string;
}

export interface UploadProgress {
  documentId: string;
  progress: number;
  stage: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
}

export class DocumentApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = '/api';
  }

  /**
   * Upload a document with metadata
   */
  async uploadDocument(
    file: File, 
    metadata: Partial<DocumentMetadata> = {}
  ): Promise<{ success: boolean; documentId?: string; error?: string }> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // Add metadata
      Object.entries(metadata).forEach(([key, value]) => {
        if (value !== undefined) {
          formData.append(key, String(value));
        }
      });

      const response = await fetch(`${this.baseUrl}/documents/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
      
    } catch (error) {
      console.error('Document upload failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed'
      };
    }
  }

  /**
   * Process document through evidence pipeline
   */
  async processDocument(documentId: string, options: {
    enableOCR?: boolean;
    enableEmbeddings?: boolean;
    enableAnalysis?: boolean;
    caseId?: string;
  } = {}): Promise<ProcessingResult> {
    try {
      const response = await fetch(`${this.baseUrl}/documents/${documentId}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(options)
      });

      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('Document processing failed:', error);
      return {
        documentId,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Processing failed'
      };
    }
  }

  /**
   * Get processing status
   */
  async getProcessingStatus(documentId: string): Promise<UploadProgress | null> {
    try {
      const response = await fetch(`${this.baseUrl}/documents/${documentId}/status`);
      
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('Status check failed:', error);
      return null;
    }
  }

  /**
   * Get document details
   */
  async getDocument(documentId: string): Promise<{
    metadata: DocumentMetadata;
    processing: ProcessingResult;
  } | null> {
    try {
      const response = await fetch(`${this.baseUrl}/documents/${documentId}`);
      
      if (!response.ok) {
        throw new Error(`Get document failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('Get document failed:', error);
      return null;
    }
  }

  /**
   * List documents for a case
   */
  async listDocuments(caseId?: string, options: {
    page?: number;
    limit?: number;
    type?: string;
    status?: string;
  } = {}): Promise<{
    documents: Array<{
      documentId: string;
      metadata: DocumentMetadata;
      status: string;
    }>;
    total: number;
    page: number;
    limit: number;
  }> {
    try {
      const params = new URLSearchParams();
      if (caseId) params.append('caseId', caseId);
      if (options.page) params.append('page', String(options.page));
      if (options.limit) params.append('limit', String(options.limit));
      if (options.type) params.append('type', options.type);
      if (options.status) params.append('status', options.status);

      const response = await fetch(`${this.baseUrl}/documents?${params}`);
      
      if (!response.ok) {
        throw new Error(`List documents failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('List documents failed:', error);
      return {
        documents: [],
        total: 0,
        page: 1,
        limit: 10
      };
    }
  }

  /**
   * Delete a document
   */
  async deleteDocument(documentId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/documents/${documentId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(`Delete failed: ${response.statusText}`);
      }

      return { success: true };
      
    } catch (error) {
      console.error('Document deletion failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Deletion failed'
      };
    }
  }

  /**
   * Search documents
   */
  async searchDocuments(query: string, options: {
    caseId?: string;
    type?: string;
    limit?: number;
    useSemanticSearch?: boolean;
  } = {}): Promise<{
    results: Array<{
      documentId: string;
      metadata: DocumentMetadata;
      relevanceScore: number;
      highlights?: string[];
    }>;
    total: number;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/documents/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          ...options
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('Document search failed:', error);
      return {
        results: [],
        total: 0
      };
    }
  }

  /**
   * Get document processing analytics
   */
  async getProcessingAnalytics(caseId?: string): Promise<{
    totalDocuments: number;
    processingStats: {
      completed: number;
      processing: number;
      failed: number;
    };
    averageProcessingTime: number;
    documentTypes: Record<string, number>;
  }> {
    try {
      const params = new URLSearchParams();
      if (caseId) params.append('caseId', caseId);

      const response = await fetch(`${this.baseUrl}/documents/analytics?${params}`);
      
      if (!response.ok) {
        throw new Error(`Analytics failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('Analytics failed:', error);
      return {
        totalDocuments: 0,
        processingStats: {
          completed: 0,
          processing: 0,
          failed: 0
        },
        averageProcessingTime: 0,
        documentTypes: {}
      };
    }
  }

  /**
   * Integrate with legal ingest API
   */
  async processLegalDocuments(files: File[], options: {
    caseId: string;
    jurisdiction?: string;
    enhanceRAG?: boolean;
  }): Promise<{
    success: boolean;
    caseId: string;
    documentsProcessed: number;
    totalProcessingTime: number;
    documents: Array<{
      id: string;
      filename: string;
      prosecutionScore: number;
      entityCount: number;
    }>;
    error?: string;
  }> {
    try {
      const formData = new FormData();
      
      files.forEach((file) => {
        formData.append('pdfFiles', file);
      });
      
      formData.append('caseId', options.caseId);
      if (options.jurisdiction) {
        formData.append('jurisdiction', options.jurisdiction);
      }
      if (options.enhanceRAG) {
        formData.append('enhanceRAG', 'true');
      }

      const response = await fetch(`${this.baseUrl}/legal/ingest`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Legal processing failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error('Legal document processing failed:', error);
      return {
        success: false,
        caseId: options.caseId,
        documentsProcessed: 0,
        totalProcessingTime: 0,
        documents: [],
        error: error instanceof Error ? error.message : 'Processing failed'
      };
    }
  }
}

// Export singleton instance
export const documentApiService = new DocumentApiService();