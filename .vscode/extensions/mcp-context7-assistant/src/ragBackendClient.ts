/**
 * Enhanced RAG Backend Client
 * Provides integration with the Enhanced RAG Backend API
 */

import * as vscode from 'vscode';
import fetch from 'node-fetch';

export interface RAGConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
}

export interface DocumentUpload {
  file: Buffer;
  fileName: string;
  title?: string;
  documentType?: string;
  caseId?: string;
  metadata?: Record<string, any>;
}

export interface SearchQuery {
  query: string;
  caseId?: string;
  documentTypes?: string[];
  limit?: number;
  threshold?: number;
  includeContent?: boolean;
  searchType?: 'vector' | 'hybrid' | 'chunks';
}

export interface WorkflowRequest {
  workflowType: 'document_analysis' | 'legal_research' | 'case_preparation' | 'contract_review' | 'evidence_analysis';
  input: unknown;
  options?: {
    skipCache?: boolean;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    timeout?: number;
    context?: Record<string, any>;
  };
}

export class RAGBackendClient {
  private config: RAGConfig;
  private outputChannel: vscode.OutputChannel;

  constructor(config?: Partial<RAGConfig>) {
    this.config = {
      baseUrl: config?.baseUrl || 'http://localhost:8000',
      timeout: config?.timeout || 30000,
      retries: config?.retries || 3
    };

    this.outputChannel = vscode.window.createOutputChannel('Enhanced RAG Backend');
  }

  /**
   * Check if the RAG backend is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.makeRequest('/health', {
        method: 'GET',
        timeout: 5000
      });

      return response.status === 'healthy';
    } catch (error) {
      this.outputChannel.appendLine(`Health check failed: ${error}`);
      return false;
    }
  }

  /**
   * Get detailed system metrics
   */
  async getSystemMetrics(): Promise<any> {
    try {
      return await this.makeRequest('/health/detailed');
    } catch (error) {
      this.outputChannel.appendLine(`Failed to get system metrics: ${error}`);
      throw error;
    }
  }

  /**
   * Search documents using semantic/vector search
   */
  async searchDocuments(query: SearchQuery): Promise<any> {
    try {
      this.outputChannel.appendLine(`Searching: "${query.query}" (${query.searchType || 'hybrid'})`);
      
      const response = await this.makeRequest('/api/v1/rag/search', {
        method: 'POST',
        body: JSON.stringify(query)
      });

      this.outputChannel.appendLine(`Found ${response.results?.length || 0} results`);
      return response;
    } catch (error) {
      this.outputChannel.appendLine(`Search failed: ${error}`);
      throw error;
    }
  }

  /**
   * Upload and process document
   */
  async uploadDocument(upload: DocumentUpload): Promise<any> {
    try {
      this.outputChannel.appendLine(`Uploading document: ${upload.fileName}`);

      const formData = new FormData();
      const blob = new Blob([upload.file]);
      formData.append('document', blob, upload.fileName);
      
      if (upload.title) formData.append('title', upload.title);
      if (upload.documentType) formData.append('documentType', upload.documentType);
      if (upload.caseId) formData.append('caseId', upload.caseId);
      if (upload.metadata) formData.append('metadata', JSON.stringify(upload.metadata));

      const response = await this.makeRequest('/api/v1/rag/upload', {
        method: 'POST',
        body: formData,
        headers: {} // Let fetch set Content-Type for FormData
      });

      this.outputChannel.appendLine(`Document processed: ${response.document?.id}`);
      return response;
    } catch (error) {
      this.outputChannel.appendLine(`Upload failed: ${error}`);
      throw error;
    }
  }

  /**
   * Execute multi-agent workflow
   */
  async executeWorkflow(request: WorkflowRequest): Promise<any> {
    try {
      this.outputChannel.appendLine(`Executing workflow: ${request.workflowType}`);

      const response = await this.makeRequest('/api/v1/agents/workflow', {
        method: 'POST',
        body: JSON.stringify(request),
        timeout: request.options?.timeout || 120000 // 2 minutes default for workflows
      });

      this.outputChannel.appendLine(`Workflow completed in ${response.result?.metadata?.processingTime}ms`);
      return response;
    } catch (error) {
      this.outputChannel.appendLine(`Workflow failed: ${error}`);
      throw error;
    }
  }

  /**
   * Chat with AI agent
   */
  async chatWithAgent(messages: Array<{role: string, content: string}>, options?: unknown): Promise<any> {
    try {
      this.outputChannel.appendLine(`Starting AI chat with ${messages.length} messages`);

      const response = await this.makeRequest('/api/v1/agents/chat', {
        method: 'POST',
        body: JSON.stringify({
          messages,
          options: options || {}
        })
      });

      return response;
    } catch (error) {
      this.outputChannel.appendLine(`Chat failed: ${error}`);
      throw error;
    }
  }

  /**
   * Analyze text with AI
   */
  async analyzeText(text: string, analysisType?: string, options?: unknown): Promise<any> {
    try {
      this.outputChannel.appendLine(`Analyzing text (${analysisType || 'general'}): ${text.substring(0, 100)}...`);

      const response = await this.makeRequest('/api/v1/rag/analyze', {
        method: 'POST',
        body: JSON.stringify({
          text,
          analysisType: analysisType || 'general',
          options: options || {}
        })
      });

      return response;
    } catch (error) {
      this.outputChannel.appendLine(`Analysis failed: ${error}`);
      throw error;
    }
  }

  /**
   * Summarize text
   */
  async summarizeText(text: string, length?: string, options?: unknown): Promise<any> {
    try {
      this.outputChannel.appendLine(`Summarizing text (${length || 'medium'}): ${text.substring(0, 100)}...`);

      const response = await this.makeRequest('/api/v1/rag/summarize', {
        method: 'POST',
        body: JSON.stringify({
          text,
          length: length || 'medium',
          options: options || {}
        })
      });

      return response;
    } catch (error) {
      this.outputChannel.appendLine(`Summarization failed: ${error}`);
      throw error;
    }
  }

  /**
   * Get RAG system statistics
   */
  async getRAGStats(): Promise<any> {
    try {
      return await this.makeRequest('/api/v1/rag/stats');
    } catch (error) {
      this.outputChannel.appendLine(`Failed to get RAG stats: ${error}`);
      throw error;
    }
  }

  /**
   * Get document by ID
   */
  async getDocument(id: string, includeContent?: boolean, includeChunks?: boolean): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (includeContent) params.set('includeContent', 'true');
      if (includeChunks) params.set('includeChunks', 'true');

      const url = `/api/v1/documents/${id}${params.toString() ? '?' + params.toString() : ''}`;
      return await this.makeRequest(url);
    } catch (error) {
      this.outputChannel.appendLine(`Failed to get document ${id}: ${error}`);
      throw error;
    }
  }

  /**
   * List documents with filters
   */
  async listDocuments(filters?: {
    caseId?: string;
    documentType?: string;
    limit?: number;
    offset?: number;
    search?: string;
  }): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (filters?.caseId) params.set('caseId', filters.caseId);
      if (filters?.documentType) params.set('documentType', filters.documentType);
      if (filters?.limit) params.set('limit', filters.limit.toString());
      if (filters?.offset) params.set('offset', filters.offset.toString());
      if (filters?.search) params.set('search', filters.search);

      const url = `/api/v1/documents${params.toString() ? '?' + params.toString() : ''}`;
      return await this.makeRequest(url);
    } catch (error) {
      this.outputChannel.appendLine(`Failed to list documents: ${error}`);
      throw error;
    }
  }

  /**
   * Find similar documents
   */
  async findSimilarDocuments(documentId: string, limit?: number, threshold?: number): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (limit) params.set('limit', limit.toString());
      if (threshold) params.set('threshold', threshold.toString());

      const url = `/api/v1/rag/similar/${documentId}${params.toString() ? '?' + params.toString() : ''}`;
      return await this.makeRequest(url);
    } catch (error) {
      this.outputChannel.appendLine(`Failed to find similar documents: ${error}`);
      throw error;
    }
  }

  /**
   * Clear cache
   */
  async clearCache(pattern?: string): Promise<any> {
    try {
      const params = new URLSearchParams();
      if (pattern) params.set('pattern', pattern);

      const url = `/api/v1/rag/cache${params.toString() ? '?' + params.toString() : ''}`;
      return await this.makeRequest(url, { method: 'DELETE' });
    } catch (error) {
      this.outputChannel.appendLine(`Failed to clear cache: ${error}`);
      throw error;
    }
  }

  /**
   * Make HTTP request with retry logic
   */
  private async makeRequest(endpoint: string, options: unknown = {}): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    const requestOptions = {
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'VS-Code-Extension/1.0.0',
        ...options.headers
      },
      body: options.body,
      timeout: options.timeout || this.config.timeout
    };

    // Remove Content-Type for FormData
    if (options.body instanceof FormData) {
      delete requestOptions.headers['Content-Type'];
    }

    let lastError: unknown;
    
    for (let attempt = 1; attempt <= this.config.retries; attempt++) {
      try {
        this.outputChannel.appendLine(`Request attempt ${attempt}/${this.config.retries}: ${requestOptions.method} ${url}`);
        
        const response = await fetch(url, requestOptions);
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          return await response.json();
        } else {
          return await response.text();
        }

      } catch (error) {
        lastError = error;
        this.outputChannel.appendLine(`Attempt ${attempt} failed: ${error}`);
        
        if (attempt < this.config.retries) {
          const delay = Math.pow(2, attempt - 1) * 1000; // Exponential backoff
          this.outputChannel.appendLine(`Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError;
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<RAGConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.outputChannel.appendLine(`Updated RAG backend config: ${JSON.stringify(this.config)}`);
  }

  /**
   * Get current configuration
   */
  getConfig(): RAGConfig {
    return { ...this.config };
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.outputChannel.dispose();
  }
}