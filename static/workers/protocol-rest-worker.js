// REST Protocol Worker
// Handles REST API communication as fallback protocol

class RESTProtocolWorker {
  constructor() {
    this.activeRequests = new Map();
    this.retryDelays = [1000, 2000, 4000]; // Exponential backoff
  }

  async handleMessage(event) {
    const { type, requestId, protocol, request, endpoint, options } = event.data;

    try {
      switch (type) {
        case 'HEALTH_CHECK':
          await this.handleHealthCheck(endpoint, protocol);
          break;
          
        case 'EXECUTE_REQUEST':
          await this.handleRequest(requestId, request, endpoint, options);
          break;
          
        default:
          this.postError(requestId, `Unknown message type: ${type}`);
      }
    } catch (error) {
      this.postError(requestId, error.message);
    }
  }

  async handleHealthCheck(endpoint, protocol) {
    try {
      const startTime = Date.now();
      
      const response = await fetch(`${endpoint}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'X-Protocol': 'REST'
        },
        signal: AbortSignal.timeout(8000)
      });

      const latency = Date.now() - startTime;
      let data = {};
      
      try {
        data = await response.json();
      } catch (e) {
        // Health endpoint might return plain text
        data = { status: response.ok ? 'healthy' : 'error' };
      }

      self.postMessage({
        type: 'HEALTH_CHECK_RESPONSE',
        protocol,
        data: {
          status: response.ok ? 'healthy' : 'degraded',
          latency,
          httpStatus: response.status,
          response: data
        }
      });
    } catch (error) {
      self.postMessage({
        type: 'HEALTH_CHECK_RESPONSE',
        protocol,
        data: {
          status: 'error',
          error: error.message
        }
      });
    }
  }

  async handleRequest(requestId, request, endpoint, options) {
    const maxRetries = options.retries || 3;
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await this.executeRESTRequest(request, endpoint, options, attempt);
        
        self.postMessage({
          type: 'REQUEST_COMPLETE',
          requestId,
          data: response
        });
        return;
      } catch (error) {
        lastError = error;
        
        if (attempt < maxRetries && this.isRetryableError(error)) {
          const delay = this.retryDelays[Math.min(attempt, this.retryDelays.length - 1)];
          await this.sleep(delay);
          continue;
        }
        break;
      }
    }

    self.postMessage({
      type: 'REQUEST_ERROR',
      requestId,
      data: { 
        message: lastError.message,
        attempts: maxRetries + 1
      }
    });
  }

  async executeRESTRequest(request, endpoint, options, attempt = 0) {
    const { type } = request;
    
    switch (type) {
      case 'rag_query':
        return await this.handleRAGQuery(request, endpoint, options, attempt);
      case 'document_upload':
        return await this.handleDocumentUpload(request, endpoint, options, attempt);
      case 'semantic_search':
        return await this.handleSemanticSearch(request, endpoint, options, attempt);
      default:
        throw new Error(`Unknown request type: ${type}`);
    }
  }

  async handleRAGQuery(request, endpoint, options, attempt) {
    const startTime = Date.now();
    
    const requestBody = {
      query: request.query,
      context: request.context,
      filters: request.filters || {},
      options: {
        maxTokens: request.maxTokens || 1000,
        temperature: request.temperature || 0.7,
        stream: request.stream || false,
        topK: request.topK || 5,
        ...options
      },
      metadata: {
        requestId: `rest_${Date.now()}_${attempt}`,
        timestamp: Date.now(),
        attempt
      }
    };

    const response = await fetch(`${endpoint}/api/rag/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Protocol': 'REST',
        'X-Request-ID': requestBody.metadata.requestId,
        'X-Attempt': attempt.toString()
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(options.timeout || 20000)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`REST RAG Query failed: ${response.status} ${response.statusText} - ${errorData.message || 'Unknown error'}`);
    }

    const data = await response.json();
    const processingTime = Date.now() - startTime;

    return {
      ...data,
      metadata: {
        ...data.metadata,
        protocol: 'REST',
        processingTime,
        httpStatus: response.status,
        attempt,
        networkLatency: processingTime - (data.metadata?.aiProcessingTime || 0)
      }
    };
  }

  async handleDocumentUpload(request, endpoint, options, attempt) {
    const formData = new FormData();
    
    // Handle different document input types
    if (request.document.file instanceof File) {
      formData.append('file', request.document.file);
    } else if (request.document.content) {
      // Create blob from content
      const blob = new Blob([request.document.content], { 
        type: request.document.mimeType || 'text/plain' 
      });
      formData.append('file', blob, request.document.title || 'document.txt');
    }
    
    formData.append('metadata', JSON.stringify({
      title: request.document.title,
      type: request.document.type,
      caseId: request.document.caseId,
      description: request.document.description,
      tags: request.document.tags || [],
      processingOptions: {
        extractText: true,
        generateEmbeddings: true,
        createSummary: true,
        ...request.document.processingOptions
      }
    }));

    const response = await fetch(`${endpoint}/api/documents/upload`, {
      method: 'POST',
      headers: {
        'X-Protocol': 'REST',
        'X-Request-ID': `rest_upload_${Date.now()}_${attempt}`,
        'X-Attempt': attempt.toString()
      },
      body: formData,
      signal: AbortSignal.timeout(options.timeout || 60000)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`REST Document Upload failed: ${response.status} ${response.statusText} - ${errorData.message || 'Unknown error'}`);
    }

    const data = await response.json();
    
    return {
      ...data,
      metadata: {
        ...data.metadata,
        protocol: 'REST',
        httpStatus: response.status,
        attempt
      }
    };
  }

  async handleSemanticSearch(request, endpoint, options, attempt) {
    const requestBody = {
      query: request.query,
      filters: request.filters || {},
      limit: request.limit || 10,
      threshold: request.threshold || 0.7,
      includeMetadata: true,
      includeContent: request.includeContent !== false,
      sortBy: request.sortBy || 'relevance'
    };

    const response = await fetch(`${endpoint}/api/search/semantic`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Protocol': 'REST',
        'X-Request-ID': `rest_search_${Date.now()}_${attempt}`,
        'X-Attempt': attempt.toString()
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(options.timeout || 10000)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`REST Semantic Search failed: ${response.status} ${response.statusText} - ${errorData.message || 'Unknown error'}`);
    }

    const data = await response.json();
    
    return {
      ...data,
      metadata: {
        ...data.metadata,
        protocol: 'REST',
        httpStatus: response.status,
        attempt,
        query: request.query
      }
    };
  }

  isRetryableError(error) {
    // Determine if an error is worth retrying
    if (error.name === 'AbortError') {
      return false; // Timeout errors shouldn't be retried immediately
    }
    
    const message = error.message.toLowerCase();
    
    // Network errors that might be temporary
    if (message.includes('network') || 
        message.includes('timeout') || 
        message.includes('connection') ||
        message.includes('502') ||
        message.includes('503') ||
        message.includes('504')) {
      return true;
    }
    
    // Server errors that might be temporary
    if (message.includes('500') || message.includes('internal server error')) {
      return true;
    }
    
    return false;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  postError(requestId, message) {
    self.postMessage({
      type: 'REQUEST_ERROR',
      requestId,
      data: { message }
    });
  }
}

// Initialize worker
const restWorker = new RESTProtocolWorker();

self.addEventListener('message', (event) => {
  restWorker.handleMessage(event);
});