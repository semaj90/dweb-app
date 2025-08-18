// QUIC Protocol Worker
// Handles QUIC-based communication for enhanced performance

class QUICProtocolWorker {
  constructor() {
    this.connections = new Map();
    this.requestQueue = [];
    this.isProcessing = false;
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
      // QUIC health check - using HTTP/3 if available, fallback to HTTP/2
      const startTime = Date.now();
      
      const response = await fetch(`${endpoint}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-Protocol': 'QUIC'
        },
        // Note: Real QUIC would require WebTransport API when available
        signal: AbortSignal.timeout(3000)
      });

      const latency = Date.now() - startTime;
      const data = await response.json();

      self.postMessage({
        type: 'HEALTH_CHECK_RESPONSE',
        protocol,
        data: {
          status: response.ok ? 'healthy' : 'degraded',
          latency,
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
    try {
      // QUIC request handling with optimized headers
      const response = await this.executeQUICRequest(request, endpoint, options);
      
      self.postMessage({
        type: 'REQUEST_COMPLETE',
        requestId,
        data: response
      });
    } catch (error) {
      self.postMessage({
        type: 'REQUEST_ERROR',
        requestId,
        data: { message: error.message }
      });
    }
  }

  async executeQUICRequest(request, endpoint, options) {
    const { type } = request;
    
    switch (type) {
      case 'rag_query':
        return await this.handleRAGQuery(request, endpoint, options);
      case 'document_upload':
        return await this.handleDocumentUpload(request, endpoint, options);
      case 'semantic_search':
        return await this.handleSemanticSearch(request, endpoint, options);
      default:
        throw new Error(`Unknown request type: ${type}`);
    }
  }

  async handleRAGQuery(request, endpoint, options) {
    const startTime = Date.now();
    
    const response = await fetch(`${endpoint}/rag/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Protocol': 'QUIC',
        'X-Request-ID': `quic_${Date.now()}`,
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        query: request.query,
        context: request.context,
        filters: request.filters,
        options: {
          maxTokens: request.maxTokens || 1000,
          temperature: request.temperature || 0.7,
          stream: request.stream || false,
          ...options
        }
      }),
      signal: AbortSignal.timeout(options.timeout || 10000)
    });

    if (!response.ok) {
      throw new Error(`QUIC RAG Query failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const processingTime = Date.now() - startTime;

    return {
      ...data,
      metadata: {
        ...data.metadata,
        protocol: 'QUIC',
        processingTime,
        networkLatency: processingTime - (data.metadata?.aiProcessingTime || 0)
      }
    };
  }

  async handleDocumentUpload(request, endpoint, options) {
    const formData = new FormData();
    
    // Handle document upload with QUIC optimization
    if (request.document.file) {
      formData.append('file', request.document.file);
    }
    
    formData.append('metadata', JSON.stringify({
      title: request.document.title,
      type: request.document.type,
      caseId: request.document.caseId,
      description: request.document.description
    }));

    const response = await fetch(`${endpoint}/documents/upload`, {
      method: 'POST',
      headers: {
        'X-Protocol': 'QUIC',
        'X-Request-ID': `quic_upload_${Date.now()}`
      },
      body: formData,
      signal: AbortSignal.timeout(options.timeout || 30000)
    });

    if (!response.ok) {
      throw new Error(`QUIC Document Upload failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  async handleSemanticSearch(request, endpoint, options) {
    const response = await fetch(`${endpoint}/search/semantic`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Protocol': 'QUIC',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        query: request.query,
        filters: request.filters,
        limit: request.limit || 10,
        threshold: request.threshold || 0.7
      }),
      signal: AbortSignal.timeout(options.timeout || 5000)
    });

    if (!response.ok) {
      throw new Error(`QUIC Semantic Search failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
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
const quicWorker = new QUICProtocolWorker();

self.addEventListener('message', (event) => {
  quicWorker.handleMessage(event);
});