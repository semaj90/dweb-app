// gRPC Protocol Worker
// Handles gRPC-based communication for structured data transfer

class GRPCProtocolWorker {
  constructor() {
    this.connections = new Map();
    this.retryIntervals = new Map();
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
      // gRPC health check using gRPC-Web
      const startTime = Date.now();
      
      const response = await fetch(`${endpoint}/health`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/grpc-web+proto',
          'X-Grpc-Web': '1',
          'Accept': 'application/grpc-web+proto'
        },
        body: new Uint8Array([]), // Empty health check request
        signal: AbortSignal.timeout(5000)
      });

      const latency = Date.now() - startTime;

      self.postMessage({
        type: 'HEALTH_CHECK_RESPONSE',
        protocol,
        data: {
          status: response.ok ? 'healthy' : 'degraded',
          latency,
          grpcStatus: response.headers.get('grpc-status') || '0'
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
      const response = await this.executeGRPCRequest(request, endpoint, options);
      
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

  async executeGRPCRequest(request, endpoint, options) {
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
    
    // Convert to gRPC-Web format
    const grpcRequest = this.createGRPCMessage('rag.RAGQuery', {
      query: request.query,
      context: request.context || '',
      filters: JSON.stringify(request.filters || {}),
      options: {
        max_tokens: request.maxTokens || 1000,
        temperature: request.temperature || 0.7,
        stream: request.stream || false
      }
    });

    const response = await fetch(`${endpoint}/rag.RAGService/Query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/grpc-web+proto',
        'X-Grpc-Web': '1',
        'Accept': 'application/grpc-web+proto',
        'X-User-Agent': 'grpc-web-javascript/0.1'
      },
      body: grpcRequest,
      signal: AbortSignal.timeout(options.timeout || 15000)
    });

    if (!response.ok) {
      const grpcStatus = response.headers.get('grpc-status');
      const grpcMessage = response.headers.get('grpc-message');
      throw new Error(`gRPC RAG Query failed: ${grpcStatus} ${grpcMessage}`);
    }

    const responseData = await this.parseGRPCResponse(response);
    const processingTime = Date.now() - startTime;

    return {
      ...responseData,
      metadata: {
        ...responseData.metadata,
        protocol: 'gRPC',
        processingTime,
        grpcStatus: response.headers.get('grpc-status') || '0'
      }
    };
  }

  async handleDocumentUpload(request, endpoint, options) {
    // Convert document to gRPC streaming format
    const chunks = this.createDocumentChunks(request.document);
    const responses = [];

    for (const chunk of chunks) {
      const grpcRequest = this.createGRPCMessage('document.UploadRequest', {
        chunk_data: chunk.data,
        metadata: chunk.metadata,
        is_last: chunk.isLast
      });

      const response = await fetch(`${endpoint}/document.DocumentService/Upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/grpc-web+proto',
          'X-Grpc-Web': '1',
          'Accept': 'application/grpc-web+proto'
        },
        body: grpcRequest,
        signal: AbortSignal.timeout(options.timeout || 30000)
      });

      if (!response.ok) {
        const grpcStatus = response.headers.get('grpc-status');
        const grpcMessage = response.headers.get('grpc-message');
        throw new Error(`gRPC Document Upload failed: ${grpcStatus} ${grpcMessage}`);
      }

      const responseData = await this.parseGRPCResponse(response);
      responses.push(responseData);
    }

    return {
      success: true,
      documentId: responses[responses.length - 1]?.document_id,
      chunks: responses.length,
      protocol: 'gRPC'
    };
  }

  async handleSemanticSearch(request, endpoint, options) {
    const grpcRequest = this.createGRPCMessage('search.SearchRequest', {
      query: request.query,
      filters: JSON.stringify(request.filters || {}),
      limit: request.limit || 10,
      threshold: request.threshold || 0.7
    });

    const response = await fetch(`${endpoint}/search.SearchService/SemanticSearch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/grpc-web+proto',
        'X-Grpc-Web': '1',
        'Accept': 'application/grpc-web+proto'
      },
      body: grpcRequest,
      signal: AbortSignal.timeout(options.timeout || 8000)
    });

    if (!response.ok) {
      const grpcStatus = response.headers.get('grpc-status');
      const grpcMessage = response.headers.get('grpc-message');
      throw new Error(`gRPC Semantic Search failed: ${grpcStatus} ${grpcMessage}`);
    }

    return await this.parseGRPCResponse(response);
  }

  createGRPCMessage(messageType, data) {
    // Simplified gRPC-Web message creation
    // In a real implementation, you'd use protobuf.js or similar
    const jsonData = JSON.stringify(data);
    const messageLength = jsonData.length;
    
    // gRPC-Web frame format: [compressed flag][message length][message]
    const frame = new Uint8Array(5 + messageLength);
    frame[0] = 0; // Not compressed
    
    // Message length (big-endian uint32)
    frame[1] = (messageLength >>> 24) & 0xFF;
    frame[2] = (messageLength >>> 16) & 0xFF;
    frame[3] = (messageLength >>> 8) & 0xFF;
    frame[4] = messageLength & 0xFF;
    
    // Message payload (JSON for simplicity)
    const encoder = new TextEncoder();
    const messageBytes = encoder.encode(jsonData);
    frame.set(messageBytes, 5);
    
    return frame;
  }

  async parseGRPCResponse(response) {
    const arrayBuffer = await response.arrayBuffer();
    const data = new Uint8Array(arrayBuffer);
    
    if (data.length < 5) {
      throw new Error('Invalid gRPC response: too short');
    }
    
    // Parse gRPC-Web frame
    const compressed = data[0];
    const messageLength = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4];
    
    if (data.length < 5 + messageLength) {
      throw new Error('Invalid gRPC response: incomplete message');
    }
    
    const messageBytes = data.slice(5, 5 + messageLength);
    const decoder = new TextDecoder();
    const jsonString = decoder.decode(messageBytes);
    
    try {
      return JSON.parse(jsonString);
    } catch (error) {
      throw new Error(`Failed to parse gRPC response JSON: ${error.message}`);
    }
  }

  createDocumentChunks(document) {
    const chunkSize = 1024 * 1024; // 1MB chunks
    const chunks = [];
    
    if (document.content) {
      const content = typeof document.content === 'string' 
        ? new TextEncoder().encode(document.content)
        : document.content;
      
      for (let i = 0; i < content.length; i += chunkSize) {
        const chunk = content.slice(i, i + chunkSize);
        chunks.push({
          data: chunk,
          metadata: i === 0 ? {
            filename: document.title,
            content_type: document.type,
            case_id: document.caseId
          } : null,
          isLast: i + chunkSize >= content.length
        });
      }
    } else {
      // Single chunk for metadata-only documents
      chunks.push({
        data: new Uint8Array(0),
        metadata: {
          filename: document.title,
          content_type: document.type,
          case_id: document.caseId
        },
        isLast: true
      });
    }
    
    return chunks;
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
const grpcWorker = new GRPCProtocolWorker();

self.addEventListener('message', (event) => {
  grpcWorker.handleMessage(event);
});