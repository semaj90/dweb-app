import { parentPort, workerData } from "worker_threads";
import { WebSocket } from "ws";
import { EventEmitter } from "events";

/**
 * Phase 4: Streaming Worker
 * Handles real-time data streaming and WebSocket connections
 */

// Type definitions
interface WorkerMessage {
  taskId: string;
  data: {
    type: string;
    data?: any;
    connectionData?: ConnectionData;
    message?: any;
    channels?: string[];
    chunks?: any[];
    connectionId?: string;
    bufferId?: string;
    operation?: string;
    bufferData?: any;
  };
  options?: any;
}

interface ConnectionData {
  url: string;
  protocols?: string[];
  headers?: Record<string, string>;
}

interface StreamingConfig {
  maxConnections: number;
  bufferSize: number;
  heartbeatInterval: number;
  maxMessageSize: number;
}

interface Connection {
  ws: WebSocket;
  url: string;
  createdAt: Date;
  lastActivity: Date;
}

interface StreamBuffer {
  type: string;
  chunks: any[];
  metadata: Record<string, any>;
  currentIndex: number;
  isComplete: boolean;
  createdAt: Date;
}

interface StreamChunk {
  content: string;
  index: number;
  timestamp: string;
}

interface ProcessedUpdate {
  id: string;
  timestamp: string;
  workerId: string;
  processed: boolean;
}

interface BroadcastResult {
  channel: string;
  status: "sent" | "connection_not_ready" | "error";
  error?: string;
}

interface StreamingStats {
  activeConnections: number;
  streamBuffers: number;
  maxConnections: number;
  workerId: string;
}

class StreamingWorker extends EventEmitter {
  public workerId: string;
  private connections: Map<string, Connection>;
  private streamBuffers: Map<string, StreamBuffer>;
  private config: StreamingConfig;

  constructor() {
    super();
    this.workerId = workerData?.workerId || "streaming-worker";
    this.connections = new Map();
    this.streamBuffers = new Map();
    this.config = {
      maxConnections: 100,
      bufferSize: 1000,
      heartbeatInterval: 30000,
      maxMessageSize: 10485760, // 10MB
    };

    console.log(`🌊 Streaming Worker ${this.workerId} initialized`);
    this.setupHeartbeat();
  }

  /**
   * Process incoming messages
   */
  handleMessage(message: WorkerMessage): void {
    const { taskId, data, options } = message;

    try {
      let result: any;

      switch (data.type) {
        case "process_stream":
          result = this.processStream(data.data, options);
          break;
        case "create_connection":
          result = this.createConnection(data.connectionData!, options);
          break;
        case "broadcast_message":
          result = this.broadcastMessage(data.message, data.channels!, options);
          break;
        case "stream_chunks":
          result = this.streamChunks(data.chunks!, data.connectionId!, options);
          break;
        case "manage_buffer":
          result = this.manageBuffer(
            data.bufferId!,
            data.operation!,
            data.bufferData
          );
          break;
        default:
          throw new Error(`Unknown streaming task type: ${data.type}`);
      }

      parentPort?.postMessage({
        taskId,
        success: true,
        data: result,
      });
    } catch (error) {
      console.error(`❌ Streaming error in ${this.workerId}:`, error);
      parentPort?.postMessage({
        taskId,
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Process streaming data
   */
  processStream(streamData: any, options: any = {}): any {
    const { type, content, metadata } = streamData;

    switch (type) {
      case "ai_response":
        return this.processAIResponseStream(content, metadata, options);
      case "document_chunks":
        return this.processDocumentChunks(content, metadata, options);
      case "real_time_updates":
        return this.processRealTimeUpdates(content, metadata, options);
      case "search_results":
        return this.processSearchResults(content, metadata, options);
      default:
        return this.processGenericStream(content, metadata, options);
    }
  }

  /**
   * Process AI response streaming
   */
  processAIResponseStream(content: string, metadata: Record<string, any>, options: any): any {
    const chunks = this.chunkAIResponse(content, options);
    const streamId = `ai_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    // Store stream buffer
    this.streamBuffers.set(streamId, {
      type: "ai_response",
      chunks,
      metadata,
      currentIndex: 0,
      isComplete: false,
      createdAt: new Date(),
    });

    return {
      streamId,
      totalChunks: chunks.length,
      firstChunk: chunks[0],
      metadata: {
        ...metadata,
        streamType: "ai_response",
        workerId: this.workerId,
      },
    };
  }

  /**
   * Chunk AI response for streaming
   */
  chunkAIResponse(content: string, options: any = {}): StreamChunk[] {
    const chunkSize = options.chunkSize || 50; // Characters per chunk
    const chunks: StreamChunk[] = [];

    // For AI responses, we want to chunk by words to avoid breaking mid-word
    const words = content.split(" ");
    let currentChunk = "";

    for (const word of words) {
      if (
        currentChunk.length + word.length + 1 > chunkSize &&
        currentChunk.length > 0
      ) {
        chunks.push({
          content: currentChunk.trim(),
          index: chunks.length,
          timestamp: new Date().toISOString(),
        });
        currentChunk = word;
      } else {
        currentChunk += (currentChunk ? " " : "") + word;
      }
    }

    // Add final chunk
    if (currentChunk.trim()) {
      chunks.push({
        content: currentChunk.trim(),
        index: chunks.length,
        timestamp: new Date().toISOString(),
      });
    }

    return chunks;
  }

  /**
   * Process document chunks for streaming
   */
  processDocumentChunks(chunks: any[], metadata: Record<string, any>, options: any): any {
    const streamId = `doc_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    // Process chunks for streaming
    const processedChunks = chunks.map((chunk: any, index: number) => ({
      ...chunk,
      streamIndex: index,
      processingTimestamp: new Date().toISOString(),
      workerId: this.workerId,
    }));

    this.streamBuffers.set(streamId, {
      type: "document_chunks",
      chunks: processedChunks,
      metadata,
      currentIndex: 0,
      isComplete: false,
      createdAt: new Date(),
    });

    return {
      streamId,
      totalChunks: processedChunks.length,
      chunkSummary: processedChunks.map((c: any) => ({
        index: c.streamIndex,
        size: c.content?.length || 0,
        type: c.type,
      })),
    };
  }

  /**
   * Process real-time updates
   */
  processRealTimeUpdates(updates: any, metadata: Record<string, any>, options: any): ProcessedUpdate[] {
    const processedUpdates = Array.isArray(updates) ? updates : [updates];

    return processedUpdates.map((update: any): ProcessedUpdate => ({
      ...update,
      id:
        update.id ||
        `update_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      workerId: this.workerId,
      processed: true,
    }));
  }

  /**
   * Process search results for streaming
   */
  processSearchResults(results: any[], metadata: Record<string, any>, options: any): any {
    const batchSize = options.batchSize || 10;
    const batches = [];

    for (let i = 0; i < results.length; i += batchSize) {
      batches.push({
        batch: Math.floor(i / batchSize),
        results: results.slice(i, i + batchSize),
        hasMore: i + batchSize < results.length,
      });
    }

    return {
      totalResults: results.length,
      totalBatches: batches.length,
      batches,
      streamingSupported: true,
    };
  }

  /**
   * Process generic streaming data
   */
  processGenericStream(content: any, metadata: Record<string, any>, options: any): any {
    return {
      processedContent: content,
      metadata: {
        ...metadata,
        processedAt: new Date().toISOString(),
        workerId: this.workerId,
      },
      streamingReady: true,
    };
  }

  /**
   * Create WebSocket connection
   */
  createConnection(connectionData: ConnectionData, options: any = {}): any {
    const { url, protocols, headers } = connectionData;
    const connectionId = `conn_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    try {
      const ws = new WebSocket(url, protocols, { headers });

      ws.on("open", () => {
        console.log(`✅ WebSocket connection ${connectionId} opened`);
        this.emit("connection_opened", { connectionId, url });
      });

      ws.on("message", (data: any) => {
        this.handleWebSocketMessage(connectionId, data);
      });

      ws.on("error", (error: Error) => {
        console.error(`❌ WebSocket error ${connectionId}:`, error);
        this.emit("connection_error", { connectionId, error: error.message });
      });

      ws.on("close", () => {
        console.log(`🔌 WebSocket connection ${connectionId} closed`);
        this.connections.delete(connectionId);
        this.emit("connection_closed", { connectionId });
      });

      this.connections.set(connectionId, {
        ws,
        url,
        createdAt: new Date(),
        lastActivity: new Date(),
      });

      return {
        connectionId,
        status: "connecting",
        url,
      };
    } catch (error) {
      throw new Error(
        `Failed to create WebSocket connection: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Handle WebSocket messages
   */
  handleWebSocketMessage(connectionId: string, data: any): void {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    connection.lastActivity = new Date();

    try {
      const message = JSON.parse(data.toString());
      this.emit("message_received", {
        connectionId,
        message,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      // Handle non-JSON messages
      this.emit("raw_message_received", {
        connectionId,
        data: data.toString(),
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Broadcast message to multiple channels
   */
  broadcastMessage(message: any, channels: string[], options: any = {}): any {
    const results: BroadcastResult[] = [];

    for (const channel of channels) {
      try {
        const connection = this.connections.get(channel);
        if (connection && connection.ws.readyState === WebSocket.OPEN) {
          connection.ws.send(JSON.stringify(message));
          results.push({ channel, status: "sent" });
        } else {
          results.push({ channel, status: "connection_not_ready" });
        }
      } catch (error) {
        results.push({
          channel,
          status: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    return {
      totalChannels: channels.length,
      successful: results.filter((r: BroadcastResult) => r.status === "sent").length,
      results,
    };
  }

  /**
   * Stream chunks to a connection
   */
  streamChunks(chunks: any[], connectionId: string, options: any = {}): any {
    const connection = this.connections.get(connectionId);
    if (!connection || connection.ws.readyState !== WebSocket.OPEN) {
      throw new Error(`Connection ${connectionId} not available`);
    }

    const delay = options.delay || 100; // ms between chunks
    let sentChunks = 0;

    const sendNext = (): void => {
      if (sentChunks < chunks.length) {
        const chunk = chunks[sentChunks];

        try {
          connection.ws.send(
            JSON.stringify({
              type: "chunk",
              index: sentChunks,
              data: chunk,
              hasMore: sentChunks < chunks.length - 1,
            })
          );

          sentChunks++;

          if (sentChunks < chunks.length) {
            setTimeout(sendNext, delay);
          }
        } catch (error) {
          throw new Error(
            `Failed to send chunk ${sentChunks}: ${error instanceof Error ? error.message : String(error)}`
          );
        }
      }
    };

    sendNext();

    return {
      connectionId,
      totalChunks: chunks.length,
      streamingStarted: true,
    };
  }

  /**
   * Manage stream buffers
   */
  manageBuffer(bufferId: string, operation: string, bufferData?: any): any {
    switch (operation) {
      case "get":
        return this.streamBuffers.get(bufferId) || null;
      case "delete":
        return this.streamBuffers.delete(bufferId);
      case "list":
        return Array.from(this.streamBuffers.keys());
      case "clear_old":
        return this.clearOldBuffers(bufferData?.maxAge || 3600000); // 1 hour
      default:
        throw new Error(`Unknown buffer operation: ${operation}`);
    }
  }

  /**
   * Clear old buffers
   */
  clearOldBuffers(maxAge: number): { cleared: number; remaining: number } {
    const now = new Date();
    let cleared = 0;

    for (const [bufferId, buffer] of this.streamBuffers) {
      if (now.getTime() - buffer.createdAt.getTime() > maxAge) {
        this.streamBuffers.delete(bufferId);
        cleared++;
      }
    }

    return { cleared, remaining: this.streamBuffers.size };
  }

  /**
   * Setup heartbeat for connection health
   */
  setupHeartbeat(): void {
    setInterval(() => {
      const now = new Date();

      for (const [connectionId, connection] of this.connections) {
        // Close stale connections
        if (now.getTime() - connection.lastActivity.getTime() > this.config.heartbeatInterval * 2) {
          console.log(`🔄 Closing stale connection ${connectionId}`);
          connection.ws.close();
          this.connections.delete(connectionId);
        } else if (connection.ws.readyState === WebSocket.OPEN) {
          // Send ping
          try {
            connection.ws.ping();
          } catch (error) {
            console.error(`❌ Ping failed for ${connectionId}:`, error);
          }
        }
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Get streaming statistics
   */
  getStats(): StreamingStats {
    return {
      activeConnections: this.connections.size,
      streamBuffers: this.streamBuffers.size,
      maxConnections: this.config.maxConnections,
      workerId: this.workerId,
    };
  }
}

// Initialize worker
const worker = new StreamingWorker();

// Handle messages from main thread
parentPort?.on("message", (message: WorkerMessage) => {
  worker.handleMessage(message);
});

// Forward worker events to main thread
worker.on("connection_opened", (data: any) => {
  parentPort?.postMessage({ type: "event", event: "connection_opened", data });
});

worker.on("connection_error", (data: any) => {
  parentPort?.postMessage({ type: "event", event: "connection_error", data });
});

worker.on("connection_closed", (data: any) => {
  parentPort?.postMessage({ type: "event", event: "connection_closed", data });
});

worker.on("message_received", (data: any) => {
  parentPort?.postMessage({ type: "event", event: "message_received", data });
});

// Send ready signal
parentPort?.postMessage({
  type: "ready",
  workerId: worker.workerId,
  stats: worker.getStats(),
});