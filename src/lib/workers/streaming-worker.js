import { parentPort, workerData } from "worker_threads";
import { WebSocket } from "ws";
import { EventEmitter } from "events";

/**
 * Phase 4: Streaming Worker
 * Handles real-time data streaming and WebSocket connections
 */

class StreamingWorker extends EventEmitter {
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
  handleMessage(message) {
    const { taskId, data, options } = message;

    try {
      let result;

      switch (data.type) {
        case "process_stream":
          result = this.processStream(data.data, options);
          break;
        case "create_connection":
          result = this.createConnection(data.connectionData, options);
          break;
        case "broadcast_message":
          result = this.broadcastMessage(data.message, data.channels, options);
          break;
        case "stream_chunks":
          result = this.streamChunks(data.chunks, data.connectionId, options);
          break;
        case "manage_buffer":
          result = this.manageBuffer(
            data.bufferId,
            data.operation,
            data.bufferData
          );
          break;
        default:
          throw new Error(`Unknown streaming task type: ${data.type}`);
      }

      parentPort.postMessage({
        taskId,
        success: true,
        data: result,
      });
    } catch (error) {
      console.error(`❌ Streaming error in ${this.workerId}:`, error);
      parentPort.postMessage({
        taskId,
        success: false,
        error: error.message,
      });
    }
  }

  /**
   * Process streaming data
   */
  processStream(streamData, options = {}) {
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
  processAIResponseStream(content, metadata, options) {
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
  chunkAIResponse(content, options = {}) {
    const chunkSize = options.chunkSize || 50; // Characters per chunk
    const chunks = [];

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
  processDocumentChunks(chunks, metadata, options) {
    const streamId = `doc_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    // Process chunks for streaming
    const processedChunks = chunks.map((chunk, index) => ({
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
      chunkSummary: processedChunks.map((c) => ({
        index: c.streamIndex,
        size: c.content?.length || 0,
        type: c.type,
      })),
    };
  }

  /**
   * Process real-time updates
   */
  processRealTimeUpdates(updates, metadata, options) {
    const processedUpdates = Array.isArray(updates) ? updates : [updates];

    return processedUpdates.map((update) => ({
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
  processSearchResults(results, metadata, options) {
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
  processGenericStream(content, metadata, options) {
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
  createConnection(connectionData, options = {}) {
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

      ws.on("message", (data) => {
        this.handleWebSocketMessage(connectionId, data);
      });

      ws.on("error", (error) => {
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
        `Failed to create WebSocket connection: ${error.message}`
      );
    }
  }

  /**
   * Handle WebSocket messages
   */
  handleWebSocketMessage(connectionId, data) {
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
  broadcastMessage(message, channels, options = {}) {
    const results = [];

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
          error: error.message,
        });
      }
    }

    return {
      totalChannels: channels.length,
      successful: results.filter((r) => r.status === "sent").length,
      results,
    };
  }

  /**
   * Stream chunks to a connection
   */
  streamChunks(chunks, connectionId, options = {}) {
    const connection = this.connections.get(connectionId);
    if (!connection || connection.ws.readyState !== WebSocket.OPEN) {
      throw new Error(`Connection ${connectionId} not available`);
    }

    const delay = options.delay || 100; // ms between chunks
    let sentChunks = 0;

    const sendNext = () => {
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
            `Failed to send chunk ${sentChunks}: ${error.message}`
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
  manageBuffer(bufferId, operation, bufferData) {
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
  clearOldBuffers(maxAge) {
    const now = new Date();
    let cleared = 0;

    for (const [bufferId, buffer] of this.streamBuffers) {
      if (now - buffer.createdAt > maxAge) {
        this.streamBuffers.delete(bufferId);
        cleared++;
      }
    }

    return { cleared, remaining: this.streamBuffers.size };
  }

  /**
   * Setup heartbeat for connection health
   */
  setupHeartbeat() {
    setInterval(() => {
      const now = new Date();

      for (const [connectionId, connection] of this.connections) {
        // Close stale connections
        if (now - connection.lastActivity > this.config.heartbeatInterval * 2) {
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
  getStats() {
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
parentPort.on("message", (message) => {
  worker.handleMessage(message);
});

// Forward worker events to main thread
worker.on("connection_opened", (data) => {
  parentPort.postMessage({ type: "event", event: "connection_opened", data });
});

worker.on("connection_error", (data) => {
  parentPort.postMessage({ type: "event", event: "connection_error", data });
});

worker.on("connection_closed", (data) => {
  parentPort.postMessage({ type: "event", event: "connection_closed", data });
});

worker.on("message_received", (data) => {
  parentPort.postMessage({ type: "event", event: "message_received", data });
});

// Send ready signal
parentPort.postMessage({
  type: "ready",
  workerId: worker.workerId,
  stats: worker.getStats(),
});
