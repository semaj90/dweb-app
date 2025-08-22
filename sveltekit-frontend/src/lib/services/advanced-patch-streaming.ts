import stream from "stream";
/**
 * Advanced JSON Patch Streaming Service
 * Real-time streaming with JSON Patch operations for live updates
 */

import type { Operation } from 'fast-json-patch';

interface StreamingPatchConfig {
  maxBufferSize: number; // Maximum buffer size before flush
  flushInterval: number; // Auto-flush interval in ms
  enableCompression: boolean; // Compress patches
  batchOperations: boolean; // Batch multiple operations
  retryAttempts: number; // Retry failed patches
  connectionTimeout: number; // WebSocket timeout
}

interface PatchStreamEvent {
  id: string;
  timestamp: number;
  patches: Operation[];
  target: string;
  metadata?: {
    source: string;
    version: number;
    checksum?: string;
  };
}

interface StreamingContext {
  connectionId: string;
  activeStreams: Map<string, ReadableStream>;
  patchBuffer: PatchStreamEvent[];
  lastFlush: number;
  metrics: {
    patchesSent: number;
    patchesReceived: number;
    bytesTransferred: number;
    connectionUptime: number;
  };
}

export class AdvancedPatchStreamer {
  private config: StreamingPatchConfig;
  private contexts: Map<string, StreamingContext> = new Map();
  private websocket: WebSocket | null = null;
  private flushTimer: number | null = null;

  constructor(config: Partial<StreamingPatchConfig> = {}) {
    this.config = {
      maxBufferSize: 100,
      flushInterval: 500,
      enableCompression: true,
      batchOperations: true,
      retryAttempts: 3,
      connectionTimeout: 30000,
      ...config,
    };
  }

  async createPatchStream(
    target: string,
    initialData: any,
    options: {
      contextId?: string;
      enableBidirectional?: boolean;
      customHeaders?: Record<string, string>;
    } = {}
  ): Promise<{ stream: ReadableStream; writer: WritableStreamDefaultWriter }> {
    const contextId =
      options.contextId || `stream_${Date.now()}_${Math.random().toString(36).slice(2)}`;

    // Initialize streaming context
    const context: StreamingContext = {
      connectionId: contextId,
      activeStreams: new Map(),
      patchBuffer: [],
      lastFlush: Date.now(),
      metrics: {
        patchesSent: 0,
        patchesReceived: 0,
        bytesTransferred: 0,
        connectionUptime: Date.now(),
      },
    };

    this.contexts.set(contextId, context);

    // Create readable stream for outgoing patches
    const { readable, writable } = new TransformStream<PatchStreamEvent, string>({
      transform: (chunk, controller) => {
        try {
          const serialized = this.serializePatchEvent(chunk);
          context.metrics.patchesSent++;
          context.metrics.bytesTransferred += serialized.length;
          controller.enqueue(serialized);
        } catch (error) {
          console.error('[PatchStreamer] Serialization failed:', error);
          controller.error(error);
        }
      },
    });

    context.activeStreams.set(target, readable);

    // Setup WebSocket connection if bidirectional
    if (options.enableBidirectional) {
      await this.setupWebSocketConnection(contextId, options.customHeaders);
    }

    // Start auto-flush timer
    this.startFlushTimer(contextId);

    // Send initial data as patch
    const writer = writable.getWriter();
    await this.sendInitialData(writer, target, initialData, contextId);

    return { stream: readable, writer };
  }

  async sendPatch(
    contextId: string,
    target: string,
    patches: Operation[],
    metadata?: unknown
  ): Promise<void> {
    const context = this.contexts.get(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    const event: PatchStreamEvent = {
      id: `patch_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      timestamp: Date.now(),
      patches: this.config.enableCompression ? this.compressPatches(patches) : patches,
      target,
      metadata: {
        source: 'client',
        version: context.metrics.patchesSent + 1,
        checksum: this.calculateChecksum(patches),
        ...metadata,
      },
    };

    // Add to buffer
    context.patchBuffer.push(event);

    // Flush if buffer is full
    if (context.patchBuffer.length >= this.config.maxBufferSize) {
      await this.flushBuffer(contextId);
    }
  }

  async streamRAGUpdates(
    query: string,
    documents: any[],
    contextId?: string
  ): Promise<ReadableStream<string>> {
    const streamContextId = contextId || `rag_${Date.now()}`;

    return new ReadableStream({
      async start(controller) {
        try {
          // Initial response structure
          let response = {
            query,
            status: 'processing',
            documents: [],
            summary: '',
            progress: 0,
            metadata: {
              startTime: Date.now(),
              documentsProcessed: 0,
              totalDocuments: documents.length,
            },
          };

          // Send initial state
          const initialPatches: Operation[] = [{ op: 'replace', path: '', value: response }];

          controller.enqueue(
            JSON.stringify({
              type: 'patch',
              patches: initialPatches,
              timestamp: Date.now(),
            }) + '\n'
          );

          // Process documents incrementally
          for (let i = 0; i < documents.length; i++) {
            const doc = documents[i];

            // Simulate document processing
            await new Promise((resolve) => setTimeout(resolve, 100));

            // Update progress
            const progressPatches: Operation[] = [
              { op: 'replace', path: '/progress', value: ((i + 1) / documents.length) * 100 },
              { op: 'replace', path: '/metadata/documentsProcessed', value: i + 1 },
              {
                op: 'add',
                path: `/documents/${i}`,
                value: {
                  id: doc.id,
                  title: doc.title,
                  relevanceScore: Math.random() * 0.8 + 0.2,
                  processedAt: Date.now(),
                },
              },
            ];

            controller.enqueue(
              JSON.stringify({
                type: 'patch',
                patches: progressPatches,
                timestamp: Date.now(),
              }) + '\n'
            );
          }

          // Generate summary (simulated)
          await new Promise((resolve) => setTimeout(resolve, 200));

          const summaryPatches: Operation[] = [
            { op: 'replace', path: '/status', value: 'completed' },
            {
              op: 'replace',
              path: '/summary',
              value: `Analysis of ${documents.length} documents completed. Key insights identified.`,
            },
            { op: 'replace', path: '/metadata/completedAt', value: Date.now() },
          ];

          controller.enqueue(
            JSON.stringify({
              type: 'patch',
              patches: summaryPatches,
              timestamp: Date.now(),
            }) + '\n'
          );

          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });
  }

  async streamDocumentAnalysis(
    documentId: string,
    options: {
      enableRealTime?: boolean;
      analysisTypes?: string[];
      contextId?: string;
    } = {}
  ): Promise<ReadableStream<string>> {
    const contextId = options.contextId || `analysis_${documentId}`;

    return new ReadableStream({
      async start(controller) {
        try {
          // Initial analysis state
          let analysis = {
            documentId,
            status: 'analyzing',
            progress: 0,
            results: {
              entities: [],
              sentiment: null,
              topics: [],
              legal_concepts: [],
              relationships: [],
            },
            metadata: {
              startTime: Date.now(),
              analysisTypes: options.analysisTypes || ['entities', 'sentiment', 'topics'],
            },
          };

          // Send initial state
          controller.enqueue(
            JSON.stringify({
              type: 'patch',
              patches: [{ op: 'replace', path: '', value: analysis }],
              timestamp: Date.now(),
            }) + '\n'
          );

          // Progressive analysis simulation
          const analysisSteps = [
            {
              type: 'entities',
              delay: 300,
              result: [
                { text: 'John Doe', type: 'PERSON', confidence: 0.95 },
                { text: 'Acme Corp', type: 'ORGANIZATION', confidence: 0.88 },
                { text: 'New York', type: 'LOCATION', confidence: 0.92 },
              ],
            },
            {
              type: 'sentiment',
              delay: 200,
              result: { score: 0.15, magnitude: 0.8, label: 'NEUTRAL' },
            },
            {
              type: 'topics',
              delay: 400,
              result: [
                { topic: 'contract law', confidence: 0.87 },
                { topic: 'liability', confidence: 0.73 },
                { topic: 'intellectual property', confidence: 0.65 },
              ],
            },
          ];

          for (let i = 0; i < analysisSteps.length; i++) {
            const step = analysisSteps[i];
            await new Promise((resolve) => setTimeout(resolve, step.delay));

            const patches: Operation[] = [
              { op: 'replace', path: '/progress', value: ((i + 1) / analysisSteps.length) * 100 },
              { op: 'replace', path: `/results/${step.type}`, value: step.result },
            ];

            controller.enqueue(
              JSON.stringify({
                type: 'patch',
                patches,
                timestamp: Date.now(),
              }) + '\n'
            );
          }

          // Complete analysis
          const completionPatches: Operation[] = [
            { op: 'replace', path: '/status', value: 'completed' },
            { op: 'replace', path: '/metadata/completedAt', value: Date.now() },
          ];

          controller.enqueue(
            JSON.stringify({
              type: 'patch',
              patches: completionPatches,
              timestamp: Date.now(),
            }) + '\n'
          );

          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });
  }

  private async setupWebSocketConnection(
    contextId: string,
    customHeaders?: Record<string, string>
  ): Promise<void> {
    try {
      const wsUrl = this.buildWebSocketURL(contextId);
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        console.log(`[PatchStreamer] WebSocket connected for context: ${contextId}`);
      };

      this.websocket.onmessage = (event) => {
        this.handleIncomingPatch(contextId, event.data);
      };

      this.websocket.onerror = (error) => {
        console.error(`[PatchStreamer] WebSocket error for context ${contextId}:`, error);
      };

      this.websocket.onclose = () => {
        console.log(`[PatchStreamer] WebSocket closed for context: ${contextId}`);
        // Attempt reconnection
        setTimeout(() => this.setupWebSocketConnection(contextId, customHeaders), 5000);
      };
    } catch (error) {
      console.error('[PatchStreamer] WebSocket setup failed:', error);
    }
  }

  private async sendInitialData(
    writer: WritableStreamDefaultWriter,
    target: string,
    data: any,
    contextId: string
  ): Promise<void> {
    const initialEvent: PatchStreamEvent = {
      id: `init_${Date.now()}`,
      timestamp: Date.now(),
      patches: [{ op: 'replace', path: '', value: data }],
      target,
      metadata: {
        source: 'server',
        version: 1,
        checksum: this.calculateChecksum([{ op: 'replace', path: '', value: data }]),
      },
    };

    await writer.write(initialEvent);
  }

  private serializePatchEvent(event: PatchStreamEvent): string {
    return JSON.stringify(event) + '\n';
  }

  private compressPatches(patches: Operation[]): Operation[] {
    // Simple compression: merge consecutive operations on same path
    const compressed: Operation[] = [];

    for (const patch of patches) {
      const existing = compressed.find((p) => p.path === patch.path && p.op === patch.op);

      if (existing && existing.op === 'replace') {
        if ('value' in patch && 'value' in existing) {
          // Both operations expose value (Add/Replace). Safe to assign.
          (existing as any).value = (patch as any).value;
        }
      } else {
        compressed.push({ ...patch });
      }
    }

    return compressed;
  }

  private calculateChecksum(patches: Operation[]): string {
    const str = JSON.stringify(patches);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(16);
  }

  private startFlushTimer(contextId: string): void {
    if (this.flushTimer) clearInterval(this.flushTimer);

    this.flushTimer = setInterval(() => {
      const context = this.contexts.get(contextId);
      if (context && context.patchBuffer.length > 0) {
        const timeSinceLastFlush = Date.now() - context.lastFlush;
        if (timeSinceLastFlush >= this.config.flushInterval) {
          this.flushBuffer(contextId);
        }
      }
    }, this.config.flushInterval / 2) as unknown as number;
  }

  private async flushBuffer(contextId: string): Promise<void> {
    const context = this.contexts.get(contextId);
    if (!context || context.patchBuffer.length === 0) return;

    try {
      // Send batched patches via WebSocket if available
      if (this.websocket?.readyState === WebSocket.OPEN) {
        const batchedEvent = {
          type: 'patch_batch',
          contextId,
          events: context.patchBuffer,
          timestamp: Date.now(),
        };

        this.websocket.send(JSON.stringify(batchedEvent));
      }

      // Clear buffer and update metrics
      context.patchBuffer = [];
      context.lastFlush = Date.now();
    } catch (error) {
      console.error(`[PatchStreamer] Failed to flush buffer for context ${contextId}:`, error);
    }
  }

  private handleIncomingPatch(contextId: string, data: string): void {
    try {
      const event = JSON.parse(data);
      const context = this.contexts.get(contextId);

      if (context) {
        context.metrics.patchesReceived++;
        // Handle incoming patch application logic here
        console.log(`[PatchStreamer] Received patch for context ${contextId}:`, event);
      }
    } catch (error) {
      console.error(`[PatchStreamer] Failed to handle incoming patch:`, error);
    }
  }

  private buildWebSocketURL(contextId: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/api/ws/patches/${contextId}`;
  }

  // Cleanup
  async closeContext(contextId: string): Promise<void> {
    const context = this.contexts.get(contextId);
    if (!context) return;

    // Close all active streams
    for (const [target, stream] of context.activeStreams) {
      try {
        await stream.cancel();
      } catch (error) {
        console.warn(`[PatchStreamer] Failed to close stream for target ${target}:`, error);
      }
    }

    // Close WebSocket
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.close();
    }

    // Clear context
    this.contexts.delete(contextId);
  }

  getMetrics(contextId: string): StreamingContext['metrics'] | null {
    return this.contexts.get(contextId)?.metrics || null;
  }
}

// Convenience functions
export async function createRAGPatchStream(
  query: string,
  documents: any[],
  options?: { contextId?: string }
): Promise<ReadableStream<string>> {
  const streamer = new AdvancedPatchStreamer();
  return streamer.streamRAGUpdates(query, documents, options?.contextId);
}

export async function createDocumentAnalysisStream(
  documentId: string,
  options?: {
    enableRealTime?: boolean;
    analysisTypes?: string[];
    contextId?: string;
  }
): Promise<ReadableStream<string>> {
  const streamer = new AdvancedPatchStreamer();
  return streamer.streamDocumentAnalysis(documentId, options);
}

// Integration test helper
export async function testAdvancedPatchStreaming(): Promise<boolean> {
  try {
    const streamer = new AdvancedPatchStreamer({
      maxBufferSize: 5,
      flushInterval: 100,
      enableCompression: true,
    });

    // Test RAG streaming
    const ragStream = await streamer.streamRAGUpdates('test query', [
      { id: 'doc1', title: 'Test Doc 1' },
      { id: 'doc2', title: 'Test Doc 2' },
    ]);

    const reader = ragStream.getReader();
    let patchCount = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Validate patch format
        const patch = JSON.parse(value);
        if (patch.type === 'patch' && Array.isArray(patch.patches)) {
          patchCount++;
        }
      }
    } finally {
      reader.releaseLock();
    }

    const isValid = patchCount > 0;

    console.log('[test] Advanced patch streaming:', isValid ? 'PASS' : 'FAIL');
    console.log('[test] Patches received:', patchCount);

    return isValid;
  } catch (error) {
    console.error('[test] Advanced patch streaming failed:', error);
    return false;
  }
}
