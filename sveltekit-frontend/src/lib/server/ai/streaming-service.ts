import { EventEmitter } from "events";

// lib/server/ai/streaming-service.ts
// Real-time streaming service for AI synthesis with progressive updates

import { logger } from './logger';
import { aiAssistantSynthesizer } from './ai-assistant-input-synthesizer';

export interface StreamEvent {
  type: 'status' | 'progress' | 'stage' | 'source' | 'complete' | 'error' | 'heartbeat';
  data: any;
}

export interface StreamSubscriber {
  callback: (event: StreamEvent) => void;
  subscribed: number;
}

export interface StreamingOptions {
  input: {
    query: string;
    context?: unknown;
    options?: unknown;
  };
  onProgress?: (stage: string, progress: number, data?: unknown) => void;
  onStage?: (stage: string, data: any) => void;
  onSource?: (source: any) => void;
  onComplete?: (result: any) => void;
  onError?: (error: Error) => void;
}

class StreamingService extends EventEmitter {
  private streams: Map<string, StreamSubscriber[]> = new Map();
  private activeProcessing: Map<string, any> = new Map();
  private streamBuffer: Map<string, StreamEvent[]> = new Map();
  private progressTracking: Map<string, any> = new Map();
  
  constructor() {
    super();
    this.initialize();
  }

  private initialize(): void {
    logger.info('[StreamingService] Initializing streaming service...');
    
    // Cleanup inactive streams periodically
    setInterval(() => this.cleanupInactiveStreams(), 60000); // Every minute
    
    logger.info('[StreamingService] Streaming service initialized');
  }

  /**
   * Subscribe to a stream
   */
  subscribe(streamId: string, callback: (event: StreamEvent) => void): () => void {
    if (!this.streams.has(streamId)) {
      this.streams.set(streamId, []);
    }
    
    const subscriber: StreamSubscriber = {
      callback,
      subscribed: Date.now()
    };
    
    this.streams.get(streamId).push(subscriber);
    
    // Send any buffered events
    const buffer = this.streamBuffer.get(streamId);
    if (buffer) {
      for (const event of buffer) {
        callback(event);
      }
      this.streamBuffer.delete(streamId);
    }
    
    logger.debug(`[StreamingService] Subscriber added to stream ${streamId}`);
    
    // Return unsubscribe function
    return () => {
      const subscribers = this.streams.get(streamId);
      if (subscribers) {
        const index = subscribers.indexOf(subscriber);
        if (index > -1) {
          subscribers.splice(index, 1);
        }
        
        if (subscribers.length === 0) {
          this.streams.delete(streamId);
          this.streamBuffer.delete(streamId);
        }
      }
      
      logger.debug(`[StreamingService] Subscriber removed from stream ${streamId}`);
    };
  }

  /**
   * Synthesize with progressive streaming updates
   */
  async synthesizeWithProgress(options: StreamingOptions): Promise<any> {
    const streamId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      logger.info(`[StreamingService] Starting progressive synthesis for stream ${streamId}`);
      
      // Track processing state
      this.activeProcessing.set(streamId, {
        startTime: Date.now(),
        status: 'processing',
        progress: 0,
        currentStage: 'initialization'
      });
      
      // Initialize progress tracking
      this.progressTracking.set(streamId, {
        stages: {
          query_analysis: { progress: 0, complete: false },
          retrieval: { progress: 0, complete: false },
          ranking: { progress: 0, complete: false },
          prompt_construction: { progress: 0, complete: false },
          quality_assessment: { progress: 0, complete: false }
        },
        sources: [],
        totalProgress: 0
      });
      
      // Stage 1: Query Analysis (0-20%)
      await this.processStage(streamId, 'query_analysis', async () => {
        options.onStage?.('query_analysis', { status: 'starting' });
        
        const result = await this.simulateQueryAnalysis(options.input.query);
        
        options.onProgress?.('query_analysis', 100, result);
        options.onStage?.('query_analysis', { status: 'complete', result });
        
        return result;
      }, 0, 20);
      
      // Stage 2: Multi-Strategy Retrieval (20-50%)
      const sources = await this.processStage(streamId, 'retrieval', async () => {
        options.onStage?.('retrieval', { status: 'starting' });
        
        const sources = await this.streamRetrieval(
          options.input,
          (source, index, total) => {
            const progress = (index / total) * 100;
            options.onProgress?.('retrieval', progress, { source, index, total });
            options.onSource?.(source);
            
            // Update progress tracking
            const tracking = this.progressTracking.get(streamId);
            if (tracking) {
              tracking.sources.push(source);
            }
          }
        );
        
        options.onStage?.('retrieval', { 
          status: 'complete', 
          sourceCount: sources.length 
        });
        
        return sources;
      }, 20, 50);
      
      // Stage 3: Ranking and Processing (50-70%)
      const rankedSources = await this.processStage(streamId, 'ranking', async () => {
        options.onStage?.('ranking', { status: 'starting' });
        
        const ranked = await this.streamRanking(sources, (progress) => {
          options.onProgress?.('ranking', progress);
        });
        
        options.onStage?.('ranking', { 
          status: 'complete',
          topSources: ranked.slice(0, 3).map(s => (s as any)?.title || 'Unknown')
        });
        
        return ranked;
      }, 50, 70);
      
      // Stage 4: Prompt Construction (70-85%)
      const prompt = await this.processStage(streamId, 'prompt_construction', async () => {
        options.onStage?.('prompt_construction', { status: 'starting' });
        
        const prompt = await this.constructPromptWithProgress(
          options.input,
          rankedSources,
          (progress) => {
            options.onProgress?.('prompt_construction', progress);
          }
        );
        
        options.onStage?.('prompt_construction', { 
          status: 'complete',
          promptLength: prompt.length
        });
        
        return prompt;
      }, 70, 85);
      
      // Stage 5: Quality Assessment (85-100%)
      const finalResult = await this.processStage(streamId, 'quality_assessment', async () => {
        options.onStage?.('quality_assessment', { status: 'starting' });
        
        // Actually call the synthesizer for the complete result
        const result = await aiAssistantSynthesizer.synthesizeInput({
          query: options.input.query,
          context: { userId: '', ...((options.input.context || {}) as Record<string, any>) },
          options: {
            enableMMR: true,
            enableCrossEncoder: true,
            enableLegalBERT: true,
            enableRAG: true,
            maxSources: 5,
            similarityThreshold: 0.7,
            diversityLambda: 0.3,
            ...((options.input.options || {}) as Record<string, any>)
          }
        });
        
        options.onProgress?.('quality_assessment', 100, {
          confidence: result.metadata.confidence,
          qualityScore: result.metadata.qualityScore
        });
        
        options.onStage?.('quality_assessment', { 
          status: 'complete',
          metrics: {
            confidence: result.metadata.confidence,
            qualityScore: result.metadata.qualityScore,
            sourceCount: result.retrievedContext.sources.length
          }
        });
        
        return result;
      }, 85, 100);
      
      // Mark processing as complete
      const processing = this.activeProcessing.get(streamId);
      if (processing) {
        processing.status = 'complete';
        processing.progress = 100;
        processing.endTime = Date.now();
        processing.duration = processing.endTime - processing.startTime;
      }
      
      // Call completion callback
      options.onComplete?.(finalResult);
      
      logger.info(`[StreamingService] Completed progressive synthesis for stream ${streamId}`);
      
      return finalResult;
      
    } catch (error: any) {
      logger.error(`[StreamingService] Progressive synthesis failed for stream ${streamId}:`, error);
      
      // Mark processing as failed
      const processing = this.activeProcessing.get(streamId);
      if (processing) {
        processing.status = 'error';
        processing.error = error.message;
      }
      
      // Call error callback
      options.onError?.(error);
      
      throw error;
      
    } finally {
      // Cleanup after delay
      setTimeout(() => {
        this.activeProcessing.delete(streamId);
        this.progressTracking.delete(streamId);
      }, 60000); // Keep for 1 minute for late subscribers
    }
  }

  /**
   * Send event to stream subscribers
   */
  private sendEvent(streamId: string, event: StreamEvent): void {
    const subscribers = this.streams.get(streamId);
    
    if (subscribers && subscribers.length > 0) {
      for (const subscriber of subscribers) {
        try {
          subscriber.callback(event);
        } catch (error: any) {
          logger.error(`[StreamingService] Failed to send event to subscriber:`, error);
        }
      }
    } else {
      // Buffer events if no subscribers yet
      if (!this.streamBuffer.has(streamId)) {
        this.streamBuffer.set(streamId, []);
      }
      
      const buffer = this.streamBuffer.get(streamId);
      buffer.push(event);
      
      // Limit buffer size
      if (buffer.length > 100) {
        buffer.shift();
      }
    }
  }

  /**
   * Process a stage with progress tracking
   */
  private async processStage(
    streamId: string,
    stageName: string,
    processor: () => Promise<any>,
    startProgress: number,
    endProgress: number
  ): Promise<any> {
    const processing = this.activeProcessing.get(streamId);
    if (processing) {
      processing.currentStage = stageName;
      processing.progress = startProgress;
    }
    
    const tracking = this.progressTracking.get(streamId);
    if (tracking) {
      tracking.stages[stageName].progress = 0;
      tracking.totalProgress = startProgress;
    }
    
    try {
      // Execute the stage processor
      const result = await processor();
      
      // Update completion status
      if (tracking) {
        tracking.stages[stageName].progress = 100;
        tracking.stages[stageName].complete = true;
        tracking.totalProgress = endProgress;
      }
      
      if (processing) {
        processing.progress = endProgress;
      }
      
      return result;
      
    } catch (error: any) {
      logger.error(`[StreamingService] Stage ${stageName} failed:`, error);
      
      if (tracking) {
        tracking.stages[stageName].error = error.message;
      }
      
      throw error;
    }
  }

  /**
   * Simulate query analysis with progress
   */
  private async simulateQueryAnalysis(query: string): Promise<any> {
    // Simulate processing time
    await this.delay(500);
    
    return {
      original: query,
      enhanced: query + ' [enhanced]',
      intent: 'legal_query',
      entities: [],
      complexity: 0.7
    };
  }

  /**
   * Stream retrieval with source-by-source updates
   */
  private async streamRetrieval(
    input: any,
    onSource: (source: any, index: number, total: number) => void
  ): Promise<any[]> {
    const sources = [];
    const totalSources = 10; // Simulate finding 10 sources
    
    for (let i = 0; i < totalSources; i++) {
      // Simulate retrieval delay
      await this.delay(200);
      
      const source = {
        id: `source_${i}`,
        title: `Legal Document ${i + 1}`,
        content: `Content of document ${i + 1}...`,
        relevanceScore: Math.random(),
        type: 'document'
      };
      
      sources.push(source);
      onSource(source, i + 1, totalSources);
    }
    
    return sources;
  }

  /**
   * Stream ranking with progress updates
   */
  private async streamRanking(
    sources: any[],
    onProgress: (progress: number) => void
  ): Promise<any[]> {
    const steps = 5;
    
    for (let i = 0; i < steps; i++) {
      await this.delay(300);
      onProgress((i + 1) / steps * 100);
    }
    
    // Sort by relevance
    return sources.sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  /**
   * Construct prompt with progress updates
   */
  private async constructPromptWithProgress(
    input: any,
    sources: any[],
    onProgress: (progress: number) => void
  ): Promise<string> {
    const steps = 3;
    let prompt = '';
    
    for (let i = 0; i < steps; i++) {
      await this.delay(200);
      
      if (i === 0) {
        prompt += 'System: You are a legal AI assistant.\n';
      } else if (i === 1) {
        prompt += `Context: ${sources.slice(0, 3).map(s => s.title).join(', ')}\n`;
      } else {
        prompt += `Query: ${input.query}\n`;
      }
      
      onProgress((i + 1) / steps * 100);
    }
    
    return prompt;
  }

  /**
   * Get stream status
   */
  getStreamStatus(streamId: string): unknown {
    const processing = this.activeProcessing.get(streamId);
    const tracking = this.progressTracking.get(streamId);
    const subscribers = this.streams.get(streamId);
    
    return {
      exists: !!processing,
      status: processing?.status || 'unknown',
      progress: processing?.progress || 0,
      currentStage: processing?.currentStage,
      stages: tracking?.stages,
      sources: tracking?.sources?.length || 0,
      subscribers: subscribers?.length || 0,
      startTime: processing?.startTime,
      duration: processing?.duration
    };
  }

  /**
   * Get all active streams
   */
  getActiveStreams(): unknown[] {
    const streams = [];
    
    for (const [streamId, processing] of Array.from(this.activeProcessing.entries())) {
      streams.push({
        streamId,
        status: processing.status,
        progress: processing.progress,
        currentStage: processing.currentStage,
        startTime: processing.startTime,
        subscribers: this.streams.get(streamId)?.length || 0
      });
    }
    
    return streams;
  }

  /**
   * Clean up inactive streams
   */
  private cleanupInactiveStreams(): void {
    const now = Date.now();
    const maxAge = 5 * 60 * 1000; // 5 minutes
    
    // Clean up old processing records
    for (const [streamId, processing] of Array.from(this.activeProcessing.entries())) {
      if (processing.endTime && (now - processing.endTime > maxAge)) {
        this.activeProcessing.delete(streamId);
        this.progressTracking.delete(streamId);
        logger.debug(`[StreamingService] Cleaned up old stream ${streamId}`);
      }
    }
    
    // Clean up orphaned buffers
    for (const [streamId, buffer] of Array.from(this.streamBuffer.entries())) {
      if (!this.streams.has(streamId) && buffer.length > 0) {
        const lastEvent = buffer[buffer.length - 1];
        if (lastEvent.type === 'complete' || lastEvent.type === 'error') {
          this.streamBuffer.delete(streamId);
        }
      }
    }
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Shutdown streaming service
   */
  async shutdown(): Promise<void> {
    // Send closing events to all active streams
    for (const [streamId, subscribers] of Array.from(this.streams.entries())) {
      const event: StreamEvent = {
        type: 'error',
        data: { message: 'Service shutting down' }
      };
      
      for (const subscriber of subscribers) {
        try {
          subscriber.callback(event);
        } catch (error: any) {
          // Ignore errors during shutdown
        }
      }
    }
    
    // Clear all data
    this.streams.clear();
    this.activeProcessing.clear();
    this.progressTracking.clear();
    this.streamBuffer.clear();
    
    logger.info('[StreamingService] Streaming service shutdown complete');
  }
}

// Export singleton instance
export const streamingService = new StreamingService();
;
// Support for Ollama local LLM integration
export class OllamaStreamingAdapter {
  private ollamaUrl: string;
  
  constructor(ollamaUrl: string = 'http://localhost:11434') {
    this.ollamaUrl = ollamaUrl;
  }

  /**
   * Stream from Ollama with progressive updates
   */
  async streamFromOllama(
    model: string,
    prompt: string,
    onToken: (token: string) => void,
    onComplete: (response: string) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama request failed: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.response) {
              fullResponse += data.response;
              onToken(data.response);
            }
            if (data.done) {
              onComplete(fullResponse);
            }
          } catch (e: any) {
            // Ignore parsing errors
          }
        }
      }
    } catch (error: any) {
      logger.error('[OllamaStreamingAdapter] Streaming failed:', error);
      throw error;
    }
  }

  /**
   * Check if Ollama is available
   */
  async checkAvailability(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      return response.ok;
    } catch (error: any) {
      return false;
    }
  }
}

// Export Ollama adapter
export const ollamaAdapter = new OllamaStreamingAdapter();
;
// Types are already exported as interfaces above

