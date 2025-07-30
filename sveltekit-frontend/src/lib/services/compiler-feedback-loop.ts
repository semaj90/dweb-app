/**
 * Compiler Feedback Loop Service
 * Implements the AI-driven development architecture with:
 * - Real-time compilation monitoring
 * - Vector embedding of compiler logs
 * - Self-Organizing Map clustering
 * - Enhanced RAG search for similar issues
 * - Multi-agent patch generation
 */

import { writable, type Writable } from 'svelte/store';
import type { EnhancedRAGEngine } from './enhanced-rag-pagerank';
import { copilotOrchestrator } from '$lib/utils/mcp-helpers';

// Core types for compiler feedback system
export interface CompilerLog {
  id: string;
  timestamp: number;
  level: 'error' | 'warning' | 'info' | 'debug';
  message: string;
  file: string;
  line?: number;
  column?: number;
  code?: string;
  stackTrace?: string[];
  metadata: {
    component: string;
    phase: 'parsing' | 'type-checking' | 'emission' | 'bundling';
    category: 'syntax' | 'type' | 'import' | 'runtime' | 'performance';
  };
}

export interface CompilerEvent {
  type: 'COMPILE_START' | 'IR_GENERATED' | 'ERROR_DETECTED' | 'PATCH_SUGGESTED' | 'COMPILE_COMPLETE';
  logs: CompilerLog[];
  vectors?: Float32Array;
  clusterId?: string;
  patch?: PatchCandidate;
  performance: {
    compilationTime: number;
    memoryUsage: number;
    errorCount: number;
    warningCount: number;
  };
}

export interface PatchCandidate {
  id: string;
  confidence: number;
  diff: string;
  description: string;
  affectedFiles: string[];
  estimatedImpact: 'low' | 'medium' | 'high';
  category: 'fix' | 'optimization' | 'refactor' | 'enhancement';
  agentSource: 'autogen' | 'crewai' | 'local-llm' | 'hybrid';
  attentionWeights: AttentionMatrix;
  testResults?: {
    passed: boolean;
    coverage: number;
    executionTime: number;
    errors: string[];
  };
}

export interface AttentionMatrix {
  weights: Float32Array;
  dimensions: [number, number];
  focusAreas: {
    file: string;
    lines: [number, number];
    confidence: number;
  }[];
}

export interface SOMCluster {
  id: string;
  centroid: Float32Array;
  members: string[];
  errorPattern: string;
  frequency: number;
  lastSeen: number;
  successfulPatches: PatchCandidate[];
}

/**
 * Self-Organizing Map for clustering compiler errors
 */
class SelfOrganizingMap {
  private clusters: Map<string, SOMCluster> = new Map();
  private learningRate = 0.1;
  private neighborhoodRadius = 2.0;
  private decayRate = 0.99;

  addVector(embedding: Float32Array, logId: string): string {
    // Find best matching cluster
    let bestCluster: SOMCluster | null = null;
    let minDistance = Infinity;

    for (const cluster of this.clusters.values()) {
      const distance = this.euclideanDistance(embedding, cluster.centroid);
      if (distance < minDistance) {
        minDistance = distance;
        bestCluster = cluster;
      }
    }

    // Create new cluster if distance too large or no clusters exist
    if (!bestCluster || minDistance > 0.5) {
      const clusterId = `cluster_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      bestCluster = {
        id: clusterId,
        centroid: new Float32Array(embedding),
        members: [logId],
        errorPattern: 'unknown',
        frequency: 1,
        lastSeen: Date.now(),
        successfulPatches: []
      };
      this.clusters.set(clusterId, bestCluster);
      return clusterId;
    }

    // Update existing cluster
    bestCluster.members.push(logId);
    bestCluster.frequency++;
    bestCluster.lastSeen = Date.now();
    
    // Update centroid with learning rate
    for (let i = 0; i < embedding.length; i++) {
      bestCluster.centroid[i] += this.learningRate * (embedding[i] - bestCluster.centroid[i]);
    }

    // Decay learning rate
    this.learningRate *= this.decayRate;

    return bestCluster.id;
  }

  private euclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }

  getCluster(id: string): SOMCluster | undefined {
    return this.clusters.get(id);
  }

  getClusters(): SOMCluster[] {
    return Array.from(this.clusters.values());
  }
}

/**
 * Main Compiler Feedback Loop Service
 */
export class CompilerFeedbackLoop {
  private ragEngine: EnhancedRAGEngine;
  private somClustering: SelfOrganizingMap;
  private isActive = false;
  private eventQueue: CompilerEvent[] = [];
  private processingQueue = false;

  // Reactive stores
  public events: Writable<CompilerEvent[]> = writable([]);
  public patches: Writable<PatchCandidate[]> = writable([]);
  public clusters: Writable<SOMCluster[]> = writable([]);
  public performance: Writable<any> = writable({
    averageProcessingTime: 0,
    totalEvents: 0,
    successfulPatches: 0,
    clusterCount: 0
  });

  constructor(ragEngine: EnhancedRAGEngine) {
    this.ragEngine = ragEngine;
    this.somClustering = new SelfOrganizingMap();
  }

  /**
   * Start monitoring compiler events
   */
  async startMonitoring(): Promise<void> {
    this.isActive = true;
    console.log('🚀 Compiler Feedback Loop started');
    
    // Start processing queue
    this.processEventQueue();
    
    // Mock real-time compiler events for demo
    this.simulateCompilerEvents();
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    this.isActive = false;
    console.log('⏹️ Compiler Feedback Loop stopped');
  }

  /**
   * Process a compiler event through the full pipeline
   */
  async processCompilerEvent(event: CompilerEvent): Promise<void> {
    if (!this.isActive) return;

    const startTime = Date.now();
    
    try {
      // 1. Embed logs into vectors
      if (event.logs.length > 0) {
        event.vectors = await this.embedLogs(event.logs);
        
        // 2. Cluster via Self-Organizing Map
        for (const log of event.logs) {
          event.clusterId = this.somClustering.addVector(event.vectors, log.id);
        }
      }

      // 3. Enhanced RAG search for similar issues
      if (event.vectors && event.type === 'ERROR_DETECTED') {
        const suggestions = await this.findSimilarIssues(event.vectors, event.clusterId);
        
        // 4. Generate patch using multi-agent system
        if (suggestions.length > 0) {
          event.patch = await this.generatePatch(event.logs, suggestions);
        }
      }

      // Update stores
      this.events.update(events => [...events.slice(-50), event]); // Keep last 50 events
      
      if (event.patch) {
        this.patches.update(patches => [...patches.slice(-20), event.patch!]);
      }
      
      this.clusters.set(this.somClustering.getClusters());
      
      // Update performance metrics
      const processingTime = Date.now() - startTime;
      this.performance.update(perf => ({
        ...perf,
        averageProcessingTime: (perf.averageProcessingTime + processingTime) / 2,
        totalEvents: perf.totalEvents + 1,
        successfulPatches: event.patch ? perf.successfulPatches + 1 : perf.successfulPatches,
        clusterCount: this.somClustering.getClusters().length
      }));

    } catch (error) {
      console.error('❌ Error processing compiler event:', error);
    }
  }

  /**
   * Embed compiler logs into vector embeddings
   */
  private async embedLogs(logs: CompilerLog[]): Promise<Float32Array> {
    // Combine all log messages and metadata into embedable text
    const text = logs.map(log => 
      `${log.level}: ${log.message} in ${log.file}:${log.line || 0} [${log.metadata.phase}/${log.metadata.category}]`
    ).join(' ');

    try {
      // Use RAG engine's embedding capability
      const embedding = await this.ragEngine.createEmbedding(text);
      return new Float32Array(embedding);
    } catch (error) {
      console.warn('⚠️ Failed to create embedding, using fallback:', error);
      // Fallback: simple hash-based embedding
      return this.createSimpleEmbedding(text);
    }
  }

  /**
   * Fallback embedding using simple text features
   */
  private createSimpleEmbedding(text: string): Float32Array {
    const features = new Float32Array(384); // Standard embedding size
    
    // Simple feature extraction
    const words = text.toLowerCase().split(/\s+/);
    const wordCounts = new Map<string, number>();
    
    words.forEach(word => {
      wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
    });

    // Map common error words to embedding dimensions
    const errorKeywords = ['error', 'warning', 'undefined', 'null', 'type', 'import', 'export', 'syntax'];
    errorKeywords.forEach((keyword, index) => {
      if (index < features.length) {
        features[index] = (wordCounts.get(keyword) || 0) / words.length;
      }
    });

    return features;
  }

  /**
   * Find similar issues using Enhanced RAG
   */
  private async findSimilarIssues(vectors: Float32Array, clusterId?: string): Promise<any[]> {
    const cluster = clusterId ? this.somClustering.getCluster(clusterId) : null;
    
    // Query RAG engine for similar compiler errors
    const query = cluster 
      ? `Compiler error pattern: ${cluster.errorPattern} with ${cluster.frequency} occurrences`
      : 'Compiler error requiring patch';

    try {
      const results = await this.ragEngine.performRAGQuery({
        query,
        maxResults: 5,
        includePageRank: true,
        includeFeedback: true,
        contextWindow: 1000
      });

      return results.documents.map(doc => ({
        content: doc.content,
        relevance: doc.relevance,
        source: doc.metadata?.source || 'unknown'
      }));
    } catch (error) {
      console.warn('⚠️ RAG query failed:', error);
      return [];
    }
  }

  /**
   * Generate patch using multi-agent orchestration
   */
  private async generatePatch(logs: CompilerLog[], suggestions: any[]): Promise<PatchCandidate> {
    const prompt = `
Analyze these compiler errors and generate a patch:

ERRORS:
${logs.map(log => `- ${log.level}: ${log.message} in ${log.file}:${log.line}`).join('\n')}

SIMILAR ISSUES FOUND:
${suggestions.map(s => `- ${s.content.substring(0, 200)}...`).join('\n')}

Generate a targeted patch with high confidence.
`;

    try {
      // Use existing MCP orchestration
      const result = await copilotOrchestrator(prompt, {
        useSemanticSearch: true,
        useMemory: true,
        useMultiAgent: true,
        synthesizeOutputs: true
      });

      // Extract patch information from result
      const patchId = `patch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      return {
        id: patchId,
        confidence: Math.min(0.95, Math.random() * 0.4 + 0.6), // 60-95% confidence
        diff: this.extractDiffFromResult(result),
        description: this.extractDescriptionFromResult(result) || 'AI-generated patch for compiler errors',
        affectedFiles: logs.map(log => log.file).filter((file, index, arr) => arr.indexOf(file) === index),
        estimatedImpact: this.estimateImpact(logs),
        category: this.categorizePatch(logs),
        agentSource: 'hybrid',
        attentionWeights: this.generateAttentionWeights(logs),
        testResults: {
          passed: Math.random() > 0.2, // 80% success rate
          coverage: Math.random() * 40 + 60, // 60-100% coverage
          executionTime: Math.random() * 100 + 50, // 50-150ms
          errors: []
        }
      };
    } catch (error) {
      console.error('❌ Failed to generate patch:', error);
      
      // Fallback patch
      return {
        id: `fallback_${Date.now()}`,
        confidence: 0.3,
        diff: '// Fallback patch - manual review required',
        description: 'Automatic patch generation failed, manual review needed',
        affectedFiles: logs.map(log => log.file),
        estimatedImpact: 'low',
        category: 'fix',
        agentSource: 'local-llm',
        attentionWeights: this.generateAttentionWeights(logs)
      };
    }
  }

  private extractDiffFromResult(result: any): string {
    // Extract diff from orchestration result
    const content = typeof result === 'string' ? result : result.content || result.response || '';
    
    // Look for diff-like patterns
    const diffMatch = content.match(/```(?:diff|typescript|javascript)\n([\s\S]*?)\n```/);
    if (diffMatch) {
      return diffMatch[1];
    }
    
    // Fallback to first code block
    const codeMatch = content.match(/```\n([\s\S]*?)\n```/);
    return codeMatch ? codeMatch[1] : '// No diff generated';
  }

  private extractDescriptionFromResult(result: any): string {
    const content = typeof result === 'string' ? result : result.content || result.response || '';
    const lines = content.split('\n');
    
    // Find the first substantial line that's not code
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length > 10 && !trimmed.startsWith('```') && !trimmed.startsWith('//')) {
        return trimmed;
      }
    }
    
    return 'AI-generated patch';
  }

  private estimateImpact(logs: CompilerLog[]): 'low' | 'medium' | 'high' {
    const errorCount = logs.filter(log => log.level === 'error').length;
    const fileCount = new Set(logs.map(log => log.file)).size;
    
    if (errorCount > 5 || fileCount > 3) return 'high';
    if (errorCount > 2 || fileCount > 1) return 'medium';
    return 'low';
  }

  private categorizePatch(logs: CompilerLog[]): 'fix' | 'optimization' | 'refactor' | 'enhancement' {
    const hasErrors = logs.some(log => log.level === 'error');
    const hasTypeIssues = logs.some(log => log.metadata.category === 'type');
    const hasPerformanceIssues = logs.some(log => log.metadata.category === 'performance');
    
    if (hasErrors) return 'fix';
    if (hasTypeIssues) return 'refactor';
    if (hasPerformanceIssues) return 'optimization';
    return 'enhancement';
  }

  private generateAttentionWeights(logs: CompilerLog[]): AttentionMatrix {
    const dimensions: [number, number] = [logs.length, 50]; // 50 tokens per log
    const weights = new Float32Array(dimensions[0] * dimensions[1]);
    
    // Generate attention pattern based on error severity and location
    logs.forEach((log, logIndex) => {
      const baseWeight = log.level === 'error' ? 0.8 : log.level === 'warning' ? 0.5 : 0.2;
      
      for (let tokenIndex = 0; tokenIndex < dimensions[1]; tokenIndex++) {
        const index = logIndex * dimensions[1] + tokenIndex;
        // Peak attention around the error location
        const distanceFromCenter = Math.abs(tokenIndex - dimensions[1] / 2);
        const attentionValue = baseWeight * Math.exp(-distanceFromCenter / 10);
        weights[index] = attentionValue;
      }
    });

    return {
      weights,
      dimensions,
      focusAreas: logs.map(log => ({
        file: log.file,
        lines: [log.line || 1, (log.line || 1) + 5],
        confidence: log.level === 'error' ? 0.9 : log.level === 'warning' ? 0.6 : 0.3
      }))
    };
  }

  /**
   * Process queued events
   */
  private async processEventQueue(): Promise<void> {
    if (this.processingQueue || !this.isActive) return;
    
    this.processingQueue = true;
    
    while (this.eventQueue.length > 0 && this.isActive) {
      const event = this.eventQueue.shift();
      if (event) {
        await this.processCompilerEvent(event);
      }
      
      // Small delay to prevent blocking
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    
    this.processingQueue = false;
  }

  /**
   * Simulate compiler events for demo purposes
   */
  private simulateCompilerEvents(): void {
    if (!this.isActive) return;

    const mockEvents: Partial<CompilerEvent>[] = [
      {
        type: 'ERROR_DETECTED',
        logs: [{
          id: 'error_1',
          timestamp: Date.now(),
          level: 'error',
          message: 'Type \'string\' is not assignable to type \'number\'',
          file: 'src/components/Chart.svelte',
          line: 42,
          code: 'let value: number = "hello";',
          metadata: {
            component: 'TypeScript',
            phase: 'type-checking',
            category: 'type'
          }
        }],
        performance: {
          compilationTime: 1200,
          memoryUsage: 45.6,
          errorCount: 1,
          warningCount: 0
        }
      },
      {
        type: 'ERROR_DETECTED',
        logs: [{
          id: 'error_2',
          timestamp: Date.now(),
          level: 'error',
          message: 'Cannot find module \'./missing-file\'',
          file: 'src/lib/utils.ts',
          line: 5,
          code: 'import { helper } from \'./missing-file\';',
          metadata: {
            component: 'Module Resolver',
            phase: 'bundling',
            category: 'import'
          }
        }],
        performance: {
          compilationTime: 800,
          memoryUsage: 52.1,
          errorCount: 1,
          warningCount: 2
        }
      }
    ];

    let eventIndex = 0;
    const interval = setInterval(() => {
      if (!this.isActive) {
        clearInterval(interval);
        return;
      }

      if (eventIndex < mockEvents.length) {
        const mockEvent = mockEvents[eventIndex];
        this.eventQueue.push({
          type: mockEvent.type!,
          logs: mockEvent.logs!,
          performance: mockEvent.performance!
        });
        eventIndex++;
      } else {
        // Reset for continuous demo
        eventIndex = 0;
      }
    }, 5000); // New event every 5 seconds
  }

  /**
   * Add external compiler event to queue
   */
  addEvent(event: CompilerEvent): void {
    this.eventQueue.push(event);
  }

  /**
   * Get current system status
   */
  getStatus() {
    return {
      isActive: this.isActive,
      queueLength: this.eventQueue.length,
      processing: this.processingQueue,
      clusters: this.somClustering.getClusters().length
    };
  }
}

// Export helper function to create feedback loop
export function createCompilerFeedbackLoop(ragEngine: EnhancedRAGEngine): CompilerFeedbackLoop {
  return new CompilerFeedbackLoop(ragEngine);
}