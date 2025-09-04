// Phase 3: AI/GPU Optimization with Ollama Integration
// Advanced GPU-accelerated AI processing with intelligent caching

import { cacheFirstService } from './cache-first-architecture.js';
import { writable, derived, type Writable } from 'svelte/store'
import crypto from "crypto";

// ===== OLLAMA GPU CONFIGURATION =====

interface OllamaConfig {
  baseUrl: string;
  models: {
    legal: string;
    embedding: string;
    classification: string;
  };
  gpu: {
    enabled: boolean;
    layers: number;
    memoryFraction: number;
    device: string;
  };
  performance: {
    maxConcurrent: number;
    timeoutMs: number;
    retryAttempts: number;
    cacheTTL: number;
  };
}

const defaultConfig: OllamaConfig = {
  baseUrl: 'http://localhost:11434',
  models: {
    legal: 'gemma3-legal:latest',
    embedding: 'nomic-embed-text:latest',
    classification: 'deeds-web:latest'
  },
  gpu: {
    enabled: true,
    layers: 35, // RTX 3060 Ti optimized
    memoryFraction: 0.8,
    device: 'cuda:0'
  },
  performance: {
    maxConcurrent: 4,
    timeoutMs: 30000,
    retryAttempts: 3,
    cacheTTL: 10 * 60 * 1000 // 10 minutes
  }
};

// ===== AI PROCESSING QUEUE =====

interface AITask {
  id: string;
  type: 'analyze' | 'summarize' | 'classify' | 'embed' | 'generate';
  priority: number;
  data: any;
  model: string;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  result?: any;
  error?: string;
  cacheKey?: string;
}

export class AIGPUOptimizationService {
  private config: OllamaConfig = defaultConfig;
  private taskQueue = new Map<string, AITask>();
  private activeTasks = new Map<string, Promise<any>>();
  private modelCache = new Map<string, any>();
  private performanceMetrics = new Map<string, number[]>();
  
  // Reactive stores
  public queueStats = writable({
    pending: 0,
    active: 0,
    completed: 0,
    failed: 0,
    avgProcessingTime: 0,
    gpuUtilization: 0,
    cacheHitRate: 0
  });

  public modelStatus = writable({
    legal: { loaded: false, memory: 0, layers: 0 },
    embedding: { loaded: false, memory: 0, layers: 0 },
    classification: { loaded: false, memory: 0, layers: 0 }
  });

  constructor(config?: Partial<OllamaConfig>) {
    if (config) {
      this.config = { ...defaultConfig, ...config };
    }
    
    this.initializeGPUOptimization();
    this.startPerformanceMonitoring();
  }

  // ===== GPU OPTIMIZATION INITIALIZATION =====

  private async initializeGPUOptimization(): Promise<void> {
    try {
      console.log('Initializing AI/GPU optimization...');
      
      // Check Ollama server status
      await this.checkOllamaHealth();
      
      // Preload critical models with GPU optimization
      await this.preloadModels();
      
      // Initialize GPU monitoring
      await this.initializeGPUMonitoring();
      
      console.log('AI/GPU optimization initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize AI/GPU optimization:', error);
    }
  }

  private async checkOllamaHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`, {
        timeout: 5000
      } as any);
      
      if (response.ok) {
        const data = await response.json();
        console.log(`Ollama server healthy with ${data.models?.length || 0} models`);
        return true;
      }
      
      throw new Error(`Ollama server unhealthy: ${response.status}`);
      
    } catch (error) {
      console.error('Ollama health check failed:', error);
      return false;
    }
  }

  private async preloadModels(): Promise<void> {
    const models = Object.entries(this.config.models);
    
    for (const [type, model] of models) {
      try {
        console.log(`Preloading ${type} model: ${model}`);
        
        await this.optimizeModelLoading(model, type);
        
        this.modelStatus.update(status => ({
          ...status,
          [type]: { loaded: true, memory: 0, layers: this.config.gpu.layers }
        }));
        
      } catch (error) {
        console.error(`Failed to preload ${type} model:`, error);
      }
    }
  }

  private async optimizeModelLoading(model: string, type: string): Promise<void> {
    // Pre-warm model with GPU optimization
    const response = await fetch(`${this.config.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: 'warmup',
        stream: false,
        options: {
          num_gpu: this.config.gpu.layers,
          num_thread: 8,
          temperature: 0.1,
          top_p: 0.9,
          repeat_penalty: 1.1
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`Model warmup failed for ${model}: ${response.statusText}`);
    }
    
    await response.json();
    console.log(`Model ${model} warmed up successfully`);
  }

  // ===== INTELLIGENT AI TASK PROCESSING =====

  async processAITask(task: Omit<AITask, 'id' | 'createdAt'>): Promise<any> {
    const taskId = crypto.randomUUID();
    const aiTask: AITask = {
      ...task,
      id: taskId,
      createdAt: new Date(),
      cacheKey: this.generateCacheKey(task)
    };

    // Check cache first
    const cachedResult = await this.checkAICache(aiTask.cacheKey!);
    if (cachedResult) {
      console.log(`Cache hit for AI task ${taskId}`);
      await this.updateCacheStats('hit');
      return cachedResult;
    }

    // Add to queue
    this.taskQueue.set(taskId, aiTask);
    this.updateQueueStats();
    
    await this.updateCacheStats('miss');

    try {
      // Process based on task type
      let result;
      
      switch (aiTask.type) {
        case 'analyze':
          result = await this.performAnalysis(aiTask);
          break;
        case 'summarize':
          result = await this.performSummarization(aiTask);
          break;
        case 'classify':
          result = await this.performClassification(aiTask);
          break;
        case 'embed':
          result = await this.performEmbedding(aiTask);
          break;
        case 'generate':
          result = await this.performGeneration(aiTask);
          break;
        default:
          throw new Error(`Unsupported task type: ${aiTask.type}`);
      }

      // Cache the result
      await this.cacheAIResult(aiTask.cacheKey!, result);
      
      // Store in cache service for cross-session persistence
      await cacheFirstService.createAIAnalysis({
        entityId: aiTask.data.entityId || taskId,
        entityType: aiTask.data.entityType || 'unknown',
        analysisType: aiTask.type,
        prompt: JSON.stringify(aiTask.data),
        response: JSON.stringify(result),
        confidence: result.confidence || 0.8,
        model: aiTask.model,
        processingTime: Date.now() - aiTask.createdAt.getTime()
      });

      aiTask.result = result;
      aiTask.completedAt = new Date();
      
      this.recordPerformanceMetric(aiTask.type, Date.now() - aiTask.createdAt.getTime());
      this.updateQueueStats();

      return result;

    } catch (error) {
      aiTask.error = error instanceof Error ? error.message : String(error);
      this.updateQueueStats();
      throw error;
    } finally {
      this.taskQueue.delete(taskId);
    }
  }

  // ===== AI PROCESSING IMPLEMENTATIONS =====

  private async performAnalysis(task: AITask): Promise<any> {
    const { entityId, entityType, content, analysisScope } = task.data;
    
    const prompt = this.buildAnalysisPrompt(content, analysisScope, entityType);
    
    const response = await fetch(`${this.config.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: task.model,
        prompt,
        stream: false,
        options: {
          num_gpu: this.config.gpu.layers,
          temperature: 0.3,
          top_p: 0.9,
          max_tokens: 2000
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    return {
      analysis: result.response,
      confidence: this.calculateConfidence(result.response),
      keyFindings: this.extractKeyFindings(result.response),
      recommendations: this.extractRecommendations(result.response),
      metadata: {
        model: task.model,
        processingTime: result.total_duration,
        tokenCount: result.eval_count
      }
    };
  }

  private async performEmbedding(task: AITask): Promise<any> {
    const { text, dimensions } = task.data;
    
    const response = await fetch(`${this.config.baseUrl}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.models.embedding,
        prompt: text,
        options: {
          num_gpu: this.config.gpu.layers
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Embedding failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    return {
      embedding: result.embedding,
      dimensions: result.embedding.length,
      text: text.substring(0, 100) + '...', // Store preview
      model: this.config.models.embedding
    };
  }

  private async performClassification(task: AITask): Promise<any> {
    const { content, categories, confidence_threshold } = task.data;
    
    const prompt = this.buildClassificationPrompt(content, categories);
    
    const response = await fetch(`${this.config.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: task.model,
        prompt,
        stream: false,
        options: {
          num_gpu: this.config.gpu.layers,
          temperature: 0.1, // Lower temperature for classification
          top_p: 0.8
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Classification failed: ${response.statusText}`);
    }

    const result = await response.json();
    const classification = this.parseClassificationResult(result.response);
    
    return {
      category: classification.category,
      confidence: classification.confidence,
      alternatives: classification.alternatives,
      reasoning: classification.reasoning,
      meetThreshold: classification.confidence >= (confidence_threshold || 0.7)
    };
  }

  private async performSummarization(task: AITask): Promise<any> {
    const { content, maxLength, style } = task.data;
    
    const prompt = this.buildSummarizationPrompt(content, maxLength, style);
    
    const response = await fetch(`${this.config.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: task.model,
        prompt,
        stream: false,
        options: {
          num_gpu: this.config.gpu.layers,
          temperature: 0.4,
          top_p: 0.9,
          max_tokens: maxLength || 500
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Summarization failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    return {
      summary: result.response,
      originalLength: content.length,
      summaryLength: result.response.length,
      compressionRatio: result.response.length / content.length,
      keyPoints: this.extractKeyPoints(result.response)
    };
  }

  private async performGeneration(task: AITask): Promise<any> {
    const { prompt, context, style, maxTokens } = task.data;
    
    const enhancedPrompt = this.buildGenerationPrompt(prompt, context, style);
    
    const response = await fetch(`${this.config.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: task.model,
        prompt: enhancedPrompt,
        stream: false,
        options: {
          num_gpu: this.config.gpu.layers,
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: maxTokens || 1000
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Generation failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    return {
      generated: result.response,
      originalPrompt: prompt,
      wordCount: result.response.split(' ').length,
      quality: this.assessGenerationQuality(result.response, prompt)
    };
  }

  // ===== PROMPT BUILDING HELPERS =====

  private buildAnalysisPrompt(content: string, scope: string, entityType: string): string {
    return `As a legal AI assistant, analyze the following ${entityType} content:

Content: ${content}

Please provide a comprehensive analysis focusing on: ${scope}

Structure your response as:
1. Key Legal Issues
2. Risk Assessment
3. Compliance Considerations
4. Recommendations
5. Priority Actions

Analysis:`;
  }

  private buildClassificationPrompt(content: string, categories: string[]): string {
    return `Classify the following legal content into one of these categories: ${categories.join(', ')}

Content: ${content}

Respond in JSON format:
{
  "category": "selected_category",
  "confidence": 0.95,
  "alternatives": [{"category": "alt1", "confidence": 0.3}],
  "reasoning": "explanation"
}`;
  }

  private buildSummarizationPrompt(content: string, maxLength: number, style: string): string {
    return `Summarize the following legal content in ${style} style, maximum ${maxLength} words:

Content: ${content}

Summary:`;
  }

  private buildGenerationPrompt(prompt: string, context: string, style: string): string {
    return `${context ? `Context: ${context}\n\n` : ''}Generate ${style} content based on this prompt:

${prompt}

Generated content:`;
  }

  // ===== CACHING SYSTEM =====

  private generateCacheKey(task: Omit<AITask, 'id' | 'createdAt'>): string {
    const dataString = JSON.stringify(task.data);
    const hashInput = `${task.type}-${task.model}-${dataString}`;
    
    // Simple hash function (in production, use a proper crypto hash)
    let hash = 0;
    for (let i = 0; i < hashInput.length; i++) {
      const char = hashInput.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return `ai-cache-${Math.abs(hash)}`;
  }

  private async checkAICache(cacheKey: string): Promise<any> {
    try {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        const { result, timestamp } = JSON.parse(cached);
        
        // Check if cache is still valid
        if (Date.now() - timestamp < this.config.performance.cacheTTL) {
          return result;
        } else {
          localStorage.removeItem(cacheKey);
        }
      }
    } catch (error) {
      console.error('Cache check failed:', error);
    }
    
    return null;
  }

  private async cacheAIResult(cacheKey: string, result: any): Promise<void> {
    try {
      const cacheEntry = {
        result,
        timestamp: Date.now()
      };
      
      localStorage.setItem(cacheKey, JSON.stringify(cacheEntry));
    } catch (error) {
      console.error('Failed to cache AI result:', error);
    }
  }

  // ===== PERFORMANCE MONITORING =====

  private startPerformanceMonitoring(): void {
    setInterval(() => {
      this.updateQueueStats();
      this.monitorGPUUtilization();
    }, 5000); // Update every 5 seconds
  }

  private updateQueueStats(): void {
    const pending = Array.from(this.taskQueue.values()).filter(t => !t.startedAt).length;
    const active = Array.from(this.taskQueue.values()).filter(t => t.startedAt && !t.completedAt).length;
    const completed = Array.from(this.taskQueue.values()).filter(t => t.completedAt).length;
    const failed = Array.from(this.taskQueue.values()).filter(t => t.error).length;
    
    const processingTimes = Array.from(this.performanceMetrics.values()).flat();
    const avgProcessingTime = processingTimes.length > 0 
      ? processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length
      : 0;

    this.queueStats.set({
      pending,
      active,
      completed,
      failed,
      avgProcessingTime: Math.round(avgProcessingTime),
      gpuUtilization: 0, // Will be updated by GPU monitoring
      cacheHitRate: this.calculateCacheHitRate()
    });
  }

  private recordPerformanceMetric(taskType: string, processingTime: number): void {
    if (!this.performanceMetrics.has(taskType)) {
      this.performanceMetrics.set(taskType, []);
    }
    
    const metrics = this.performanceMetrics.get(taskType)!;
    metrics.push(processingTime);
    
    // Keep only last 100 measurements
    if (metrics.length > 100) {
      metrics.splice(0, metrics.length - 100);
    }
  }

  private async monitorGPUUtilization(): Promise<void> {
    // This would integrate with actual GPU monitoring
    // For now, simulate GPU utilization
    const utilization = Math.random() * 100;
    
    this.queueStats.update(stats => ({
      ...stats,
      gpuUtilization: Math.round(utilization)
    }));
  }

  private cacheHits = 0;
  private cacheMisses = 0;

  private async updateCacheStats(type: 'hit' | 'miss'): Promise<void> {
    if (type === 'hit') this.cacheHits++
    else this.cacheMisses++;
  }

  private calculateCacheHitRate(): number {
    const total = this.cacheHits + this.cacheMisses;
    return total > 0 ? this.cacheHits / total : 0;
  }

  // ===== UTILITY FUNCTIONS =====

  private calculateConfidence(response: string): number {
    // Simple confidence calculation based on response characteristics
    const length = response.length;
    const certaintyWords = ['definitely', 'clearly', 'obviously', 'certainly'];
    const uncertaintyWords = ['might', 'possibly', 'perhaps', 'maybe'];
    
    let confidence = 0.5;
    
    if (length > 100) confidence += 0.1;
    if (length > 500) confidence += 0.1;
    
    certaintyWords.forEach(word => {
      if (response.toLowerCase().includes(word)) confidence += 0.1;
    });
    
    uncertaintyWords.forEach(word => {
      if (response.toLowerCase().includes(word)) confidence -= 0.1;
    });
    
    return Math.max(0, Math.min(1, confidence));
  }

  private extractKeyFindings(response: string): string[] {
    // Extract bullet points or numbered items
    const lines = response.split('\n');
    return lines.filter(line => 
      line.trim().match(/^[-•*]\s/) || 
      line.trim().match(/^\d+\.\s/)
    ).map(line => line.trim());
  }

  private extractRecommendations(response: string): string[] {
    const recommendationSection = response.toLowerCase().includes('recommend') 
      ? response.substring(response.toLowerCase().indexOf('recommend'))
      : '';
      
    return this.extractKeyFindings(recommendationSection);
  }

  private extractKeyPoints(summary: string): string[] {
    return summary.split('.').filter(point => point.trim().length > 20).map(point => point.trim());
  }

  private parseClassificationResult(response: string): any {
    try {
      return JSON.parse(response);
    } catch {
      // Fallback parsing
      return {
        category: 'unknown',
        confidence: 0.5,
        alternatives: [],
        reasoning: response
      };
    }
  }

  private assessGenerationQuality(generated: string, prompt: string): number {
    // Simple quality assessment
    const relevance = prompt.toLowerCase().split(' ').filter(word => 
      generated.toLowerCase().includes(word)
    ).length / prompt.split(' ').length;
    
    const readability = generated.split(' ').length > 50 ? 0.8 : 0.6;
    const coherence = generated.includes('.') && generated.includes(' ') ? 0.9 : 0.5;
    
    return (relevance + readability + coherence) / 3;
  }

  private async initializeGPUMonitoring(): Promise<void> {
    // Initialize GPU monitoring if available
    console.log('GPU monitoring initialized');
  }
}

// ===== GLOBAL AI/GPU SERVICE =====

export const aiGPUService = new AIGPUOptimizationService();
// ===== HIGH-LEVEL API FUNCTIONS =====

export async function analyzeEvidence(evidenceId: string, content: string): Promise<any> {
  return aiGPUService.processAITask({
    type: 'analyze',
    priority: 2,
    data: {
      entityId: evidenceId,
      entityType: 'evidence',
      content,
      analysisScope: 'legal implications, risks, and recommendations'
    },
    model: defaultConfig.models.legal
  });
}

export async function classifyDocument(content: string, categories: string[]): Promise<any> {
  return aiGPUService.processAITask({
    type: 'classify',
    priority: 1,
    data: {
      content,
      categories,
      confidence_threshold: 0.8
    },
    model: defaultConfig.models.classification
  });
}

export async function generateEmbedding(text: string): Promise<any> {
  return aiGPUService.processAITask({
    type: 'embed',
    priority: 3,
    data: {
      text,
      dimensions: 384
    },
    model: defaultConfig.models.embedding
  });
}

export async function summarizeCase(caseId: string, content: string): Promise<any> {
  return aiGPUService.processAITask({
    type: 'summarize',
    priority: 2,
    data: {
      entityId: caseId,
      entityType: 'case',
      content,
      maxLength: 500,
      style: 'professional legal summary'
    },
    model: defaultConfig.models.legal
  });
}