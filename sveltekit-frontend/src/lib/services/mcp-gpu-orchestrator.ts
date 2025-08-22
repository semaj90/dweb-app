/**
 * MCP GPU Orchestrator - Advanced Multi-Protocol AI Task Dispatcher
 * Coordinates GPU processing, RAG analysis, and autosolve remediation
 * Integrates with existing 37 Go services and Ollama cluster
 */

import { productionServiceClient } from './production-service-client.js';
import type { ServiceResponse } from './production-service-client.js';

export interface GPUTask {
  id: string;
  type: 'legal_analysis' | 'document_processing' | 'vector_embedding' | 'som_clustering' | 'attention_analysis' | 'error_remediation' | 'security_analysis' | 'security_validation';
  priority: 'critical' | 'high' | 'medium' | 'low';
  data: any;
  context?: {
    userId?: string;
    caseId?: string;
    documentId?: string;
    errorContext?: string;
    action?: string;
    enhancedSecurity?: boolean;
    legalProfessionalCheck?: boolean;
    enhancedValidation?: boolean;
  };
  config?: GPUTaskConfig;
  metadata?: Record<string, any>;
}

export interface GPUTaskConfig {
  useGPU: boolean;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  useRAG?: boolean;
  useContext7?: boolean;
  enableSOMClustering?: boolean;
  enableAttentionAnalysis?: boolean;
  protocol?: 'quic' | 'grpc' | 'http';
  timeout?: number;
}

export interface GPUTaskResult {
  taskId: string;
  success: boolean;
  result: any;
  metrics: {
    processingTime: number;
    gpuUtilization?: number;
    memoryUsage?: number;
    protocol: string;
    model?: string;
  };
  error?: string;
  recommendations?: string[];
  riskScore?: number;
  securityScore?: number;
  legalVerification?: {
    verified: boolean;
    confidence: number;
    details?: unknown;
  };
}

export interface ClusterMetrics {
  spawned: Record<string, number>;
  deferredActive: number;
  deferredTotal: number;
  lastAllocation: {
    type: string;
    port: number;
    timestamp: string;
  };
  events: Array<{
    type: string;
    timestamp: string;
    details: any;
  }>;
  workers: Array<{
    pid: number;
    port: number;
    uptimeSec: number;
    status: string;
  }>;
  deferredQueue: Array<{
    type: string;
    attempts: number;
    lastAttempt: string;
  }>;
}

export interface AutosolveContext {
  errorCount: number;
  errorTypes: string[];
  clusterMetrics: ClusterMetrics;
  threshold: number;
  lastRun: string;
  suggestedActions: string[];
}

class MCPGPUOrchestrator {
  private taskQueue: Map<string, GPUTask> = new Map();
  private activeGPUTasks: Set<string> = new Set();
  private clusterMetrics: ClusterMetrics | null = null;
  private autosolveContext: AutosolveContext | null = null;
  private modelConfigs: Map<string, any> = new Map();

  constructor() {
    this.initializeModels();
    this.startMetricsCollection();
  }

  private initializeModels() {
    // Gemma3 Legal Configuration
    this.modelConfigs.set('gemma3-legal', {
      name: 'gemma3-legal:latest',
      port: 11434,
      capabilities: ['legal_analysis', 'document_processing', 'contract_review'],
      gpu_layers: 35,
      memory_requirement: '7.3GB',
      context_length: 8192,
      temperature: 0.1,
      top_p: 0.9
    });

    // Nomic Embeddings Configuration
    this.modelConfigs.set('nomic-embed-text', {
      name: 'nomic-embed-text:latest',
      port: 11436,
      capabilities: ['vector_embedding', 'similarity_search'],
      dimensions: 768,
      memory_requirement: '274MB',
      batch_size: 32
    });

    // Enhanced RAG Configuration
    this.modelConfigs.set('enhanced-rag', {
      service: 'enhanced-rag',
      port: 8094,
      capabilities: ['rag_analysis', 'context_retrieval', 'document_search'],
      protocols: ['quic', 'grpc', 'http'],
      gpu_enabled: true
    });
  }

  private async startMetricsCollection() {
    // Disabled metrics collection to prevent SvelteKit fetch errors
    // These .vscode files are not accessible as web URLs
    console.log('Metrics collection disabled - .vscode files not accessible via HTTP');
    return;
    
    /* // Only run metrics collection in browser context
    if (typeof window === 'undefined') return;
    
    // Poll cluster metrics every 3 seconds
    setInterval(async () => {
      try {
        // Use absolute URL for SvelteKit compatibility
        const metricsResponse = await fetch(`${window.location.origin}/.vscode/cluster-metrics.json`);
        if (metricsResponse.ok) {
          this.clusterMetrics = await metricsResponse.json();
        }
      } catch (error) {
        console.warn('Failed to load cluster metrics:', error);
      }

      try {
        // Use absolute URL for SvelteKit compatibility  
        const autosolveResponse = await fetch(`${window.location.origin}/.vscode/auto-solve-report.json`);
        if (autosolveResponse.ok) {
          const report = await autosolveResponse.json();
          this.autosolveContext = report.autosolveContext || null;
        }
      } catch (error) {
        console.warn('Failed to load autosolve context:', error);
      }
    }, 3000); */
  }

  /**
   * Main GPU task dispatch method
   */
  async dispatchGPUTask(task: GPUTask): Promise<GPUTaskResult> {
    const startTime = performance.now();
    
    try {
      // Add task to queue
      this.taskQueue.set(task.id, task);
      this.activeGPUTasks.add(task.id);

      // Determine optimal processing route based on task type
      const result = await this.routeTaskToOptimalService(task);
      
      const processingTime = performance.now() - startTime;

      // Clean up
      this.taskQueue.delete(task.id);
      this.activeGPUTasks.delete(task.id);

      return {
        taskId: task.id,
        success: true,
        result: result.data,
        metrics: {
          processingTime,
          gpuUtilization: await this.getGPUUtilization(),
          memoryUsage: await this.getMemoryUsage(),
          protocol: result.protocol,
          model: task.config?.model
        },
        recommendations: await this.generateRecommendations(task, result),
        riskScore: result.data?.riskScore,
        securityScore: result.data?.securityScore,
        legalVerification: result.data?.legalVerification
      };

    } catch (error) {
      this.taskQueue.delete(task.id);
      this.activeGPUTasks.delete(task.id);

      return {
        taskId: task.id,
        success: false,
        result: null,
        metrics: {
          processingTime: performance.now() - startTime,
          protocol: 'failed'
        },
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  private async routeTaskToOptimalService(task: GPUTask): Promise<ServiceResponse> {
    const config = task.config || {};
    
    switch (task.type) {
      case 'legal_analysis':
        return this.processLegalAnalysis(task);
      
      case 'document_processing':
        return this.processDocument(task);
      
      case 'vector_embedding':
        return this.generateEmbeddings(task);
      
      case 'som_clustering':
        return this.performSOMClustering(task);
      
      case 'attention_analysis':
        return this.performAttentionAnalysis(task);
      
      case 'error_remediation':
        return this.performErrorRemediation(task);
      
      case 'security_analysis':
        return this.performSecurityAnalysis(task);
      
      case 'security_validation':
        return this.performSecurityValidation(task);
      
      default:
        throw new Error(`Unknown task type: ${task.type}`);
    }
  }

  private async processLegalAnalysis(task: GPUTask): Promise<ServiceResponse> {
    const prompt = this.buildLegalPrompt(task.data, task.context);
    
    // Use Enhanced RAG + Gemma3 Legal
    if (task.config?.useRAG) {
      const ragContext = await productionServiceClient.queryRAG(
        task.data.query || task.data.document,
        {
          caseId: task.context?.caseId,
          documentId: task.context?.documentId,
          includeGraph: true
        }
      );

      if (ragContext.success) {
        task.data.context = ragContext.data;
      }
    }

    // Route to Legal AI service with gRPC preference
    return productionServiceClient.getLegalAnalysis(prompt, {
      model: 'gemma3-legal',
      useGPU: task.config?.useGPU !== false,
      temperature: task.config?.temperature || 0.1,
      maxTokens: task.config?.maxTokens || 2048
    });
  }

  private async processDocument(task: GPUTask): Promise<ServiceResponse> {
    // Route to upload service with enhanced processing
    const uploadResult = await productionServiceClient.uploadDocument(
      task.data.file,
      {
        processWithAI: true,
        extractEntities: true,
        generateSummary: true,
        userId: task.context?.userId,
        caseId: task.context?.caseId
      }
    );

    if (uploadResult.success && task.config?.useRAG) {
      // Trigger RAG indexing
      await productionServiceClient.callService('/api/v1/vector/index', {
        documentId: uploadResult.data.documentId,
        content: uploadResult.data.extractedText
      });
    }

    return uploadResult;
  }

  private async generateEmbeddings(task: GPUTask): Promise<ServiceResponse> {
    // Route to embeddings service
    return productionServiceClient.callService('/api/v1/embeddings', {
      texts: Array.isArray(task.data.text) ? task.data.text : [task.data.text],
      model: 'nomic-embed-text',
      batch_size: task.config?.model === 'nomic-embed-text' ? 32 : 16
    }, {
      preferredProtocol: 'http',
      timeout: 30000
    });
  }

  private async performSOMClustering(task: GPUTask): Promise<ServiceResponse> {
    // Route to clustering service
    return productionServiceClient.callService('/api/v1/clustering/som', {
      vectors: task.data.vectors,
      map_size: task.data.mapSize || [10, 10],
      learning_rate: task.data.learningRate || 0.1,
      iterations: task.data.iterations || 1000
    }, {
      timeout: 60000
    });
  }

  private async performAttentionAnalysis(task: GPUTask): Promise<ServiceResponse> {
    // Custom attention analysis using GPU processing
    return productionServiceClient.callService('/api/v1/ai/attention-analysis', {
      text: task.data.text,
      model: task.config?.model || 'gemma3-legal',
      layer_analysis: true,
      token_importance: true
    }, {
      preferredProtocol: 'grpc',
      timeout: 45000
    });
  }

  private async performErrorRemediation(task: GPUTask): Promise<ServiceResponse> {
    // Enhanced error remediation with Context7 integration
    const errorContext = task.context?.errorContext || task.data.error;
    
    // Get Context7 documentation for relevant libraries
    const context7Docs = await this.getContext7Documentation(errorContext);
    
    // Build remediation prompt
    const remediationPrompt = this.buildRemediationPrompt(errorContext, context7Docs);
    
    // Use Enhanced RAG for similar error patterns
    const similarErrors = await productionServiceClient.queryRAG(errorContext, {
      includeErrorPatterns: true,
      includeCodeExamples: true
    });

    return productionServiceClient.callService('/api/v1/ai/remediation', {
      error: errorContext,
      context: context7Docs,
      similarPatterns: similarErrors.data,
      prompt: remediationPrompt,
      includeCodeFix: true
    }, {
      preferredProtocol: 'grpc',
      timeout: 60000
    });
  }

  private buildLegalPrompt(data: any, context?: unknown): string {
    const basePrompt = `You are a legal AI assistant specialized in document analysis and case law research.`;
    
    if (context?.caseId) {
      return `${basePrompt}\n\nCase Context: ${context.caseId}\n\nAnalyze the following legal document:\n\n${data.document || data.text}`;
    }
    
    return `${basePrompt}\n\nAnalyze the following:\n\n${data.document || data.text || data.query}`;
  }

  private buildRemediationPrompt(error: string, context7Docs: any): string {
    return `You are a TypeScript/SvelteKit expert. Fix this error using best practices:

Error: ${error}

Available documentation:
${context7Docs}

Provide a complete, working fix with explanation.`;
  }

  private async performSecurityAnalysis(task: GPUTask): Promise<ServiceResponse> {
    const { email, timestamp, userAgent, fingerprint } = task.data;
    
    try {
      // Use production service client for enhanced security analysis
      const response = await productionServiceClient.callService('/api/security/analyze', {
        email,
        timestamp,
        userAgent,
        fingerprint,
        context: task.context
      }, {
        preferredProtocol: task.config?.protocol || 'auto',
        timeout: task.config?.timeout || 10000
      });

      // Enhanced analysis using Ollama if available
      let aiAnalysis = null;
      if (this.modelConfigs.has('gemma3-legal')) {
        const prompt = `Analyze the following authentication attempt for security risks:
Email: ${email}
User Agent: ${userAgent}
Timestamp: ${timestamp}
Context: ${JSON.stringify(task.context)}

Provide a risk assessment on a scale of 0-1 where:
- 0.0-0.3: Low risk
- 0.4-0.6: Medium risk  
- 0.7-0.8: High risk
- 0.9-1.0: Critical risk

Consider factors like:
- Email domain reputation
- Device fingerprint anomalies
- Login patterns
- User agent consistency

Respond with JSON: {"riskScore": 0.0, "reasoning": "explanation", "recommendations": ["list"]}`;

        try {
          const aiResponse = await productionServiceClient.callService('/api/ollama/generate', {
            model: 'gemma3-legal',
            prompt,
            system: 'You are a cybersecurity expert specializing in authentication security analysis.',
            temperature: 0.1,
            max_tokens: 512
          });
          
          if (aiResponse.success) {
            const match = aiResponse.data.response.match(/\{.*\}/s);
            if (match) {
              aiAnalysis = JSON.parse(match[0]);
            }
          }
        } catch (aiError) {
          console.warn('AI analysis failed, using fallback:', aiError);
        }
      }

      // Calculate composite risk score
      const baseRiskScore = response.data?.riskScore || 0.1;
      const aiRiskScore = aiAnalysis?.riskScore || 0.1;
      const compositeRiskScore = Math.min(1.0, (baseRiskScore + aiRiskScore) / 2);

      return {
        success: true,
        data: {
          riskScore: compositeRiskScore,
          securityScore: Math.round((1 - compositeRiskScore) * 100),
          analysis: response.data?.analysis,
          aiAnalysis,
          recommendations: aiAnalysis?.recommendations || [],
          flags: response.data?.flags || []
        },
        protocol: response.protocol || 'http',
        latency: response.latency || 0
      };
      
    } catch (error) {
      // Fallback security analysis
      return {
        success: true,
        data: {
          riskScore: 0.5, // Medium risk as fallback
          securityScore: 50,
          analysis: 'Fallback security analysis',
          error: error instanceof Error ? error.message : 'Unknown error'
        },
        protocol: 'fallback',
        latency: 0
      };
    }
  }

  private async performSecurityValidation(task: GPUTask): Promise<ServiceResponse> {
    const { email, firstName, lastName, role, department, jurisdiction, badgeNumber } = task.data;
    
    try {
      // Legal professional validation using production service
      const validationResponse = await productionServiceClient.callService('/api/validation/legal-professional', {
        email,
        firstName,
        lastName,
        role,
        department,
        jurisdiction,
        badgeNumber,
        timestamp: new Date().toISOString()
      }, {
        preferredProtocol: task.config?.protocol || 'auto',
        timeout: 15000
      });

      // Enhanced AI validation using Ollama
      let legalVerification = { verified: false, confidence: 0 };
      
      if (this.modelConfigs.has('gemma3-legal')) {
        const verificationPrompt = `Validate the following legal professional registration:
Name: ${firstName} ${lastName}
Email: ${email}
Role: ${role}
Department: ${department}
Jurisdiction: ${jurisdiction}
Badge Number: ${badgeNumber || 'Not provided'}

Assess the credibility and consistency of this legal professional profile.
Consider:
- Email domain appropriateness for legal role
- Department/jurisdiction alignment
- Role consistency with provided information
- Professional naming conventions

Respond with JSON: {"verified": true/false, "confidence": 0.0-1.0, "concerns": ["list"], "recommendations": ["list"]}`;

        try {
          const aiResponse = await productionServiceClient.callService('/api/ollama/generate', {
            model: 'gemma3-legal',
            prompt: verificationPrompt,
            system: 'You are a legal verification expert specializing in validating legal professional credentials.',
            temperature: 0.1,
            max_tokens: 512
          });
          
          if (aiResponse.success) {
            const match = aiResponse.data.response.match(/\{.*\}/s);
            if (match) {
              const parsed = JSON.parse(match[0]);
              legalVerification = {
                verified: parsed.verified && parsed.confidence > 0.7,
                confidence: parsed.confidence || 0,
                details: parsed
              };
            }
          }
        } catch (aiError) {
          console.warn('AI legal verification failed:', aiError);
        }
      }

      // Calculate security score based on validation results
      const baseScore = validationResponse.data?.validationScore || 70;
      const aiScore = legalVerification.confidence * 100;
      const compositeScore = Math.round((baseScore + aiScore) / 2);

      return {
        success: true,
        data: {
          riskScore: Math.max(0, (100 - compositeScore) / 100),
          securityScore: compositeScore,
          legalVerification,
          validation: validationResponse.data,
          professionalVerification: legalVerification,
          compositeScore
        },
        protocol: validationResponse.protocol || 'http',
        latency: validationResponse.latency || 0
      };
      
    } catch (error) {
      return {
        success: false,
        data: {
          riskScore: 0.8, // High risk on validation failure
          securityScore: 20,
          legalVerification: { verified: false, confidence: 0 },
          error: error instanceof Error ? error.message : 'Validation failed',
          fallback: true
        },
        protocol: 'fallback',
        latency: 0
      };
    }
  }

  private async getContext7Documentation(errorContext: string): Promise<string> {
    // Simulate Context7.2 documentation retrieval
    try {
      const response = await productionServiceClient.callService('/api/context7', {
        query: errorContext,
        libraries: ['svelte5', 'sveltekit', 'typescript', 'drizzle'],
        format: 'typescript'
      });
      
      return response.success ? response.data.content : '';
    } catch {
      return '';
    }
  }

  private async generateRecommendations(task: GPUTask, result: ServiceResponse): Promise<string[]> {
    const recommendations: string[] = [];
    
    if (result.latency > 5000) {
      recommendations.push('Consider using QUIC protocol for better performance');
    }
    
    if (task.type === 'legal_analysis' && !task.config?.useRAG) {
      recommendations.push('Enable RAG for enhanced legal context');
    }
    
    if (this.activeGPUTasks.size > 5) {
      recommendations.push('Consider implementing task queuing for better resource management');
    }
    
    return recommendations;
  }

  private async getGPUUtilization(): Promise<number> {
    try {
      const response = await productionServiceClient.callService('/api/gpu/metrics', {}, {
        timeout: 5000
      });
      return response.success ? response.data.utilization : 0;
    } catch {
      return 0;
    }
  }

  private async getMemoryUsage(): Promise<number> {
    try {
      const response = await productionServiceClient.callService('/api/gpu/memory-status', {}, {
        timeout: 5000
      });
      return response.success ? response.data.memory_used : 0;
    } catch {
      return 0;
    }
  }

  // Public API methods

  /**
   * Process legal document with full AI pipeline
   */
  async processLegalDocument(
    document: string | File,
    options: {
      caseId?: string;
      userId?: string;
      includeRAG?: boolean;
      includeGraph?: boolean;
      generateSummary?: boolean;
    } = {}
  ): Promise<GPUTaskResult> {
    const task: GPUTask = {
      id: `legal_${Date.now()}`,
      type: 'legal_analysis',
      priority: 'high',
      data: { document },
      context: {
        caseId: options.caseId,
        userId: options.userId
      },
      config: {
        useGPU: true,
        useRAG: options.includeRAG !== false,
        model: 'gemma3-legal',
        protocol: 'grpc'
      }
    };

    return this.dispatchGPUTask(task);
  }

  /**
   * Trigger autosolve maintenance cycle
   */
  async triggerAutosolve(options: {
    threshold?: number;
    includeClusterMetrics?: boolean;
    forceRun?: boolean;
  } = {}): Promise<GPUTaskResult> {
    const task: GPUTask = {
      id: `autosolve_${Date.now()}`,
      type: 'error_remediation',
      priority: 'critical',
      data: {
        threshold: options.threshold || 5,
        clusterMetrics: options.includeClusterMetrics ? this.clusterMetrics : null,
        forceRun: options.forceRun
      },
      config: {
        useGPU: false,
        useContext7: true,
        protocol: 'http'
      }
    };

    return this.dispatchGPUTask(task);
  }

  /**
   * Get current cluster status and metrics
   */
  async getClusterStatus(): Promise<{
    metrics: ClusterMetrics | null;
    autosolveContext: AutosolveContext | null;
    activeGPUTasks: number;
    queueSize: number;
  }> {
    return {
      metrics: this.clusterMetrics,
      autosolveContext: this.autosolveContext,
      activeGPUTasks: this.activeGPUTasks.size,
      queueSize: this.taskQueue.size
    };
  }

  /**
   * Route GPU task dispatch from SvelteKit API
   */
  async routeAPIRequest(
    endpoint: string,
    data: any,
    context: any
  ): Promise<GPUTaskResult> {
    const taskType = this.mapEndpointToTaskType(endpoint);
    
    const task: GPUTask = {
      id: `api_${Date.now()}`,
      type: taskType,
      priority: 'medium',
      data,
      context,
      config: {
        useGPU: true,
        useRAG: true,
        protocol: 'quic'
      }
    };

    return this.dispatchGPUTask(task);
  }

  private mapEndpointToTaskType(endpoint: string): GPUTask['type'] {
    if (endpoint.includes('legal')) return 'legal_analysis';
    if (endpoint.includes('upload')) return 'document_processing';
    if (endpoint.includes('embed')) return 'vector_embedding';
    if (endpoint.includes('cluster')) return 'som_clustering';
    if (endpoint.includes('attention')) return 'attention_analysis';
    if (endpoint.includes('autosolve')) return 'error_remediation';
    return 'legal_analysis';
  }
}

// Singleton instance
export const mcpGPUOrchestrator = new MCPGPUOrchestrator();

export default mcpGPUOrchestrator;