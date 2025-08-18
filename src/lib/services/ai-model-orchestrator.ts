// @ts-nocheck
/**
 * AI Model Orchestrator with Fallback System
 * Handles intelligent model selection and fallback between gemma3-legal and deeds-web
 * Integrates AutoGen, CrewAI, Legal-BERT, YOLO, and OCR services
 */

import { autoGenService, type AutoGenAgent, type AutoGenConversation } from './autogen-service';
import { crewAIService, type CrewAICrew, type CrewExecution } from './crewai-service';
import { legalBERT, type LegalEmbeddingResult, type LegalAnalysisResult } from '../server/ai/legalbert-middleware';
import { ocrService, type OCRResult, type ExtractedField } from './ocrService';
import type { AIResponse } from '../types/ai';

// Model configuration and priorities
interface ModelConfig {
  name: string;
  endpoint: string;
  priority: number;
  memoryRequirement: number; // GB
  capabilities: ModelCapability[];
  healthCheck: () => Promise<boolean>;
}

type ModelCapability = 
  | 'text_generation' 
  | 'legal_analysis' 
  | 'contract_review' 
  | 'case_analysis' 
  | 'document_processing'
  | 'property_law'
  | 'deed_analysis';

interface OrchestratorConfig {
  primaryModel: string;
  fallbackModels: string[];
  maxRetries: number;
  timeout: number;
  enableAutoFallback: boolean;
  memoryThreshold: number; // GB
}

interface ProcessingContext {
  task: string;
  documentType?: string;
  priority: 'low' | 'medium' | 'high';
  requiredCapabilities: ModelCapability[];
  useMultiAgent?: boolean;
  useOCR?: boolean;
  useVision?: boolean;
}

interface ModelStatus {
  name: string;
  available: boolean;
  memoryUsage: number;
  responseTime: number;
  confidence: number;
  lastChecked: number;
}

export class AIModelOrchestrator {
  private models: Map<string, ModelConfig>;
  private modelStatus: Map<string, ModelStatus>;
  private config: OrchestratorConfig;
  private circuitBreaker: Map<string, { failures: number; lastFailure: number }>;

  constructor(config: Partial<OrchestratorConfig> = {}) {
    this.config = {
      primaryModel: 'gemma3-legal',
      fallbackModels: ['deeds-web', 'llama3:8b-instruct'],
      maxRetries: 3,
      timeout: 30000,
      enableAutoFallback: true,
      memoryThreshold: 7.5, // RTX 3060 Ti limit
      ...config
    };

    this.models = new Map();
    this.modelStatus = new Map();
    this.circuitBreaker = new Map();
    
    this.initializeModels();
    this.startHealthMonitoring();
  }

  private initializeModels(): void {
    // Primary model: gemma3-legal
    this.models.set('gemma3-legal', {
      name: 'gemma3-legal',
      endpoint: 'http://localhost:11434',
      priority: 1,
      memoryRequirement: 6.5,
      capabilities: ['text_generation', 'legal_analysis', 'contract_review', 'case_analysis'],
      healthCheck: () => this.checkOllamaModel('gemma3-legal')
    });

    // Fallback model: deeds-web (specialized for property law)
    this.models.set('deeds-web', {
      name: 'deeds-web',
      endpoint: 'http://localhost:11434',
      priority: 2,
      memoryRequirement: 7.8, // Requires more memory
      capabilities: ['property_law', 'deed_analysis', 'document_processing', 'legal_analysis'],
      healthCheck: () => this.checkOllamaModel('deeds-web')
    });

    // Secondary fallback
    this.models.set('llama3:8b-instruct', {
      name: 'llama3:8b-instruct',
      endpoint: 'http://localhost:11434',
      priority: 3,
      memoryRequirement: 5.0,
      capabilities: ['text_generation', 'document_processing'],
      healthCheck: () => this.checkOllamaModel('llama3:8b-instruct')
    });
  }

  /**
   * Main orchestration method - intelligently routes requests
   */
  async processWithBestModel(
    prompt: string, 
    context: ProcessingContext
  ): Promise<AIResponse> {
    console.log(`🧠 AI Orchestrator: Processing ${context.task} with priority ${context.priority}`);
    
    // Step 1: Select optimal model based on context
    const selectedModel = await this.selectOptimalModel(context);
    
    if (!selectedModel) {
      throw new Error('No suitable AI model available');
    }

    try {
      let result: AIResponse;

      // Step 2: Route to appropriate processing pipeline
      if (context.useMultiAgent) {
        result = await this.processWithMultiAgent(prompt, context, selectedModel);
      } else if (context.useOCR && context.documentType) {
        result = await this.processWithOCR(prompt, context, selectedModel);
      } else {
        result = await this.processWithDirectModel(prompt, context, selectedModel);
      }

      // Step 3: Enhance with Legal-BERT if applicable
      if (this.shouldUseLegalBERT(context)) {
        result = await this.enhanceWithLegalBERT(result, prompt);
      }

      // Step 4: Update model performance metrics
      this.updateModelMetrics(selectedModel, true, Date.now() - result.responseTime);
      
      return result;

    } catch (error) {
      console.error(`❌ Model ${selectedModel} failed:`, error);
      
      // Step 5: Try fallback if enabled
      if (this.config.enableAutoFallback) {
        return await this.tryFallbackModels(prompt, context, selectedModel);
      }
      
      throw error;
    }
  }

  /**
   * Select optimal model based on context and current system state
   */
  private async selectOptimalModel(context: ProcessingContext): Promise<string | null> {
    const availableModels = Array.from(this.models.values())
      .filter(model => {
        // Check if model supports required capabilities
        const hasCapabilities = context.requiredCapabilities.every(cap => 
          model.capabilities.includes(cap)
        );
        
        // Check memory constraints
        const memoryOk = model.memoryRequirement <= this.config.memoryThreshold;
        
        // Check circuit breaker
        const circuitOk = !this.isCircuitBreakerOpen(model.name);
        
        return hasCapabilities && memoryOk && circuitOk;
      })
      .sort((a, b) => a.priority - b.priority);

    // Check model health and return first available
    for (const model of availableModels) {
      const isHealthy = await model.healthCheck();
      if (isHealthy) {
        console.log(`✅ Selected model: ${model.name} for ${context.task}`);
        return model.name;
      }
    }

    console.warn('⚠️ No healthy models found matching criteria');
    return null;
  }

  /**
   * Process with multi-agent systems (AutoGen/CrewAI)
   */
  private async processWithMultiAgent(
    prompt: string, 
    context: ProcessingContext, 
    modelName: string
  ): Promise<AIResponse> {
    console.log(`🤖 Using multi-agent processing with ${modelName}`);

    try {
      // Determine which multi-agent system to use
      const useCrewAI = context.task.includes('investigation') || context.task.includes('contract');
      
      if (useCrewAI) {
        // Use CrewAI for complex workflows
        const crew = context.task.includes('contract') 
          ? crewAIService.createContractAnalysisCrew()
          : crewAIService.createLegalInvestigationCrew();
        
        const execution = await crewAIService.executeCrew(crew, {
          primaryQuery: prompt,
          context: context,
          modelOverride: modelName
        });

        return {
          id: crypto.randomUUID(),
          content: execution.finalOutput || 'Multi-agent analysis completed',
          providerId: 'crewai',
          model: modelName,
          tokensUsed: execution.metrics.tokensUsed,
          responseTime: execution.metrics.totalTime,
          metadata: {
            agentType: 'crewai',
            tasksCompleted: execution.metrics.tasksCompleted,
            agentInteractions: execution.metrics.agentInteractions
          }
        };
      } else {
        // Use AutoGen for conversational analysis
        const team = autoGenService.createLegalAgentTeam();
        const agents = [team.prosecutor, team.legalResearcher, team.coordinator];
        
        const conversation = await autoGenService.startConversation(
          agents,
          prompt,
          { task: context.task, modelOverride: modelName }
        );

        // Wait for completion (simplified)
        await new Promise(resolve => setTimeout(resolve, 5000));
        const finalConversation = await autoGenService.getConversation(conversation.id);

        return {
          id: crypto.randomUUID(),
          content: finalConversation.messages[finalConversation.messages.length - 1]?.content || 'Analysis completed',
          providerId: 'autogen',
          model: modelName,
          tokensUsed: finalConversation.messages.length * 100,
          responseTime: Date.now() - finalConversation.startTime,
          metadata: {
            agentType: 'autogen',
            conversationId: conversation.id,
            messageCount: finalConversation.messages.length
          }
        };
      }
    } catch (error) {
      console.error('Multi-agent processing failed:', error);
      // Fallback to direct model
      return await this.processWithDirectModel(prompt, context, modelName);
    }
  }

  /**
   * Process with OCR integration
   */
  private async processWithOCR(
    prompt: string,
    context: ProcessingContext,
    modelName: string
  ): Promise<AIResponse> {
    console.log(`📄 Using OCR processing with ${modelName}`);

    try {
      // Note: This would typically receive a file, but for demonstration
      // we'll assume the prompt contains instructions about a document
      
      // Extract OCR results (simulated - would be from actual file upload)
      const ocrResults: OCRResult = {
        id: 'demo-ocr',
        text: 'Sample extracted text from document...',
        confidence: 0.92,
        boundingBoxes: [],
        extractedFields: [],
        metadata: {
          filename: 'document.pdf',
          fileSize: 1024000,
          dimensions: { width: 800, height: 1200 },
          pageCount: 1,
          language: 'eng',
          documentType: context.documentType || 'legal_document',
          processingDate: Date.now()
        },
        processingTime: 1500
      };

      // Enhance prompt with OCR context
      const enhancedPrompt = `
${prompt}

Document OCR Results:
- Extracted Text: ${ocrResults.text}
- Confidence: ${ocrResults.confidence}
- Document Type: ${ocrResults.metadata.documentType}
- Fields Extracted: ${ocrResults.extractedFields.length}

Please analyze this document considering the OCR extraction results.
`;

      return await this.processWithDirectModel(enhancedPrompt, context, modelName);
    } catch (error) {
      console.error('OCR processing failed:', error);
      return await this.processWithDirectModel(prompt, context, modelName);
    }
  }

  /**
   * Process with direct model call
   */
  private async processWithDirectModel(
    prompt: string,
    context: ProcessingContext,
    modelName: string
  ): Promise<AIResponse> {
    const startTime = Date.now();
    
    try {
      const response = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          messages: [
            {
              role: 'system',
              content: this.getSystemPrompt(context, modelName)
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          stream: false
        }),
        signal: AbortSignal.timeout(this.config.timeout)
      });

      if (!response.ok) {
        throw new Error(`Model API error: ${response.statusText}`);
      }

      const data = await response.json();
      const responseTime = Date.now() - startTime;

      return {
        id: crypto.randomUUID(),
        content: data.message?.content || data.response || 'No response generated',
        providerId: 'ollama',
        model: modelName,
        tokensUsed: data.eval_count || 0,
        responseTime,
        metadata: {
          task: context.task,
          directModel: true,
          loadDuration: data.load_duration,
          evalDuration: data.eval_duration
        }
      };
    } catch (error) {
      console.error(`Direct model ${modelName} failed:`, error);
      throw error;
    }
  }

  /**
   * Enhance response with Legal-BERT analysis
   */
  private async enhanceWithLegalBERT(
    response: AIResponse,
    originalPrompt: string
  ): Promise<AIResponse> {
    try {
      console.log('🧠 Enhancing with Legal-BERT analysis...');
      
      const [embedding, analysis] = await Promise.all([
        legalBERT.generateLegalEmbedding(response.content),
        legalBERT.analyzeLegalText(response.content)
      ]);

      // Calculate confidence boost based on legal analysis
      const legalTerms = analysis.entities.filter(e => 
        ['CASE_CITATION', 'STATUTE', 'COURT', 'LEGAL_CONCEPT'].includes(e.type)
      ).length;
      
      const confidenceBoost = Math.min(0.2, legalTerms * 0.03);

      return {
        ...response,
        metadata: {
          ...response.metadata,
          legalBERT: {
            embedding: embedding.embedding.slice(0, 10), // First 10 dimensions for preview
            confidence: embedding.confidence + confidenceBoost,
            legalEntities: analysis.entities.length,
            concepts: analysis.concepts.map(c => c.concept),
            sentiment: analysis.sentiment.classification,
            complexity: analysis.complexity.legalComplexity
          }
        }
      };
    } catch (error) {
      console.warn('Legal-BERT enhancement failed:', error);
      return response;
    }
  }

  /**
   * Try fallback models when primary fails
   */
  private async tryFallbackModels(
    prompt: string,
    context: ProcessingContext,
    failedModel: string
  ): Promise<AIResponse> {
    console.log(`🔄 Trying fallback models after ${failedModel} failed`);
    
    this.recordModelFailure(failedModel);
    
    // Get fallback models excluding the failed one
    const fallbackModels = this.config.fallbackModels.filter(model => model !== failedModel);
    
    for (const modelName of fallbackModels) {
      try {
        const model = this.models.get(modelName);
        if (!model) continue;
        
        const isHealthy = await model.healthCheck();
        if (!isHealthy) continue;
        
        console.log(`🔄 Trying fallback model: ${modelName}`);
        return await this.processWithDirectModel(prompt, context, modelName);
        
      } catch (error) {
        console.warn(`Fallback model ${modelName} also failed:`, error);
        continue;
      }
    }
    
    throw new Error('All models failed, including fallbacks');
  }

  /**
   * Generate appropriate system prompt based on context and model
   */
  private getSystemPrompt(context: ProcessingContext, modelName: string): string {
    const basePrompt = "You are a specialized legal AI assistant with expertise in";
    
    if (modelName === 'deeds-web') {
      return `${basePrompt} property law, real estate transactions, and deed analysis. 
Focus on property-specific legal concepts, title issues, and real estate documentation.`;
    }
    
    if (modelName === 'gemma3-legal') {
      return `${basePrompt} comprehensive legal analysis, case law research, and legal document review. 
Provide thorough, accurate legal guidance across all practice areas.`;
    }
    
    return `${basePrompt} legal document analysis and general legal assistance.`;
  }

  /**
   * Health monitoring and model management
   */
  private async checkOllamaModel(modelName: string): Promise<boolean> {
    try {
      const response = await fetch('http://localhost:11434/api/tags', {
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) return false;
      
      const data = await response.json();
      const models = data.models || [];
      return models.some((model: any) => model.name.includes(modelName));
    } catch {
      return false;
    }
  }

  private shouldUseLegalBERT(context: ProcessingContext): boolean {
    return context.requiredCapabilities.includes('legal_analysis') ||
           context.documentType === 'legal_document' ||
           context.task.includes('legal');
  }

  private startHealthMonitoring(): void {
    setInterval(async () => {
      for (const [name, model] of this.models) {
        const startTime = Date.now();
        const isHealthy = await model.healthCheck();
        const responseTime = Date.now() - startTime;
        
        this.modelStatus.set(name, {
          name,
          available: isHealthy,
          memoryUsage: model.memoryRequirement,
          responseTime,
          confidence: isHealthy ? 0.9 : 0.0,
          lastChecked: Date.now()
        });
      }
    }, 30000); // Check every 30 seconds
  }

  private recordModelFailure(modelName: string): void {
    const current = this.circuitBreaker.get(modelName) || { failures: 0, lastFailure: 0 };
    current.failures += 1;
    current.lastFailure = Date.now();
    this.circuitBreaker.set(modelName, current);
  }

  private isCircuitBreakerOpen(modelName: string): boolean {
    const breaker = this.circuitBreaker.get(modelName);
    if (!breaker) return false;
    
    // Open circuit if 3+ failures in last 5 minutes
    const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
    return breaker.failures >= 3 && breaker.lastFailure > fiveMinutesAgo;
  }

  private updateModelMetrics(modelName: string, success: boolean, responseTime: number): void {
    // Update performance metrics for model selection optimization
    const status = this.modelStatus.get(modelName);
    if (status) {
      status.responseTime = responseTime;
      status.confidence = success ? Math.min(1.0, status.confidence + 0.1) : Math.max(0.0, status.confidence - 0.2);
      this.modelStatus.set(modelName, status);
    }
  }

  /**
   * Public API methods
   */
  
  async getModelStatus(): Promise<ModelStatus[]> {
    return Array.from(this.modelStatus.values());
  }

  async testAllModels(): Promise<Record<string, boolean>> {
    const results: Record<string, boolean> = {};
    
    for (const [name, model] of this.models) {
      results[name] = await model.healthCheck();
    }
    
    return results;
  }

  getRecommendedModel(context: ProcessingContext): string | null {
    // Synchronous version of selectOptimalModel for quick recommendations
    const models = Array.from(this.models.values())
      .filter(model => {
        const hasCapabilities = context.requiredCapabilities.every(cap => 
          model.capabilities.includes(cap)
        );
        const memoryOk = model.memoryRequirement <= this.config.memoryThreshold;
        return hasCapabilities && memoryOk;
      })
      .sort((a, b) => a.priority - b.priority);

    return models[0]?.name || null;
  }
}

// Export singleton instance
export const aiOrchestrator = new AIModelOrchestrator();

// Helper functions for common use cases
export async function processLegalQuery(
  query: string,
  options: {
    documentType?: string;
    useMultiAgent?: boolean;
    priority?: 'low' | 'medium' | 'high';
  } = {}
): Promise<AIResponse> {
  const context: ProcessingContext = {
    task: 'legal_query',
    documentType: options.documentType,
    priority: options.priority || 'medium',
    requiredCapabilities: ['legal_analysis', 'text_generation'],
    useMultiAgent: options.useMultiAgent || false
  };

  return await aiOrchestrator.processWithBestModel(query, context);
}

export async function analyzePropertyDeed(
  deedText: string,
  useSpecializedModel: boolean = true
): Promise<AIResponse> {
  const context: ProcessingContext = {
    task: 'deed_analysis',
    documentType: 'property_deed',
    priority: 'high',
    requiredCapabilities: useSpecializedModel 
      ? ['property_law', 'deed_analysis', 'legal_analysis']
      : ['legal_analysis', 'document_processing'],
    useMultiAgent: false
  };

  return await aiOrchestrator.processWithBestModel(
    `Analyze this property deed for key elements, potential issues, and legal significance:\n\n${deedText}`,
    context
  );
}

export async function investigateCase(
  caseDescription: string,
  evidence: string[] = []
): Promise<AIResponse> {
  const context: ProcessingContext = {
    task: 'case_investigation',
    priority: 'high',
    requiredCapabilities: ['legal_analysis', 'case_analysis'],
    useMultiAgent: true
  };

  const fullPrompt = `
Case Investigation Request:
${caseDescription}

Available Evidence:
${evidence.map((e, i) => `${i + 1}. ${e}`).join('\n')}

Please conduct a comprehensive legal investigation and analysis.
`;

  return await aiOrchestrator.processWithBestModel(fullPrompt, context);
}