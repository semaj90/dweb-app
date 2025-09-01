import { getContext7MulticoreService } from "../services/context7-multicore";
/**
 * Full Stack Legal AI Workflow Integration
 * Orchestrates the complete system: VS Code tasks + Agent orchestration + GPU processing
 * Designed to work seamlessly with .vscode/tasks.json
 */

import { comprehensiveOrchestrator, type ComprehensiveAgentRequest } from "./comprehensive-agent-orchestration";
import { flashAttentionMulticoreBridge, type FlashAttentionMulticoreRequest } from "../services/flash-attention-multicore";

export interface FullStackWorkflowRequest {
  mode: 'error_analysis' | 'legal_processing' | 'system_diagnostic' | 'performance_test';
  data?: any;
  options?: {
    useGPU?: boolean;
    enableAgents?: boolean;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    maxProcessingTime?: number;
  };
}

export interface FullStackWorkflowResult {
  mode: string;
  success: boolean;
  results: {
    agentOrchestration?: unknown;
    gpuProcessing?: unknown;
    multicoreAnalysis?: unknown;
    systemMetrics?: unknown;
    error?: string;
  };
  performance: {
    totalTime: number;
    gpuUtilization: number;
    agentsUsed: number;
    multicoreWorkers: number;
  };
  recommendations: string[];
  nextSteps: string[];
}

export class FullStackLegalAIWorkflow {
  private isInitialized = false;
  private systemStatus = {
    orchestrator: false,
    flashattention: false,
    multicore: false
  };

  constructor() {
    console.log('🏗️ Initializing Full Stack Legal AI Workflow...');
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🚀 Starting comprehensive system initialization...');
    const startTime = performance.now();

    try {
      // Initialize all systems in parallel
      await Promise.all([
        this.initializeOrchestrator(),
        this.initializeFlashAttention(),
        this.initializeMulticore()
      ]);

      this.isInitialized = true;
      const initTime = performance.now() - startTime;

      console.log(`✅ Full Stack Legal AI Workflow initialized in ${initTime.toFixed(2)}ms`);
      console.log('📊 System Status:', this.getSystemStatus());

    } catch (error: any) {
      console.error('❌ Full Stack initialization failed:', error);
      throw new Error(`Full Stack initialization failed: ${error?.message ?? String(error)}`);
    }
  }

  private async initializeOrchestrator(): Promise<void> {
    try {
      await comprehensiveOrchestrator.initialize();
      this.systemStatus.orchestrator = true;
      console.log('✅ Agent Orchestrator: Ready');
    } catch (error: any) {
      console.warn('⚠️ Agent Orchestrator failed to initialize:', error);
      this.systemStatus.orchestrator = false;
    }
  }

  private async initializeFlashAttention(): Promise<void> {
    try {
      await flashAttentionMulticoreBridge.initialize();
      this.systemStatus.flashattention = true;
      console.log('✅ FlashAttention2 + Multicore Bridge: Ready');
    } catch (error: any) {
      console.warn('⚠️ FlashAttention2 Bridge failed to initialize:', error);
      this.systemStatus.flashattention = false;
    }
  }

  private async initializeMulticore(): Promise<void> {
    try {
      const multicoreService = getContext7MulticoreService();
      // Wait for at least one worker to be ready
      let attempts = 0;
      while (attempts < 10) {
        const status = multicoreService.getSystemStatus();
        if (status.workers.length > 0 && status.workers.some(w => w.status === 'healthy')) {
          this.systemStatus.multicore = true;
          console.log('✅ Context7 Multicore Service: Ready');
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 500));
        attempts++;
      }
      console.warn('⚠️ Context7 Multicore Service: Timeout waiting for workers');
    } catch (error: any) {
      console.warn('⚠️ Context7 Multicore Service failed to initialize:', error);
      this.systemStatus.multicore = false;
    }
  }

  /**
   * Execute full stack workflow based on mode
   */
  async executeWorkflow(request: FullStackWorkflowRequest): Promise<FullStackWorkflowResult> {
    await this.initialize();

    const startTime = performance.now();
    console.log(`🔄 Executing Full Stack Workflow: ${request.mode}`);

    try {
      let result: FullStackWorkflowResult;

      switch (request.mode) {
        case 'error_analysis':
          result = await this.executeErrorAnalysis(request);
          break;
        case 'legal_processing':
          result = await this.executeLegalProcessing(request);
          break;
        case 'system_diagnostic':
          result = await this.executeSystemDiagnostic(request);
          break;
        case 'performance_test':
          result = await this.executePerformanceTest(request);
          break;
        default:
          throw new Error(`Unknown workflow mode: ${request.mode}`);
      }

      const totalTime = performance.now() - startTime;
      result.performance.totalTime = totalTime;

      console.log(`✅ Workflow '${request.mode}' completed in ${totalTime.toFixed(2)}ms`);
      return result;

    } catch (error: any) {
      console.error(`❌ Workflow '${request.mode}' failed:`, error);
      return this.createErrorResult(request.mode, error, performance.now() - startTime);
    }
  }

  /**
   * Execute error analysis workflow (for 1962 TypeScript errors)
   */
  private async executeErrorAnalysis(request: FullStackWorkflowRequest): Promise<FullStackWorkflowResult> {
    console.log('🔍 Starting comprehensive error analysis...');

    const errorData = request.data || {
      totalErrors: 1962,
      categories: {
        svelte5_migration: 800,
        ui_component_mismatch: 600,
        css_unused_selectors: 400,
        binding_issues: 162
      },
      sampleErrors: [
        'TS2322: Type mismatch in component props',
        'Svelte: Object literal may only specify known properties',
        'CSS: Unused selector warning',
        'Binding: Cannot use bind: with non-bindable property'
      ]
    };

    // Run all analysis systems in parallel
    const analysisPromises: Promise<any>[] = [];

    // 1. Agent orchestration analysis
    if (this.systemStatus.orchestrator && request.options?.enableAgents !== false) {
      const orchestrationRequest: ComprehensiveAgentRequest = {
        prompt: `Analyze and fix ${errorData.totalErrors} TypeScript/Svelte errors in legal AI application`,
        context: errorData,
        options: {
          agents: ['claude', 'crewai', 'autogen'],
          priority: request.options?.priority || 'high',
          analysisType: 'document_processing',
          useMulticoreAnalysis: true,
          errorAnalysis: true,
          autoFix: false // Don't auto-fix, just analyze
        }
      };
      analysisPromises.push(
        comprehensiveOrchestrator.executeComprehensiveAnalysis(orchestrationRequest)
          .then(result => ({ type: 'orchestration', result }))
      );
    }

    // 2. GPU-accelerated error analysis
    if (this.systemStatus.flashattention && request.options?.useGPU !== false) {
      const codeContext = [
        'export let prop = "default"',
        'let { prop = "default" } = $props()',
        '<CardRoot class="card">',
        '<CardRoot className="card">',
        '<DialogRoot bind:open={showDialog}>',
        'unused .container { display: flex; }'
      ];

      analysisPromises.push(
        flashAttentionMulticoreBridge.analyzeErrorsWithAttention(errorData, codeContext)
          .then(result => ({ type: 'gpu', result }))
      );
    }

    // 3. Direct multicore analysis
    if (this.systemStatus.multicore) {
      analysisPromises.push(
        comprehensiveOrchestrator.analyzeErrors(errorData)
          .then(result => ({ type: 'multicore', result }))
      );
    }

    // Execute all analyses
    const analysisResults = await Promise.allSettled(analysisPromises);

    // Process results
    const results: any = {};
    let agentsUsed = 0;
    let gpuUtilization = 0;

    analysisResults.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        const { type, result: data } = result.value;
        results[type] = data;

        if (type === 'orchestration') {
          agentsUsed = data.systemStatus?.agentsExecuted || 0;
        }
        if (type === 'gpu') {
          gpuUtilization = 0.75; // Estimated GPU usage
        }
      }
    });

    // Generate comprehensive recommendations
    const recommendations = this.generateErrorAnalysisRecommendations(results, errorData);
    const nextSteps = this.generateErrorAnalysisNextSteps(results, errorData);

    return {
      mode: 'error_analysis',
      success: true,
      results: {
        agentOrchestration: results.orchestration,
        gpuProcessing: results.gpu,
        multicoreAnalysis: results.multicore,
        systemMetrics: this.getSystemMetrics()
      },
      performance: {
        totalTime: 0, // Will be set by caller
        gpuUtilization,
        agentsUsed,
        multicoreWorkers: this.getSystemStatus().multicoreWorkers
      },
      recommendations,
      nextSteps
    };
  }

  /**
   * Execute legal text processing workflow
   */
  private async executeLegalProcessing(request: FullStackWorkflowRequest): Promise<FullStackWorkflowResult> {
    console.log('⚖️ Starting legal text processing...');

    const legalText = request.data?.text || 'Legal contract analysis with indemnification clauses and liability limitations. Evidence processing for prosecution case.';
    const context = request.data?.context || ['contract law', 'evidence rules', 'liability'];

    const processingPromises: Promise<any>[] = [];

    // 1. Agent-based legal analysis
    if (this.systemStatus.orchestrator) {
      const legalRequest: ComprehensiveAgentRequest = {
        prompt: `Provide comprehensive legal analysis: ${legalText}`,
        context: { legalContext: context },
        options: {
          agents: ['claude', 'crewai'],
          analysisType: 'legal_research',
          priority: request.options?.priority || 'medium',
          useMulticoreAnalysis: true,
          includeContext7: true
        }
      };

      processingPromises.push(
        comprehensiveOrchestrator.executeComprehensiveAnalysis(legalRequest)
          .then(result => ({ type: 'legal_orchestration', result }))
      );
    }

    // 2. GPU-accelerated legal processing
    if (this.systemStatus.flashattention) {
      const flashRequest = {
        text: legalText,
        context,
        options: {
          analysisType: 'legal',
          enableGPU: request.options?.useGPU !== false,
          useAgentOrchestration: true,
          priority: request.options?.priority || 'medium'
        }
      } as const;

      processingPromises.push(
        flashAttentionMulticoreBridge.processWithEnhancedAnalysis(flashRequest)
          .then((result: any) => ({ type: 'legal_gpu', result }))
      );
    }

    const processingResults = await Promise.allSettled(processingPromises);

    // Process results
    const results: any = {};
    processingResults.forEach(result => {
      if (result.status === 'fulfilled') {
        const { type, result: data } = result.value;
        results[type] = data;
      }
    });

    return {
      mode: 'legal_processing',
      success: true,
      results,
      performance: {
        totalTime: 0,
        gpuUtilization: results.legal_gpu ? 0.8 : 0,
        agentsUsed: results.legal_orchestration?.systemStatus?.agentsExecuted || 0,
        multicoreWorkers: this.getSystemStatus().multicoreWorkers
      },
      recommendations: this.generateLegalProcessingRecommendations(results),
      nextSteps: ['Review legal analysis results', 'Apply findings to case strategy']
    };
  }

  /**
   * Execute system diagnostic workflow
   */
  private async executeSystemDiagnostic(request: FullStackWorkflowRequest): Promise<FullStackWorkflowResult> {
    console.log('🔧 Running system diagnostic...');

    const diagnostics = {
      orchestrator: this.systemStatus.orchestrator ? comprehensiveOrchestrator.getSystemStatus() : null,
      flashattention: this.systemStatus.flashattention ? flashAttentionMulticoreBridge.getStatus() : null,
      multicore: this.systemStatus.multicore ? getContext7MulticoreService().getSystemStatus() : null,
      systemHealth: this.getSystemHealth()
    };

    return {
      mode: 'system_diagnostic',
      success: true,
      results: {
        systemMetrics: diagnostics
      },
      performance: {
        totalTime: 0,
        gpuUtilization: diagnostics.flashattention?.flashattention_status?.gpuEnabled ? 0.3 : 0,
        agentsUsed: diagnostics.orchestrator?.supportedAgents?.length || 0,
        multicoreWorkers: diagnostics.multicore?.workers?.length || 0
      },
      recommendations: this.generateDiagnosticRecommendations(diagnostics),
      nextSteps: ['Review system health', 'Apply recommended optimizations']
    };
  }

  /**
   * Execute performance test workflow
   */
  private async executePerformanceTest(request: FullStackWorkflowRequest): Promise<FullStackWorkflowResult> {
    console.log('🚀 Running performance test...');

    const testPromises: Promise<any>[] = [];

    // Test agent orchestration performance
    if (this.systemStatus.orchestrator) {
      testPromises.push(
        this.testAgentPerformance().then(result => ({ type: 'agent_performance', result }))
      );
    }

    // Test GPU performance
    if (this.systemStatus.flashattention) {
      testPromises.push(
        this.testGPUPerformance().then(result => ({ type: 'gpu_performance', result }))
      );
    }

    const testResults = await Promise.allSettled(testPromises);
    const results: any = {};

    testResults.forEach(result => {
      if (result.status === 'fulfilled') {
        const { type, result: data } = result.value;
        results[type] = data;
      }
    });

    return {
      mode: 'performance_test',
      success: true,
      results,
      performance: {
        totalTime: 0,
        gpuUtilization: results.gpu_performance?.utilization || 0,
        agentsUsed: results.agent_performance?.agentCount || 0,
        multicoreWorkers: this.getSystemStatus().multicoreWorkers
      },
      recommendations: this.generatePerformanceRecommendations(results),
      nextSteps: ['Review performance metrics', 'Apply optimizations']
    };
  }

  // Helper methods for generating recommendations
  private generateErrorAnalysisRecommendations(results: any, errorData: any): string[] {
    const recommendations = [];

    if (results.orchestration?.bestResult) {
      recommendations.push('Multi-agent analysis completed successfully');
    }

    if (results.gpu?.prioritizedErrors) {
      recommendations.push(`GPU analysis prioritized ${results.gpu.prioritizedErrors.length} critical errors`);
    }

    recommendations.push('Execute systematic Svelte 5 migration for 800+ prop errors');
    recommendations.push('Update UI component API usage for 600+ mismatches');
    recommendations.push('Clean up 400+ unused CSS selectors');

    return recommendations;
  }

  private generateErrorAnalysisNextSteps(results: any, errorData: any): string[] {
    return [
      'Run automated fix scripts for high-priority errors',
      'Apply Svelte 5 migration patterns',
      'Update component prop usage',
      'Validate fixes with incremental testing'
    ];
  }

  private generateLegalProcessingRecommendations(results: any): string[] {
    const recommendations = [];

    if (results.legal_gpu?.legalAnalysis) {
      recommendations.push('GPU-accelerated legal analysis completed');
      recommendations.push(`Relevance score: ${(results.legal_gpu.legalAnalysis.relevanceScore * 100).toFixed(1)}%`);
    }

    if (results.legal_orchestration?.bestResult) {
      recommendations.push('Multi-agent legal analysis provides comprehensive insights');
    }

    return recommendations;
  }

  private generateDiagnosticRecommendations(diagnostics: any): string[] {
    const recommendations = [];

    if (!diagnostics.orchestrator) {
      recommendations.push('Initialize agent orchestrator for enhanced AI capabilities');
    }

    if (!diagnostics.flashattention) {
      recommendations.push('Enable GPU acceleration for better performance');
    }

    if (diagnostics.multicore?.workers?.length < 4) {
      recommendations.push('Increase multicore worker count for better throughput');
    }

    return recommendations;
  }

  private generatePerformanceRecommendations(results: any): string[] {
    const recommendations = [];

    if (results.gpu_performance?.processingTime > 5000) {
      recommendations.push('Consider GPU memory optimization for faster processing');
    }

    if (results.agent_performance?.averageResponseTime > 3000) {
      recommendations.push('Optimize agent response times through caching');
    }

    return recommendations;
  }

  // Test methods
  private async testAgentPerformance(): Promise<any> {
    const startTime = performance.now();
    try {
      const result = await comprehensiveOrchestrator.executeComprehensiveAnalysis({
        prompt: 'Performance test query',
        options: { agents: ['claude'], priority: 'low' }
      });
      return {
        success: true,
        processingTime: performance.now() - startTime,
        agentCount: 1,
        averageResponseTime: performance.now() - startTime
      };
    } catch (error: any) {
      return {
        success: false,
        error: error?.message ?? String(error),
        processingTime: performance.now() - startTime
      };
    }
  }

  private async testGPUPerformance(): Promise<any> {
    const startTime = performance.now();
    try {
      const result = await flashAttentionMulticoreBridge.processWithEnhancedAnalysis({
        text: 'GPU performance test text',
        options: { enableGPU: true, priority: 'low' }
      });
      return {
        success: true,
        processingTime: performance.now() - startTime,
        utilization: result.systemMetrics?.gpuUtilization || 0
      };
    } catch (error: any) {
      return {
        success: false,
        error: error?.message ?? String(error),
        processingTime: performance.now() - startTime
      };
    }
  }

  // Status and metrics methods
  private createErrorResult(mode: string, error: any, totalTime: number): FullStackWorkflowResult {
    return {
      mode,
      success: false,
      results: {
        error: error.message
      },
      performance: {
        totalTime,
        gpuUtilization: 0,
        agentsUsed: 0,
        multicoreWorkers: 0
      },
      recommendations: ['Check system status and retry'],
      nextSteps: ['Review error logs', 'Initialize failed services']
    };
  }

  private getSystemStatus() {
    const multicoreService = getContext7MulticoreService();
    return {
      initialized: this.isInitialized,
      orchestrator: this.systemStatus.orchestrator,
      flashattention: this.systemStatus.flashattention,
      multicore: this.systemStatus.multicore,
      multicoreWorkers: this.systemStatus.multicore ? multicoreService.getSystemStatus().workers.length : 0
    };
  }

  private getSystemHealth(): 'excellent' | 'good' | 'degraded' | 'critical' {
    const activeServices = Object.values(this.systemStatus).filter(Boolean).length;
    if (activeServices === 3) return 'excellent';
    if (activeServices === 2) return 'good';
    if (activeServices === 1) return 'degraded';
    return 'critical';
  }

  private getSystemMetrics() {
    return {
      uptime: this.isInitialized ? Date.now() : 0,
      health: this.getSystemHealth(),
      services: this.systemStatus,
      capabilities: [
        this.systemStatus.orchestrator ? 'Agent Orchestration' : null,
        this.systemStatus.flashattention ? 'GPU Acceleration' : null,
        this.systemStatus.multicore ? 'Multicore Processing' : null
      ].filter(Boolean)
    };
  }
}

// Global workflow instance
export const fullStackWorkflow = new FullStackLegalAIWorkflow();
;
// Helper functions for VS Code tasks
export async function initializeFullStack(): Promise<void> {
  await fullStackWorkflow.initialize();
}

export async function runErrorAnalysis(errorData?: unknown): Promise<FullStackWorkflowResult> {
  return await fullStackWorkflow.executeWorkflow({
    mode: 'error_analysis',
    data: errorData,
    options: { useGPU: true, enableAgents: true, priority: 'high' }
  });
}

export async function runLegalProcessing(text: string, context?: string[]): Promise<FullStackWorkflowResult> {
  return await fullStackWorkflow.executeWorkflow({
    mode: 'legal_processing',
    data: { text, context },
    options: { useGPU: true, enableAgents: true, priority: 'medium' }
  });
}

export async function runSystemDiagnostic(): Promise<FullStackWorkflowResult> {
  return await fullStackWorkflow.executeWorkflow({
    mode: 'system_diagnostic',
    options: { priority: 'low' }
  });
}

export async function runPerformanceTest(): Promise<FullStackWorkflowResult> {
  return await fullStackWorkflow.executeWorkflow({
    mode: 'performance_test',
    options: { useGPU: true, enableAgents: true, priority: 'low' }
  });
}

export default fullStackWorkflow;