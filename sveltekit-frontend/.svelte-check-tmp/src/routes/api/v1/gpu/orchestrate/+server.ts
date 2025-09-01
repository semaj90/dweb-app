import type { RequestHandler } from './$types';

/**
 * GPU Orchestration API - Advanced Task Dispatch & Automation
 * Handles legal document analysis, autosolve, and GPU task routing
 */

import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator.js';
import { GEMMA3_LEGAL_CONFIG } from '$lib/config/gemma3-legal-config.js';

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const { action, data, config } = await request.json();
    
    switch (action) {
      case 'legal_analysis':
        return handleLegalAnalysis(data, config);
      
      case 'document_processing':
        return handleDocumentProcessing(data, config);
      
      case 'autosolve':
        return handleAutosolve(data, config);
      
      case 'gpu_task':
        return handleGPUTask(data, config);
      
      case 'cluster_status':
        return handleClusterStatus();
      
      default:
        return json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('GPU orchestration error:', error);
    return json({
      error: 'GPU orchestration failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

async function handleLegalAnalysis(data: any, config: any): Promise<any> {
  const { document, context, options } = data;
  
  // Enhanced legal analysis with Gemma3 configuration
  const analysisConfig = {
    ...GEMMA3_LEGAL_CONFIG.generation,
    ...config,
    useGPU: true,
    useRAG: options?.includeRAG !== false,
    model: 'gemma3-legal'
  };
  
  const result = await mcpGPUOrchestrator.processLegalDocument(document, {
    caseId: context?.caseId,
    userId: context?.userId,
    includeRAG: options?.includeRAG,
    includeGraph: options?.includeGraph,
    generateSummary: options?.generateSummary
  });
  
  // Add legal-specific post-processing
  if (result.success && options?.extractEntities) {
    result.result.entities = extractLegalEntities(document);
  }
  
  if (result.success && options?.riskAssessment) {
    result.result.riskAssessment = await performRiskAssessment(result.result);
  }
  
  return json({
    success: result.success,
    analysis: result.result,
    metrics: result.metrics,
    recommendations: result.recommendations,
    timestamp: new Date().toISOString()
  });
}

async function handleDocumentProcessing(data: any, config: any): Promise<any> {
  const { files, context, options } = data;
  
  const results = [];
  
  for (const file of files) {
    const result = await mcpGPUOrchestrator.dispatchGPUTask({
      id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'document_processing',
      priority: 'high',
      data: { file },
      context: {
        userId: context?.userId,
        caseId: context?.caseId,
        documentId: context?.documentId
      },
      config: {
        useGPU: true,
        useRAG: options?.enableRAG !== false,
        protocol: 'http'
      }
    });
    
    results.push(result);
  }
  
  return json({
    success: true,
    results,
    processed: results.length,
    failed: results.filter(r => !r.success).length,
    timestamp: new Date().toISOString()
  });
}

async function handleAutosolve(data: any, config: any): Promise<any> {
  const { threshold, forceRun, includeClusterMetrics } = data;
  
  // Trigger autosolve maintenance cycle
  const result = await mcpGPUOrchestrator.triggerAutosolve({
    threshold: threshold || 5,
    includeClusterMetrics: includeClusterMetrics !== false,
    forceRun: forceRun === true
  });
  
  // Get current cluster status for context
  const clusterStatus = await mcpGPUOrchestrator.getClusterStatus();
  
  return json({
    success: result.success,
    autosolve: {
      result: result.result,
      metrics: result.metrics,
      recommendations: result.recommendations
    },
    cluster: clusterStatus,
    timestamp: new Date().toISOString()
  });
}

async function handleGPUTask(data: any, config: any): Promise<any> {
  const { taskType, taskData, priority, context } = data;
  
  const result = await mcpGPUOrchestrator.dispatchGPUTask({
    id: `gpu_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    type: taskType,
    priority: priority || 'medium',
    data: taskData,
    context,
    config: {
      useGPU: true,
      ...config
    }
  });
  
  return json({
    success: result.success,
    taskId: result.taskId,
    result: result.result,
    metrics: result.metrics,
    error: result.error,
    recommendations: result.recommendations,
    timestamp: new Date().toISOString()
  });
}

async function handleClusterStatus(): Promise<any> {
  const clusterStatus = await mcpGPUOrchestrator.getClusterStatus();
  
  return json({
    cluster: clusterStatus,
    timestamp: new Date().toISOString()
  });
}

// Helper functions for legal processing
function extractLegalEntities(document: string) {
  const entities = {
    parties: [],
    dates: [],
    citations: [],
    amounts: [],
    clauses: []
  };
  
  // Import patterns from config
  const patterns = {
    parties: [
      /\b(plaintiff|defendant|appellant|appellee|petitioner|respondent)\b/gi,
      /\b([A-Z][a-z]+ (?:v\.|vs\.|versus) [A-Z][a-z]+)\b/g,
      /\b([A-Z][A-Za-z\s&,.]+ (?:Inc\.|LLC|Corp\.|Corporation|Company|Co\.))\b/g
    ],
    dates: [
      /\b(\d{1,2}\/\d{1,2}\/\d{4})\b/g,
      /\b([A-Z][a-z]+ \d{1,2}, \d{4})\b/g,
      /\b(\d{4}-\d{2}-\d{2})\b/g
    ],
    citations: [
      /\b(\d+ [A-Z][a-z.]+ \d+(?:, \d+)? \(\d{4}\))\b/g,
      /\b(\d+ U\.S\.C\. (?:§ )?\d+(?:\([a-z0-9]+\))?)\b/g
    ],
    amounts: [
      /\$[\d,]+(?:\.\d{2})?/g,
      /\b(\d+(?:,\d{3})*) dollars?\b/gi
    ],
    clauses: [
      /\b(indemnification|limitation of liability|force majeure|termination|confidentiality)\b/gi
    ]
  };
  
  // Extract entities using patterns
  for (const [category, categoryPatterns] of Object.entries(patterns)) {
    for (const pattern of categoryPatterns) {
      const matches = document.match(pattern) || [];
      entities[category as keyof typeof entities].push(...matches);
    }
  }
  
  return entities;
}

async function performRiskAssessment(analysisResult: any): Promise<any> {
  // Simplified risk assessment based on analysis
  const risks = {
    financial: 0,
    legal: 0,
    operational: 0,
    reputational: 0,
    overall: 0
  };
  
  const text = analysisResult.text || analysisResult.summary || '';
  
  // Risk indicators
  const riskKeywords = {
    financial: ['liability', 'damages', 'penalty', 'fine', 'cost', 'expense'],
    legal: ['violation', 'breach', 'non-compliance', 'lawsuit', 'litigation'],
    operational: ['disruption', 'delay', 'failure', 'inability', 'restriction'],
    reputational: ['public', 'media', 'reputation', 'image', 'scandal']
  };
  
  // Calculate risk scores based on keyword frequency
  for (const [category, keywords] of Object.entries(riskKeywords)) {
    const score = keywords.reduce((sum, keyword) => {
      const matches = (text.toLowerCase().match(new RegExp(keyword, 'g')) || []).length;
      return sum + matches;
    }, 0);
    
    risks[category as keyof typeof risks] = Math.min(10, score); // Cap at 10
  }
  
  // Calculate overall risk
  risks.overall = Math.round(
    (risks.financial + risks.legal + risks.operational + risks.reputational) / 4
  );
  
  return {
    scores: risks,
    level: risks.overall >= 7 ? 'high' : risks.overall >= 4 ? 'medium' : 'low',
    recommendations: generateRiskRecommendations(risks)
  };
}

function generateRiskRecommendations(risks: any): string[] {
  const recommendations = [];
  
  if (risks.financial >= 7) {
    recommendations.push('Obtain financial insurance or bonding');
    recommendations.push('Establish escrow or security deposits');
  }
  
  if (risks.legal >= 7) {
    recommendations.push('Consult specialized legal counsel');
    recommendations.push('Review compliance requirements');
  }
  
  if (risks.operational >= 7) {
    recommendations.push('Develop contingency plans');
    recommendations.push('Implement monitoring systems');
  }
  
  if (risks.reputational >= 7) {
    recommendations.push('Prepare crisis communication plan');
    recommendations.push('Implement stakeholder engagement strategy');
  }
  
  if (risks.overall >= 7) {
    recommendations.push('Consider risk transfer mechanisms');
    recommendations.push('Establish risk monitoring dashboard');
  }
  
  return recommendations;
}

// Health check endpoint
export const GET: RequestHandler = async () => {
  try {
    const clusterStatus = await mcpGPUOrchestrator.getClusterStatus();
    
    return json({
      status: 'healthy',
      service: 'gpu-orchestrator',
      cluster: clusterStatus,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    return json({
      status: 'unhealthy',
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};