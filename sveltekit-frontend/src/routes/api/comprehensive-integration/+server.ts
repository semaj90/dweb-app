// Comprehensive Integration API - Full System Integration
// Combines database orchestrator, Context7 MCP, event loops, and fetch operations

import { json, type RequestHandler } from '@sveltejs/kit';
import {
  databaseOrchestrator,
  synthesizeEvidence,
  performLegalResearch,
  optimizeSystem,
} from '$lib/services/comprehensive-database-orchestrator';
import { db } from '$lib/server/db/drizzle';
import { cases, evidence, legalDocuments, personsOfInterest } from '$lib/server/db/schema-postgres';
import { eq, sql, desc, and, or  } from "drizzle-orm";

// Full system integration endpoints
const INTEGRATION_ENDPOINTS = {
  ollama: 'http://localhost:11434',
  enhanced_rag: 'http://localhost:8094',
  upload_service: 'http://localhost:8093',
  recommendation_service: 'http://localhost:8096',
  mcp_wrapper: 'http://localhost:4000',
  mcp_legal: 'http://localhost:4001',
};

// GET /api/comprehensive-integration - System overview
export const GET: RequestHandler = async () => {
  try {
    // Check all service health
    const healthChecks = await Promise.all(
      Object.entries(INTEGRATION_ENDPOINTS).map(async ([name, endpoint]) => {
        try {
          const controller = new AbortController();
          const t = setTimeout(() => controller.abort(), 3000);
          const response = await fetch(`${endpoint}/health`, { signal: controller.signal });
          clearTimeout(t);
          return {
            service: name,
            endpoint,
            status: response.ok ? 'healthy' : 'unhealthy',
            response_code: response.status,
          };
        } catch (error) {
          return {
            service: name,
            endpoint,
            status: 'error',
            error: error.message,
          };
        }
      })
    );

    // Get database statistics
    const [caseCount, evidenceCount, documentCount, poiCount] = await Promise.all([
      db.select({ count: sql`count(*)` }).from(cases),
      db.select({ count: sql`count(*)` }).from(evidence),
      db.select({ count: sql`count(*)` }).from(legalDocuments),
      db.select({ count: sql`count(*)` }).from(personsOfInterest),
    ]);

    // Get orchestrator status
    const orchestratorStatus = databaseOrchestrator.getStatus();

    return json({
      success: true,
      system_overview: {
        services: healthChecks,
        healthy_services: healthChecks.filter((s) => s.status === 'healthy').length,
        total_services: healthChecks.length,
        database_stats: {
          cases: caseCount[0].count,
          evidence: evidenceCount[0].count,
          documents: documentCount[0].count,
          persons_of_interest: poiCount[0].count,
          total_records:
            Number(caseCount[0].count) +
            Number(evidenceCount[0].count) +
            Number(documentCount[0].count) +
            Number(poiCount[0].count),
        },
        orchestrator: {
          running: orchestratorStatus.isRunning,
          active_loops: orchestratorStatus.activeLoops,
          active_conditions: orchestratorStatus.activeConditions,
          queue_length: orchestratorStatus.queueLength,
        },
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// POST /api/comprehensive-integration - Execute comprehensive operations
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { operation, data } = await request.json();

    switch (operation) {
      case 'full_document_processing':
        return await processDocumentComprehensive(data);

      case 'intelligent_case_analysis':
        return await analyzeCase(data);

      case 'evidence_synthesis':
        return await synthesizeEvidence(data);

      case 'legal_research':
        return await performLegalResearch(data);

      case 'system_optimization':
        return await optimizeSystem();

      case 'real_time_analysis':
        return await performRealTimeAnalysis(data);

      case 'context7_integration':
        return await integrateContext7(data);

      default:
        return json(
          {
            success: false,
            error: `Unknown operation: ${operation}`,
          },
          { status: 400 }
        );
    }
  } catch (error) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// Comprehensive document processing with all integrations
async function processDocumentComprehensive(data: any) {
  const { document_content, document_type, case_id, metadata } = data;

  try {
    // Step 1: Save document to database via orchestrator
    const savedDocument = await databaseOrchestrator.saveToDatabase(
      {
        content: document_content,
        document_type,
        case_id,
        metadata,
        processing_status: 'started',
        created_at: new Date(),
      },
      'legal_documents'
    );

    // Step 2: Process with Enhanced RAG
    const ragResponse = await fetch(
      `${INTEGRATION_ENDPOINTS.enhanced_rag}/api/ai/process-document`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: document_content,
          document_type,
          practice_area: metadata?.practice_area || 'general',
          jurisdiction: metadata?.jurisdiction || 'US',
        }),
      }
    );

    const ragResult = ragResponse.ok ? await ragResponse.json() : null;

    // Step 3: Generate embeddings via Ollama
    const embeddingResponse = await fetch(`${INTEGRATION_ENDPOINTS.ollama}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: document_content.substring(0, 2000), // Limit for embedding
      }),
    });

    const embeddingResult = embeddingResponse.ok ? await embeddingResponse.json() : null;

    // Step 4: Context7 analysis
    const context7Response = await fetch(`${INTEGRATION_ENDPOINTS.mcp_legal}/tools/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'legal_rag_query',
        arguments: {
          query: `Analyze this ${document_type}: ${document_content.substring(0, 500)}`,
          documentTypes: [document_type],
          maxResults: 5,
        },
      }),
    });

    const context7Result = context7Response.ok ? await context7Response.json() : null;

    // Step 5: Update document with all results
    const finalResult = {
      document_id: savedDocument.id,
      rag_analysis: ragResult,
      embeddings: embeddingResult?.embedding,
      context7_analysis: context7Result,
      processing_status: 'completed',
      processed_at: new Date(),
    };

    // Save final results to database
    await databaseOrchestrator.saveToDatabase(finalResult, 'document_processing_results');

    // Trigger real-time event
    databaseOrchestrator.emit('document:fully_processed', {
      document_id: savedDocument.id,
      case_id,
      processing_time: Date.now(),
      analysis_count: Object.keys(finalResult).filter((k) => k.includes('analysis')).length,
    });

    return json({
      success: true,
      message: 'Document processed comprehensively',
      document_id: savedDocument.id,
      results: finalResult,
      processing_steps: [
        'database_save',
        'rag_analysis',
        'embeddings',
        'context7_analysis',
        'final_save',
      ],
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    // Log error via orchestrator
    await databaseOrchestrator.saveToDatabase(
      {
        operation: 'process_document_comprehensive',
        error: error.message,
        input_data: data,
        timestamp: new Date(),
        status: 'failed',
      },
      'operation_logs'
    );

    throw error;
  }
}

// Intelligent case analysis with AI and vector search
async function analyzeCase(data: any) {
  const { case_id, analysis_type } = data;

  try {
    // Get case data
    const caseData = await databaseOrchestrator.queryDatabase(eq(cases.id, case_id), 'cases');

    if (!caseData.length) {
      throw new Error(`Case not found: ${case_id}`);
    }

    const case_ = caseData[0];

    // Get related evidence
    const relatedEvidence = await databaseOrchestrator.queryDatabase(
      eq(evidence.caseId, case_id),
      'evidence'
    );

    // Get related documents
    const relatedDocuments = await databaseOrchestrator.queryDatabase(
      eq((legalDocuments as any).caseId ?? (legalDocuments as any).case_id, case_id),
      'legal_documents'
    );

    // Prepare analysis context
    const analysisContext = {
      case: case_,
      evidence_count: relatedEvidence.length,
      document_count: relatedDocuments.length,
      evidence_summary: relatedEvidence.map((e) => ({
        type: e.type,
        description: e.description,
        date: e.uploadedAt,
      })),
      analysis_type,
    };

    // Call Context7 for case summary
    const summaryResponse = await fetch(`${INTEGRATION_ENDPOINTS.mcp_legal}/tools/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'get_case_summary',
        arguments: {
          caseId: case_id,
          includeEvidence: true,
        },
      }),
    });

    const summaryResult = summaryResponse.ok ? await summaryResponse.json() : null;

    // Enhanced RAG analysis
    const ragResponse = await fetch(`${INTEGRATION_ENDPOINTS.enhanced_rag}/api/ai/vector-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `Legal analysis for case: ${case_.title} ${case_.description}`,
        model: 'gemma3-legal',
        limit: 10,
        filters: { case_type: case_.type },
      }),
    });

    const ragResult = ragResponse.ok ? await ragResponse.json() : null;

    // Compile comprehensive analysis
    const comprehensiveAnalysis = {
      case_id,
      analysis_type,
      context: analysisContext,
      ai_summary: summaryResult,
      vector_analysis: ragResult,
      recommendations: generateCaseRecommendations(analysisContext, summaryResult, ragResult),
      generated_at: new Date(),
    };

    // Save analysis to database
    await databaseOrchestrator.saveToDatabase(comprehensiveAnalysis, 'case_analyses');

    return json({
      success: true,
      message: 'Case analysis completed',
      case_id,
      analysis: comprehensiveAnalysis,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    throw error;
  }
}

// Real-time analysis with event loops
async function performRealTimeAnalysis(data: any) {
  const { query, real_time_duration = 30000 } = data; // 30 second default

  try {
    const analysisResults = [];
    const startTime = Date.now();

    // Set up real-time condition
    const realtimeCondition = {
      id: `realtime_analysis_${Date.now()}`,
      type: 'timer',
      condition: { interval: 5000 }, // Every 5 seconds
      action: 'perform_realtime_query',
      isActive: true,
      metadata: { query, start_time: startTime, duration: real_time_duration },
    };

    databaseOrchestrator.addCondition(realtimeCondition);

    // Listen for analysis results
    const resultCollector = (result: any) => {
      analysisResults.push({
        ...result,
        timestamp: new Date().toISOString(),
      });
    };

    databaseOrchestrator.on('realtime:analysis_result', resultCollector);

    // Wait for the specified duration
    await new Promise((resolve) => setTimeout(resolve, real_time_duration));

    // Clean up
    databaseOrchestrator.removeCondition(realtimeCondition.id);
    databaseOrchestrator.off('realtime:analysis_result', resultCollector);

    return json({
      success: true,
      message: 'Real-time analysis completed',
      duration_ms: real_time_duration,
      results: analysisResults,
      result_count: analysisResults.length,
      query,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    throw error;
  }
}

// Context7 MCP integration
async function integrateContext7(data: any) {
  const { operation, context7_data } = data;

  try {
    // Call multiple Context7 services
    const [wrapperResult, legalResult] = await Promise.all([
      fetch(`${INTEGRATION_ENDPOINTS.mcp_wrapper}/tools/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'get_context7_status',
          arguments: {},
        }),
      }),
      fetch(`${INTEGRATION_ENDPOINTS.mcp_legal}/tools/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'legal_rag_query',
          arguments: context7_data,
        }),
      }),
    ]);

    const context7Results = {
      wrapper_status: wrapperResult.ok ? await wrapperResult.json() : null,
      legal_query: legalResult.ok ? await legalResult.json() : null,
      integration_timestamp: new Date().toISOString(),
    };

    // Save to database via orchestrator
    await databaseOrchestrator.saveToDatabase(
      {
        operation: 'context7_integration',
        input_data: context7_data,
        results: context7Results,
        timestamp: new Date(),
      },
      'context7_integrations'
    );

    return json({
      success: true,
      message: 'Context7 integration completed',
      results: context7Results,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    throw error;
  }
}

// Helper functions
function generateCaseRecommendations(context: any, summary: any, vectorAnalysis: any) {
  const recommendations = [];

  if (context.evidence_count === 0) {
    recommendations.push({
      type: 'evidence_collection',
      priority: 'high',
      message: 'No evidence collected yet. Consider gathering supporting documentation.',
    });
  }

  if (context.document_count < 3) {
    recommendations.push({
      type: 'document_review',
      priority: 'medium',
      message: 'Limited documents in case. Review for additional relevant materials.',
    });
  }

  if (vectorAnalysis?.results?.length > 0) {
    recommendations.push({
      type: 'precedent_analysis',
      priority: 'medium',
      message: `Found ${vectorAnalysis.results.length} similar cases for reference.`,
    });
  }

  return recommendations;
}

