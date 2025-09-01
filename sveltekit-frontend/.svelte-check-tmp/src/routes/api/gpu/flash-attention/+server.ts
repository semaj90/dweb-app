import { FlashAttention2RTX3060Service } from '$lib/services/flashattention2-rtx3060.js';
import type { RequestHandler } from './$types';


// Initialize FlashAttention2 service with RTX 3060 Ti configuration
const flashAttentionService = new FlashAttention2RTX3060Service({
  enableGPUOptimization: true,
  memoryOptimization: 'balanced',
  batchSize: 8,
  maxSequenceLength: 2048
});

// POST /api/gpu/flash-attention - Process legal text with FlashAttention2
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const { text, context = [], analysisType = 'legal' } = body;
    
    if (!text) {
      return json({
        success: false,
        error: 'Text is required for FlashAttention2 processing'
      }, { status: 400 });
    }
    
    const startTime = performance.now();
    
    // Process with FlashAttention2 RTX 3060 Ti service
    const result = await flashAttentionService.processLegalText(text, context, analysisType);
    
    const processingTime = performance.now() - startTime;
    
    return json({
      success: true,
      processingTime,
      memoryUsage: result.memoryUsage,
      confidence: result.confidence,
      result: {
        embeddings: Array.from(result.embeddings.slice(0, 10)), // First 10 values for demo
        attentionWeights: Array.from(result.attentionWeights.slice(0, 16)), // First 4x4 matrix
        legalAnalysis: result.legalAnalysis,
        processingTime: result.processingTime
      },
      metadata: {
        rtx_3060_ti: true,
        flash_attention2: true,
        legal_domain: analysisType === 'legal',
        sequence_length: text.length,
        context_count: context.length
      },
      timestamp: Date.now()
    });
    
  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      timestamp: Date.now()
    }, { status: 500 });
  }
};

// GET /api/gpu/flash-attention - Get FlashAttention2 service status
export const GET: RequestHandler = async () => {
  try {
    const status = flashAttentionService.getServiceStatus();
    
    return json({
      success: true,
      status,
      capabilities: {
        rtx_3060_ti_optimized: true,
        memory_efficient: true,
        legal_domain_specialized: true,
        supported_operations: ['embedding', 'similarity', 'legal_analysis', 'error_processing']
      },
      timestamp: Date.now()
    });
    
  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      timestamp: Date.now()
    }, { status: 500 });
  }
};

export const prerender = false;