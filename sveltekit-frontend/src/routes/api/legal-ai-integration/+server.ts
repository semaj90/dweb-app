import { json } from '@sveltejs/kit';
import { legalAIIntegration } from '$lib/services/quic-legal-ai-integration.js';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// GET: System status and health
export const GET: RequestHandler = async ({ url }) => {
  const action = url.searchParams.get('action');
  
  try {
    switch (action) {
      case 'status':
        const status = legalAIIntegration.getStatus();
        return json({
          success: true,
          data: status,
          timestamp: new Date().toISOString()
        });
        
      case 'health':
        const health = await legalAIIntegration.getSystemHealth();
        return json({
          success: true,
          data: health,
          timestamp: new Date().toISOString()
        });
        
      case 'autosolve':
        const autosolveResult = await legalAIIntegration.runAutosolve();
        return json({
          success: true,
          data: autosolveResult,
          timestamp: new Date().toISOString()
        });
        
      default:
        return json({
          success: true,
          data: {
            message: 'QUIC-Enhanced Legal AI System',
            version: '1.0.0',
            endpoints: {
              'GET ?action=status': 'Get system status',
              'GET ?action=health': 'Get system health',
              'GET ?action=autosolve': 'Run autosolve cycle',
              'POST': 'Process legal document'
            }
          }
        });
    }
  } catch (error: any) {
    console.error('Legal AI Integration API error:', error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

// POST: Process legal documents
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const { content, options = {} } = body;
    
    if (!content || typeof content !== 'string') {
      return json({
        success: false,
        error: 'Content is required and must be a string'
      }, { status: 400 });
    }
    
    // Process document with QUIC acceleration
    const result = await legalAIIntegration.processLegalDocument(content, {
      useQuic: options.useQuic !== false,
      enableAutosolve: options.enableAutosolve === true,
      generateSuggestions: options.generateSuggestions !== false,
      ...options
    });
    
    return json({
      success: true,
      data: {
        ...result,
        metadata: {
          contentLength: content.length,
          quicEnabled: options.useQuic !== false,
          processingMode: 'ultra_fast_quic'
        }
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('Document processing error:', error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Document processing failed',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

// PUT: Update integration configuration
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const { config } = body;
    
    // For now, just return the current config
    // In a full implementation, you'd update the integration config
    const currentStatus = legalAIIntegration.getStatus();
    
    return json({
      success: true,
      data: {
        message: 'Configuration update received',
        current: currentStatus.config,
        requested: config
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('Config update error:', error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Config update failed',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};