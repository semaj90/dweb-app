import type { RequestHandler } from './$types.js';

/**
 * MCP API Endpoint - Tool Call Router
 * Handles all MCP tool invocations from XState machine services
 */

import { json } from '@sveltejs/kit';

// Import MCP Tools
import * as casesMCP from '../../../../../lib/mcp/cases.mcp.js';
import { URL } from "url";

// MCP Tool Registry
const MCP_TOOLS = {
  // Cases management tools
  'cases.loadCase': casesMCP.loadCase,
  'cases.createCase': casesMCP.createCase,
  'cases.updateCase': casesMCP.updateCase,
  'cases.addEvidence': casesMCP.addEvidence,
  'cases.searchCases': casesMCP.searchCases,
  'cases.getUserCases': casesMCP.getUserCases,
  'cases.healthCheck': casesMCP.healthCheck,
  
  // Future MCP tools can be added here:
  // 'documents.processDocument': documentsMCP.processDocument,
  // 'evidence.analyzeEvidence': evidenceMCP.analyzeEvidence,
  // 'vector.semanticSearch': vectorMCP.semanticSearch,
} as const;

type MCPToolName = keyof typeof MCP_TOOLS;

export interface MCPCallRequest {
  tool: MCPToolName;
  args: Record<string, any>;
  metadata?: {
    requestId?: string;
    userId?: string;
    timestamp?: number;
  };
}

export interface MCPCallResponse {
  success: boolean;
  result?: any;
  error?: string;
  metadata?: {
    requestId?: string;
    executionTime?: number;
    tool: string;
    timestamp: number;
  };
}

export const POST: RequestHandler = async ({ request, getClientAddress, url }) => {
  const startTime = Date.now();
  let requestBody: MCPCallRequest;

  try {
    requestBody = await request.json();
  } catch (error: any) {
    return json<MCPCallResponse>({
      success: false,
      error: 'Invalid JSON in request body',
      metadata: {
        tool: 'unknown',
        timestamp: Date.now(),
        executionTime: Date.now() - startTime
      }
    }, { status: 400 });
  }

  const { tool, args = {}, metadata = {} } = requestBody;

  // Validate tool name
  if (!tool || !(tool in MCP_TOOLS)) {
    return json<MCPCallResponse>({
      success: false,
      error: `Unknown MCP tool: ${tool}. Available tools: ${Object.keys(MCP_TOOLS).join(', ')}`,
      metadata: {
        requestId: metadata.requestId,
        tool: tool || 'unknown',
        timestamp: Date.now(),
        executionTime: Date.now() - startTime
      }
    }, { status: 400 });
  }

  // Add request metadata for logging and tracing
  const requestMetadata = {
    requestId: metadata.requestId || `mcp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    userId: metadata.userId,
    clientAddress: getClientAddress(),
    userAgent: request.headers.get('user-agent'),
    timestamp: Date.now(),
    tool
  };

  console.log('🔧 MCP Tool Call:', {
    tool,
    args: Object.keys(args),
    requestId: requestMetadata.requestId,
    userId: requestMetadata.userId
  });

  try {
    // Get the tool function
    const toolFunction = MCP_TOOLS[tool];
    
    // Execute the tool with arguments
    const result = await toolFunction(args);

    const executionTime = Date.now() - startTime;

    console.log('✅ MCP Tool Success:', {
      tool,
      requestId: requestMetadata.requestId,
      executionTime: `${executionTime}ms`
    });

    return json<MCPCallResponse>({
      success: true,
      result,
      metadata: {
        requestId: requestMetadata.requestId,
        executionTime,
        tool,
        timestamp: Date.now()
      }
    });

  } catch (error: any) {
    const executionTime = Date.now() - startTime;
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';

    console.error('❌ MCP Tool Error:', {
      tool,
      requestId: requestMetadata.requestId,
      error: errorMessage,
      executionTime: `${executionTime}ms`
    });

    return json<MCPCallResponse>({
      success: false,
      error: errorMessage,
      metadata: {
        requestId: requestMetadata.requestId,
        executionTime,
        tool,
        timestamp: Date.now()
      }
    }, { status: 500 });
  }
};

// Health check endpoint for MCP tools
export const GET: RequestHandler = async ({ url }) => {
  try {
    // Test database connectivity through health check tool
    const healthResult = await casesMCP.healthCheck();
    
    const response = {
      status: 'operational',
      timestamp: Date.now(),
      tools: {
        available: Object.keys(MCP_TOOLS),
        count: Object.keys(MCP_TOOLS).length
      },
      database: healthResult,
      endpoint: url.pathname
    };

    return json(response);

  } catch (error: any) {
    return json({
      status: 'error',
      timestamp: Date.now(),
      error: error instanceof Error ? error.message : 'Health check failed',
      endpoint: url.pathname
    }, { status: 500 });
  }
};