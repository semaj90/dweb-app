import type { RequestHandler } from './$types';

/**
 * MCP Health Check API - Test Database Integration
 * Simple endpoint to verify MCP tools and database connectivity
 */

import { json } from '@sveltejs/kit';
import * as casesMCP from '../../../../../lib/mcp/cases.mcp';

export const GET: RequestHandler = async ({ getClientAddress }) => {
  try {
    console.log('🔍 Testing MCP Tools & Database Integration...');
    
    // Test database connectivity through MCP health check
    const healthResult = await casesMCP.healthCheck();
    
    const response = {
      success: true,
      timestamp: Date.now(),
      services: {
        mcp: 'operational',
        database: healthResult.status,
        drizzle: 'connected'
      },
      details: {
        ...healthResult.details,
        clientAddress: getClientAddress(),
        endpoint: '/api/v1/mcp/health'
      },
      message: 'MCP + PostgreSQL + Drizzle ORM integration is working!'
    };

    console.log('✅ MCP Health Check Passed');
    return json(response);

  } catch (error: any) {
    console.error('❌ MCP Health Check Failed:', error);
    
    return json({
      success: false,
      timestamp: Date.now(),
      error: error instanceof Error ? error.message : 'Unknown error',
      services: {
        mcp: 'error',
        database: 'unknown',
        drizzle: 'error'
      }
    }, { status: 500 });
  }
};