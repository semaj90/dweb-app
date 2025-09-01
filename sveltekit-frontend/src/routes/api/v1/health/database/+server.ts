import type { RequestHandler } from './$types.js';

/**
 * PostgreSQL + pgvector Health Check API
 * Validates database connectivity for startup validation
 */

import { json, error } from '@sveltejs/kit';

import { ensureError } from '$lib/utils/ensure-error';
import { checkDatabaseHealth, dbHealthChecker } from '$lib/server/db/health-check';
import { URL } from "url";

export const GET: RequestHandler = async ({ url }) => {
  const startTime = Date.now();
  
  try {
    const action = url.searchParams.get('action') || 'health';
    
    switch (action) {
      case 'health':
        const health = await checkDatabaseHealth();
        return json({
          success: true,
          health,
          metadata: {
            processingTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
            endpoint: '/api/v1/health/database'
          }
        });
        
      case 'metrics':
        const metrics = await dbHealthChecker.getDatabaseMetrics();
        return json({
          success: true,
          data: metrics,
          metadata: {
            processingTime: Date.now() - startTime,
            timestamp: new Date().toISOString()
          }
        });
        
      case 'validate':
        const isValid = await dbHealthChecker.validateSchema();
        return json({
          success: true,
          data: {
            schemaValid: isValid.valid,
            missingTables: isValid.missingTables
          },
          metadata: {
            processingTime: Date.now() - startTime
          }
        });
        
      case 'vector':
        const vectorTest = await dbHealthChecker.testVectorOperations();
        return json({
          success: true,
          data: {
            vectorOperationsWorking: vectorTest,
            pgvectorEnabled: await dbHealthChecker.isPgVectorEnabled()
          },
          metadata: {
            processingTime: Date.now() - startTime
          }
        });
        
      default:
        return error(400, ensureError({ 
          message: `Invalid action: ${action}. Available: health, metrics, validate, vector` 
        }));
    }
    
  } catch (err: any) {
    console.error('Database health check error:', err);
    return json({
      success: false,
      error: err instanceof Error ? err.message : 'Database health check failed',
      metadata: {
        processingTime: Date.now() - startTime,
        timestamp: new Date().toISOString()
      }
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  
  try {
    const body = await request.json();
    const { action } = body;
    
    switch (action) {
      case 'clear_cache':
        dbHealthChecker.clearCache();
        return json({
          success: true,
          message: 'Health check cache cleared',
          metadata: {
            processingTime: Date.now() - startTime
          }
        });
        
      case 'force_check':
        const health = await dbHealthChecker.checkHealth(false); // Force fresh check
        return json({
          success: true,
          data: health,
          metadata: {
            processingTime: Date.now() - startTime,
            cached: false
          }
        });
        
      default:
        return error(400, ensureError({ 
          message: `Invalid action: ${action}. Available: clear_cache, force_check` 
        }));
    }
    
  } catch (err: any) {
    console.error('Database health check POST error:', err);
    return json({
      success: false,
      error: err instanceof Error ? err.message : 'Database health check failed',
      metadata: {
        processingTime: Date.now() - startTime
      }
    }, { status: 500 });
  }
};