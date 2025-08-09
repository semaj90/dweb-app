// @ts-nocheck
// Cache Management API - Production-ready endpoints for multi-layer cache system
// Provides REST interface for cache operations, health checks, and statistics

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { cachingService, getCacheStats, getCacheHealth } from '$lib/services/caching-service';
import { z } from 'zod';

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

const CacheGetSchema = z.object({
  key: z.string().min(1).max(255),
  options: z.object({
    ttl: z.number().optional(),
    tags: z.array(z.string()).optional(),
    priority: z.enum(['low', 'medium', 'high']).optional(),
    layer: z.enum(['memory', 'loki', 'redis', 'postgres', 'all']).optional()
  }).optional()
});

const CacheSetSchema = z.object({
  key: z.string().min(1).max(255),
  value: z.any(),
  options: z.object({
    ttl: z.number().min(1000).max(86400000).optional(), // 1 second to 24 hours
    tags: z.array(z.string()).optional(),
    priority: z.enum(['low', 'medium', 'high']).optional(),
    layer: z.enum(['memory', 'loki', 'redis', 'postgres', 'all']).optional()
  }).optional()
});

const CacheDeleteSchema = z.object({
  key: z.string().min(1).max(255)
});

const BatchOperationSchema = z.object({
  operations: z.array(z.object({
    operation: z.enum(['get', 'set', 'delete']),
    key: z.string().min(1).max(255),
    value: z.any().optional(),
    options: z.object({
      ttl: z.number().optional(),
      tags: z.array(z.string()).optional(),
      priority: z.enum(['low', 'medium', 'high']).optional()
    }).optional()
  })).max(100) // Limit batch size
});

// ============================================================================
// GET - Retrieve cached value
// ============================================================================

export const GET: RequestHandler = async ({ url }) => {
  try {
    const key = url.searchParams.get('key');
    const action = url.searchParams.get('action');

    // Handle different GET actions
    switch (action) {
      case 'stats':
        const stats = await getCacheStats();
        return json({
          success: true,
          stats
        });

      case 'health':
        const health = await getCacheHealth();
        return json({
          success: true,
          health
        });

      case 'get':
        if (!key) {
          return json({ 
            success: false, 
            error: 'Key parameter is required' 
          }, { status: 400 });
        }

        const value = await cachingService.get(key);
        return json({
          success: true,
          key,
          value,
          cached: value !== null
        });

      default:
        return json({
          success: false,
          error: 'Invalid action. Use: get, stats, or health'
        }, { status: 400 });
    }
  } catch (error) {
    console.error('Cache GET error:', error);
    return json({
      success: false,
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// ============================================================================
// POST - Set cached value or batch operations
// ============================================================================

export const POST: RequestHandler = async ({ request }) => {
  try {
    const rawData = await request.json();
    
    // Check if this is a batch operation
    if (rawData.operations) {
      return await handleBatchOperation(rawData);
    }

    // Validate single set operation
    const validationResult = CacheSetSchema.safeParse(rawData);
    if (!validationResult.success) {
      return json({
        success: false,
        error: 'Invalid request data',
        details: validationResult.error.flatten()
      }, { status: 400 });
    }

    const { key, value, options = {} } = validationResult.data;

    // Set the cached value
    const success = await cachingService.set(key, value, options);
    
    return json({
      success,
      key,
      message: success ? 'Value cached successfully' : 'Failed to cache value',
      options
    });

  } catch (error) {
    console.error('Cache POST error:', error);
    return json({
      success: false,
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// ============================================================================
// PUT - Update existing cached value
// ============================================================================

export const PUT: RequestHandler = async ({ request }) => {
  try {
    const rawData = await request.json();
    const validationResult = CacheSetSchema.safeParse(rawData);
    
    if (!validationResult.success) {
      return json({
        success: false,
        error: 'Invalid request data',
        details: validationResult.error.flatten()
      }, { status: 400 });
    }

    const { key, value, options = {} } = validationResult.data;

    // Check if key exists
    const existing = await cachingService.get(key);
    if (existing === null) {
      return json({
        success: false,
        error: 'Key not found in cache'
      }, { status: 404 });
    }

    // Update the cached value
    const success = await cachingService.set(key, value, options);
    
    return json({
      success,
      key,
      message: success ? 'Value updated successfully' : 'Failed to update value',
      previousValue: existing,
      newValue: value
    });

  } catch (error) {
    console.error('Cache PUT error:', error);
    return json({
      success: false,
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// ============================================================================
// DELETE - Remove cached value or clear cache
// ============================================================================

export const DELETE: RequestHandler = async ({ request, url }) => {
  try {
    const key = url.searchParams.get('key');
    const action = url.searchParams.get('action');
    const tag = url.searchParams.get('tag');

    if (action === 'clear') {
      // Clear entire cache
      const success = await cachingService.clear();
      return json({
        success,
        message: success ? 'Cache cleared successfully' : 'Failed to clear cache'
      });
    }

    if (action === 'invalidate-tag' && tag) {
      // Invalidate by tag
      const count = await cachingService.invalidateByTag(tag);
      return json({
        success: count > 0,
        message: `Invalidated ${count} cache entries with tag '${tag}'`,
        invalidatedCount: count
      });
    }

    if (!key) {
      return json({
        success: false,
        error: 'Key parameter is required'
      }, { status: 400 });
    }

    // Delete single key
    const success = await cachingService.delete(key);
    
    return json({
      success,
      key,
      message: success ? 'Key deleted successfully' : 'Key not found or failed to delete'
    });

  } catch (error) {
    console.error('Cache DELETE error:', error);
    return json({
      success: false,
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// ============================================================================
// BATCH OPERATIONS HANDLER
// ============================================================================

async function handleBatchOperation(data: any) {
  const validationResult = BatchOperationSchema.safeParse(data);
  if (!validationResult.success) {
    return json({
      success: false,
      error: 'Invalid batch operation data',
      details: validationResult.error.flatten()
    }, { status: 400 });
  }

  const { operations } = validationResult.data;
  const results = [];

  for (const op of operations) {
    try {
      let result;
      
      switch (op.operation) {
        case 'get':
          const value = await cachingService.get(op.key, op.options);
          result = {
            operation: 'get',
            key: op.key,
            success: true,
            value,
            cached: value !== null
          };
          break;

        case 'set':
          if (op.value === undefined) {
            result = {
              operation: 'set',
              key: op.key,
              success: false,
              error: 'Value is required for set operation'
            };
          } else {
            const success = await cachingService.set(op.key, op.value, op.options);
            result = {
              operation: 'set',
              key: op.key,
              success,
              error: success ? undefined : 'Failed to set value'
            };
          }
          break;

        case 'delete':
          const success = await cachingService.delete(op.key);
          result = {
            operation: 'delete',
            key: op.key,
            success,
            error: success ? undefined : 'Failed to delete key'
          };
          break;

        default:
          result = {
            operation: op.operation,
            key: op.key,
            success: false,
            error: 'Invalid operation'
          };
      }

      results.push(result);
    } catch (error) {
      results.push({
        operation: op.operation,
        key: op.key,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  const successCount = results.filter((r: any) => r.success).length;
  
  return json({
    success: true,
    message: `Batch operation completed: ${successCount}/${operations.length} successful`,
    results,
    summary: {
      total: operations.length,
      successful: successCount,
      failed: operations.length - successCount
    }
  });
}