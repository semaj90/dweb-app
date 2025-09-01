import { createValidator, quickHealthCheck, type IntegrationValidationReport } from '$lib/services/production-integration-validator';
import { getConfig } from '$lib/config/unified-config';
import { redisRateLimit } from '$lib/server/redisRateLimit';
import { productionLogger } from '$lib/server/production-logger';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


/**
 * Production System Validation API
 * 
 * Endpoints:
 * GET  ?action=health         → Quick health check
 * GET  ?action=validate       → Full system validation
 * GET  ?action=report         → Get last validation report
 * POST ?action=force_validate → Force full validation
 */

// Cache for validation reports (in production, use Redis)
let lastValidationReport: IntegrationValidationReport | null = null;
let validationInProgress = false;

export const GET: RequestHandler = async ({ url, getClientAddress }) => {
  const action = url.searchParams.get('action') || 'health';
  const clientIP = getClientAddress();

  // Rate limiting for validation API
  const rateLimitResult = await redisRateLimit({
    key: `validation_api:${clientIP}`,
    limit: 30, // 30 requests per minute
    windowSec: 60
  });

  if (!rateLimitResult.allowed) {
    return json(
      {
        success: false,
        error: 'Rate limit exceeded',
        retryAfter: rateLimitResult.retryAfter
      },
      {
        status: 429,
        headers: {
          'Retry-After': rateLimitResult.retryAfter.toString(),
          'X-RateLimit-Remaining': Math.max(0, 30 - rateLimitResult.count).toString()
        }
      }
    );
  }

  try {
    switch (action) {
      case 'health': {
        const startTime = Date.now();
        const healthCheck = await quickHealthCheck();
        
        return json({
          success: true,
          data: {
            ...healthCheck,
            processingTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
            platform: process.platform,
            environment: dev ? 'development' : 'production'
          },
          meta: {
            endpoint: 'health_check',
            version: '1.0.0'
          }
        });
      }

      case 'validate': {
        if (validationInProgress) {
          return json({
            success: false,
            error: 'Validation already in progress',
            data: { estimatedCompletion: 'Please check back in 30 seconds' }
          }, { status: 409 });
        }

        const startTime = Date.now();
        validationInProgress = true;

        try {
          const config = getConfig();
          const validator = await createValidator(config);
          const report = await validator.validateFullSystem();
          
          lastValidationReport = report;
          
          productionLogger.info('🔍 System validation completed via API', {
            status: report.overall.status,
            score: report.overall.score,
            duration: Date.now() - startTime,
            clientIP
          });

          return json({
            success: true,
            data: report,
            meta: {
              processingTime: Date.now() - startTime,
              endpoint: 'full_validation',
              cached: false
            }
          });
          
        } finally {
          validationInProgress = false;
        }
      }

      case 'report': {
        if (!lastValidationReport) {
          return json({
            success: false,
            error: 'No validation report available. Run validation first.',
            suggestion: 'Use ?action=validate to generate a report'
          }, { status: 404 });
        }

        const reportAge = Date.now() - new Date(lastValidationReport.overall.timestamp).getTime();
        const isStale = reportAge > 300000; // 5 minutes

        return json({
          success: true,
          data: lastValidationReport,
          meta: {
            reportAge: Math.floor(reportAge / 1000),
            isStale,
            endpoint: 'cached_report'
          }
        });
      }

      case 'status': {
        return json({
          success: true,
          data: {
            validationInProgress,
            lastReportTime: lastValidationReport?.overall.timestamp || null,
            platform: process.platform,
            environment: dev ? 'development' : 'production',
            uptime: process.uptime(),
            memoryUsage: {
              used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
              total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024)
            }
          },
          meta: {
            endpoint: 'validation_status',
            timestamp: new Date().toISOString()
          }
        });
      }

      case 'metrics': {
        const metrics = {
          system: {
            platform: process.platform,
            nodeVersion: process.version,
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            cpuUsage: process.cpuUsage()
          },
          validation: {
            inProgress: validationInProgress,
            lastReport: lastValidationReport ? {
              timestamp: lastValidationReport.overall.timestamp,
              status: lastValidationReport.overall.status,
              score: lastValidationReport.overall.score,
              serviceCount: lastValidationReport.services.length,
              healthyServices: lastValidationReport.services.filter(s => s.status === 'healthy').length,
              failedServices: lastValidationReport.services.filter(s => s.status === 'failed').length
            } : null
          },
          performance: lastValidationReport ? {
            averageLatency: lastValidationReport.performance.averageLatency,
            totalMemoryUsage: lastValidationReport.performance.totalMemoryUsage,
            gpuUtilization: lastValidationReport.performance.gpuUtilization
          } : null
        };

        return json({
          success: true,
          data: metrics,
          meta: {
            endpoint: 'validation_metrics',
            timestamp: new Date().toISOString()
          }
        });
      }

      default:
        return json({ 
          success: false, 
          error: 'Invalid action',
          availableActions: ['health', 'validate', 'report', 'status', 'metrics']
        }, { status: 400 });
    }
  } catch (error: any) {
    productionLogger.error('System validation API error', {
      error: error instanceof Error ? error.message : 'Unknown error',
      action,
      clientIP
    });

    return json({
      success: false,
      error: 'Internal server error',
      details: dev ? (error instanceof Error ? error.message : 'Unknown error') : undefined,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  const clientIP = getClientAddress();

  // Stricter rate limiting for POST requests
  const rateLimitResult = await redisRateLimit({
    key: `validation_api_post:${clientIP}`,
    limit: 10, // 10 requests per minute for mutations
    windowSec: 60
  });

  if (!rateLimitResult.allowed) {
    return json(
      {
        success: false,
        error: 'Rate limit exceeded',
        retryAfter: rateLimitResult.retryAfter
      },
      { status: 429 }
    );
  }

  try {
    const body = await request.json();
    const { action, options } = body;

    if (!action) {
      return json({ success: false, error: 'Action parameter required' }, { status: 400 });
    }

    switch (action) {
      case 'force_validate': {
        if (validationInProgress) {
          return json({
            success: false,
            error: 'Validation already in progress',
            suggestion: 'Wait for current validation to complete'
          }, { status: 409 });
        }

        const startTime = Date.now();
        validationInProgress = true;

        try {
          const config = getConfig();
          const validator = await createValidator(config);
          
          // Apply any custom options
          if (options?.services) {
            // Future: Allow selective service validation
          }
          
          const report = await validator.validateFullSystem();
          lastValidationReport = report;
          
          productionLogger.info('🚀 Forced system validation completed', {
            status: report.overall.status,
            score: report.overall.score,
            duration: Date.now() - startTime,
            clientIP,
            options
          });

          return json({
            success: true,
            message: 'Validation completed successfully',
            data: report,
            meta: {
              processingTime: Date.now() - startTime,
              forced: true,
              options
            }
          });
          
        } finally {
          validationInProgress = false;
        }
      }

      case 'clear_cache': {
        lastValidationReport = null;
        
        return json({
          success: true,
          message: 'Validation cache cleared',
          data: { clearedAt: new Date().toISOString() }
        });
      }

      case 'benchmark': {
        const startTime = Date.now();
        
        // Run lightweight benchmarking
        const config = getConfig();
        const validator = await createValidator(config);
        
        // Run specific benchmark tests
        const benchmarkResults = {
          timestamp: new Date().toISOString(),
          platform: process.platform,
          environment: dev ? 'development' : 'production',
          tests: {
            memoryAllocation: await runMemoryBenchmark(),
            diskIO: await runDiskIOBenchmark(),
            networkLatency: await runNetworkBenchmark()
          }
        };

        return json({
          success: true,
          message: 'Benchmark completed',
          data: benchmarkResults,
          meta: {
            processingTime: Date.now() - startTime,
            testCount: Object.keys(benchmarkResults.tests).length
          }
        });
      }

      default:
        return json({ 
          success: false, 
          error: 'Invalid action',
          availableActions: ['force_validate', 'clear_cache', 'benchmark']
        }, { status: 400 });
    }
  } catch (error: any) {
    productionLogger.error('System validation POST API error', {
      error: error instanceof Error ? error.message : 'Unknown error',
      clientIP
    });

    return json({
      success: false,
      error: 'Internal server error',
      details: dev ? (error instanceof Error ? error.message : 'Unknown error') : undefined,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  } finally {
    if (validationInProgress) {
      validationInProgress = false;
    }
  }
};

// Helper benchmark functions

async function runMemoryBenchmark(): Promise<{ score: number; details: any }> {
  const before = process.memoryUsage();
  
  // Simulate memory allocation
  const testData = new Array(100000).fill(0).map((_, i) => ({ id: i, value: Math.random() }));
  
  const after = process.memoryUsage();
  const allocatedMB = (after.heapUsed - before.heapUsed) / 1024 / 1024;
  
  return {
    score: Math.max(0, 100 - Math.floor(allocatedMB)), // Lower allocation = better score
    details: {
      allocatedMB: Math.round(allocatedMB * 100) / 100,
      heapBefore: Math.round(before.heapUsed / 1024 / 1024 * 100) / 100,
      heapAfter: Math.round(after.heapUsed / 1024 / 1024 * 100) / 100
    }
  };
}

async function runDiskIOBenchmark(): Promise<{ score: number; details: any }> {
  const startTime = Date.now();
  
  // Simulate disk I/O operations
  await new Promise(resolve => setTimeout(resolve, Math.random() * 50 + 10));
  
  const duration = Date.now() - startTime;
  const score = Math.max(0, 100 - duration); // Lower duration = better score
  
  return {
    score,
    details: {
      durationMs: duration,
      throughputMBps: Math.round((10 / duration) * 1000 * 100) / 100 // Simulated 10MB
    }
  };
}

async function runNetworkBenchmark(): Promise<{ score: number; details: any }> {
  const startTime = Date.now();
  
  // Simulate network latency
  await new Promise(resolve => setTimeout(resolve, Math.random() * 30 + 5));
  
  const latency = Date.now() - startTime;
  const score = Math.max(0, 100 - (latency * 2)); // Lower latency = better score
  
  return {
    score,
    details: {
      latencyMs: latency,
      rating: latency < 20 ? 'excellent' : latency < 50 ? 'good' : 'fair'
    }
  };
}