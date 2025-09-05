/**
 * Process Health Monitoring API
 * Exposes comprehensive health metrics for cached Node.js, Go, and Python/CUDA workers
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { legalAIHealthMonitor } from '$lib/services/process-health-monitor';
import { legalAIProcessPool } from '$lib/services/process-pool-manager';
import { legalAIWorkerClient } from '$lib/services/worker-pool-client';
import { legalAIGPUManager } from '$lib/services/gpu-memory-manager';

// GET /api/v1/process-health - Get comprehensive health summary
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  try {
    const format = url.searchParams.get('format') || 'summary';
    const includeMetrics = url.searchParams.get('metrics') === 'true';
    
    switch (format) {
      case 'summary':
        const summary = await legalAIHealthMonitor.getSystemHealthSummary();
        return json({
          success: true,
          timestamp: Date.now(),
          data: summary,
          ...(includeMetrics && { 
            poolStats: legalAIWorkerClient.getPoolStats(),
            gpuStats: await legalAIGPUManager.getGPUStats()
          })
        });
        
      case 'alerts':
        const summary2 = await legalAIHealthMonitor.getSystemHealthSummary();
        return json({
          success: true,
          alerts: summary2.activeAlerts,
          alertSummary: {
            total: summary2.activeAlerts.length,
            critical: summary2.activeAlerts.filter(a => a.severity === 'critical').length,
            high: summary2.activeAlerts.filter(a => a.severity === 'high').length,
            medium: summary2.activeAlerts.filter(a => a.severity === 'medium').length,
            low: summary2.activeAlerts.filter(a => a.severity === 'low').length
          }
        });
        
      case 'gpu':
        const gpuStats = await legalAIGPUManager.getGPUStats();
        const gpuRecommendations = legalAIGPUManager.getMemoryRecommendations();
        
        return json({
          success: true,
          gpu: {
            stats: gpuStats,
            recommendations: gpuRecommendations,
            health: {
              memoryStatus: gpuStats.usedMemoryMB / gpuStats.totalMemoryMB > 0.8 ? 'warning' : 'good',
              temperatureStatus: gpuStats.temperature > 75 ? 'warning' : 'good'
            }
          }
        });
        
      case 'pools':
        const poolStats = legalAIProcessPool.getStats();
        const clientStats = legalAIWorkerClient.getPoolStats();
        
        return json({
          success: true,
          pools: {
            manager: poolStats,
            client: clientStats,
            summary: {
              totalPools: Object.keys(poolStats).length,
              totalWorkers: Object.values(poolStats).reduce((sum: number, pool: any) => sum + pool.totalWorkers, 0),
              totalRequests: Object.values(poolStats).reduce((sum: number, pool: any) => sum + pool.totalRequests, 0)
            }
          }
        });
        
      default:
        return json({
          success: false,
          error: 'Invalid format parameter. Use: summary, alerts, gpu, or pools'
        }, { status: 400 });
    }
    
  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    }, { status: 500 });
  }
};

// POST /api/v1/process-health/alert/:alertId/resolve - Resolve specific alert
export const POST: RequestHandler = async ({ request, url }): Promise<any> => {
  try {
    const action = url.searchParams.get('action');
    
    if (action === 'resolve-alert') {
      const { alertId } = await request.json();
      
      if (!alertId) {
        return json({
          success: false,
          error: 'Alert ID is required'
        }, { status: 400 });
      }
      
      legalAIHealthMonitor.resolveAlert(alertId);
      
      return json({
        success: true,
        message: `Alert ${alertId} resolved`,
        timestamp: Date.now()
      });
    }
    
    if (action === 'optimize-gpu') {
      await legalAIGPUManager.optimizeForLegalWorkload();
      
      return json({
        success: true,
        message: 'GPU optimization triggered',
        timestamp: Date.now()
      });
    }
    
    if (action === 'health-check') {
      // Force immediate health check
      const summary = await legalAIHealthMonitor.getSystemHealthSummary();
      
      return json({
        success: true,
        message: 'Health check completed',
        summary,
        timestamp: Date.now()
      });
    }
    
    return json({
      success: false,
      error: 'Invalid action. Use: resolve-alert, optimize-gpu, or health-check'
    }, { status: 400 });
    
  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    }, { status: 500 });
  }
};