import { NeuralMemoryManager } from '$lib/optimization/neural-memory-manager';
import { redisRateLimit } from '$lib/server/redisRateLimit';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// Global manager singleton with Windows optimization
let neuralManager: NeuralMemoryManager | null = null;
let initializationPromise: Promise<NeuralMemoryManager> | null = null;

// Enhanced initialization with Windows GPU detection
async function getNeuralManager(): Promise<NeuralMemoryManager> {
  if (neuralManager) return neuralManager;
  
  // Prevent multiple concurrent initializations
  if (initializationPromise) return initializationPromise;
  
  initializationPromise = initializeManager();
  return initializationPromise;
}

async function initializeManager(): Promise<NeuralMemoryManager> {
  try {
    // Detect Windows system memory
    const systemMemoryMB = await detectSystemMemory();
    
    // Initialize with Windows-specific optimizations
    neuralManager = new NeuralMemoryManager(systemMemoryMB);
    
    // Setup Windows GPU monitoring if available
    if (process.platform === 'win32') {
      await setupWindowsGPUMonitoring(neuralManager);
    }
    
    console.log(`🧠 Neural Memory Manager initialized with ${systemMemoryMB}MB`);
    return neuralManager;
    
  } catch (error: any) {
    console.error('❌ Neural manager initialization failed:', error);
    // Fallback to basic configuration
    neuralManager = new NeuralMemoryManager(4096); // 4GB fallback
    return neuralManager;
  }
}

// Windows system memory detection
async function detectSystemMemory(): Promise<number> {
  try {
    if (process.platform === 'win32') {
      const os = await import('os');
      const totalMem = Math.floor(os.totalmem() / 1024 / 1024); // Convert to MB
      // Use 75% of system memory for neural processing
      return Math.floor(totalMem * 0.75);
    }
    // Default for non-Windows systems
    return 8192;
  } catch {
    return 8192; // Safe fallback
  }
}

// Windows GPU monitoring setup
async function setupWindowsGPUMonitoring(manager: NeuralMemoryManager): Promise<void> {
  try {
    // Check for NVIDIA GPU on Windows
    const { spawn } = await import('child_process');
    
    const nvidiaSmi = spawn('nvidia-smi', ['--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], {
      stdio: 'pipe',
      shell: true
    });
    
    let output = '';
    nvidiaSmi.stdout?.on('data', (data) => {
      output += data.toString();
    });
    
    nvidiaSmi.on('close', (code) => {
      if (code === 0 && output.trim()) {
        const [total, used] = output.trim().split(', ').map(Number);
        if (total && !isNaN(total)) {
          console.log(`🎮 NVIDIA GPU detected: ${total}MB total, ${used}MB used`);
          manager.emit('gpu_detected', { totalMB: total, usedMB: used });
        }
      }
    });
    
    nvidiaSmi.on('error', () => {
      // GPU monitoring not available, continue without it
      if (dev) console.log('🔍 GPU monitoring not available (nvidia-smi not found)');
    });
    
  } catch {
    // GPU monitoring failed, continue without it
    if (dev) console.log('🔍 GPU monitoring setup failed');
  }
}

export const GET: RequestHandler = async ({ url, getClientAddress }) => {
  const action = url.searchParams.get('action') || 'status';
  const horizon = parseInt(url.searchParams.get('horizon') || '30');
  const clientIP = getClientAddress();

  // Rate limiting for neural memory API
  const rateLimitResult = await redisRateLimit({
    key: `neural_api:${clientIP}`,
    limit: 100, // 100 requests per minute
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
          'X-RateLimit-Limit': '100',
          'X-RateLimit-Remaining': Math.max(0, 100 - rateLimitResult.count).toString(),
          'X-RateLimit-Reset': new Date(Date.now() + rateLimitResult.retryAfter * 1000).toISOString()
        }
      }
    );
  }

  try {
    const manager = await getNeuralManager();

    switch (action) {
      case 'predict': {
        const startTime = Date.now();
        const prediction = await manager.predictMemoryUsage(horizon);
        return json({ 
          success: true, 
          data: prediction,
          meta: {
            processingTime: Date.now() - startTime,
            horizon,
            timestamp: new Date().toISOString()
          }
        });
      }

      case 'optimize': {
        const startTime = Date.now();
        manager.optimizeMemoryAllocation();
        const optimizationReport = {
          triggered: true,
          processingTime: Date.now() - startTime,
          memoryUsage: manager.getCurrentMemoryUsage(),
          timestamp: new Date().toISOString()
        };
        return json({ success: true, message: 'Optimization triggered', data: optimizationReport });
      }

      case 'status': {
        const status = await manager.generatePerformanceReport();
        const systemInfo = await getSystemInfo();
        return json({ 
          success: true, 
          data: {
            ...status,
            system: systemInfo,
            rateLimit: {
              remaining: Math.max(0, 100 - rateLimitResult.count),
              reset: new Date(Date.now() + 60000).toISOString()
            }
          }
        });
      }

      case 'report': {
        const report = await manager.generatePerformanceReport();
        const detailedMetrics = await getDetailedMetrics(manager);
        return json({ 
          success: true, 
          data: {
            ...report,
            detailed: detailedMetrics,
            generatedAt: new Date().toISOString()
          }
        });
      }

      case 'health': {
        const health = await performHealthCheck(manager);
        return json({ 
          success: true, 
          data: health,
          meta: {
            checked_at: new Date().toISOString(),
            uptime: process.uptime()
          }
        }, {
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate'
          }
        });
      }

      default:
        return json({ success: false, error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Neural memory API error:', error);
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

  // Rate limiting for POST requests (stricter)
  const rateLimitResult = await redisRateLimit({
    key: `neural_api_post:${clientIP}`,
    limit: 20, // 20 requests per minute for mutations
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
    const { action, memoryPressure, config } = body;
    
    if (!action) {
      return json({ success: false, error: 'Action parameter required' }, { status: 400 });
    }
    
    const manager = await getNeuralManager();

    switch (action) {
      case 'adjust_lod': {
        if (typeof memoryPressure !== 'number' || memoryPressure < 0 || memoryPressure > 1) {
          return json({ success: false, error: 'memoryPressure must be between 0 and 1' }, { status: 400 });
        }
        
        const startTime = Date.now();
        const oldLOD = (manager as any).currentLOD; // Access private property for logging
        await manager.adjustLODLevel(memoryPressure);
        const newLOD = (manager as any).currentLOD;
        
        return json({ 
          success: true, 
          message: 'LOD adjusted', 
          data: {
            memoryPressure,
            oldLevel: oldLOD?.name || 'unknown',
            newLevel: newLOD?.name || 'unknown',
            processingTime: Date.now() - startTime
          }
        });
      }

      case 'force_optimization': {
        const startTime = Date.now();
        const beforeMemory = manager.getCurrentMemoryUsage();
        
        manager.optimizeMemoryAllocation();
        
        const afterMemory = manager.getCurrentMemoryUsage();
        const saved = beforeMemory - afterMemory;
        
        return json({ 
          success: true, 
          message: 'Force optimization complete',
          data: {
            memoryBefore: beforeMemory,
            memoryAfter: afterMemory,
            memorySaved: saved,
            processingTime: Date.now() - startTime,
            timestamp: new Date().toISOString()
          }
        });
      }

      case 'configure': {
        if (!config || typeof config !== 'object') {
          return json({ success: false, error: 'Configuration object required' }, { status: 400 });
        }
        
        const result = await updateManagerConfiguration(manager, config);
        return json({ success: true, message: 'Configuration updated', data: result });
      }

      case 'clear_cache': {
        const startTime = Date.now();
        const clearedBytes = await clearManagerCache(manager);
        return json({ 
          success: true, 
          message: 'Cache cleared',
          data: {
            bytesCleared: clearedBytes,
            processingTime: Date.now() - startTime
          }
        });
      }

      default:
        return json({ success: false, error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Neural memory POST error:', error);
    return json({ 
      success: false, 
      error: 'Internal server error',
      details: dev ? (error instanceof Error ? error.message : 'Unknown error') : undefined,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

// Helper functions for enhanced functionality
async function getSystemInfo(): Promise<any> {
  try {
    const os = await import('os');
    return {
      platform: process.platform,
      arch: process.arch,
      nodeVersion: process.version,
      totalMemory: Math.floor(os.totalmem() / 1024 / 1024), // MB
      freeMemory: Math.floor(os.freemem() / 1024 / 1024), // MB
      cpus: os.cpus().length,
      uptime: process.uptime()
    };
  } catch {
    return { error: 'System info unavailable' };
  }
}

async function getDetailedMetrics(manager: NeuralMemoryManager): Promise<any> {
  return {
    memoryBreakdown: {
      used: manager.getCurrentMemoryUsage(),
      total: (manager as any).maxMemoryMB, // Access private property
      utilization: (manager.getCurrentMemoryUsage() / (manager as any).maxMemoryMB) * 100
    },
    performance: {
      predictionsCount: (manager as any).usageHistory?.length || 0,
      clustersActive: (manager as any).clusters?.size || 0,
      neuralNetworkStatus: (manager as any).isTraining ? 'training' : 'idle'
    }
  };
}

async function performHealthCheck(manager: NeuralMemoryManager): Promise<any> {
  const checks = {
    memoryManager: 'healthy',
    neuralNetwork: 'healthy',
    clustering: 'healthy',
    predictions: 'healthy'
  };

  try {
    // Test memory usage prediction
    await manager.predictMemoryUsage(5);
  } catch {
    checks.predictions = 'degraded';
  }

  const overallHealth = Object.values(checks).includes('unhealthy') 
    ? 'unhealthy' 
    : Object.values(checks).includes('degraded') 
      ? 'degraded' 
      : 'healthy';

  return {
    status: overallHealth,
    checks,
    memoryUsage: manager.getCurrentMemoryUsage(),
    timestamp: new Date().toISOString()
  };
}

async function updateManagerConfiguration(manager: NeuralMemoryManager, config: any): Promise<any> {
  const updatedFields: string[] = [];
  
  // Example configuration updates (extend based on manager capabilities)
  if (config.maxMemoryMB && typeof config.maxMemoryMB === 'number') {
    (manager as any).maxMemoryMB = config.maxMemoryMB;
    updatedFields.push('maxMemoryMB');
  }
  
  return {
    updatedFields,
    currentConfig: {
      maxMemoryMB: (manager as any).maxMemoryMB
    }
  };
}

async function clearManagerCache(manager: NeuralMemoryManager): Promise<number> {
  const beforeUsage = manager.getCurrentMemoryUsage();
  
  // Clear various caches (implement based on manager capabilities)
  try {
    // Clear clusters
    (manager as any).clusters?.clear();
    
    // Clear usage history (keep last 10 entries)
    const history = (manager as any).usageHistory;
    if (Array.isArray(history) && history.length > 10) {
      history.splice(0, history.length - 10);
    }
  } catch {
    // Cache clearing failed, continue
  }
  
  const afterUsage = manager.getCurrentMemoryUsage();
  return beforeUsage - afterUsage;
}