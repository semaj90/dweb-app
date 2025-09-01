
import type { RequestHandler } from './$types';

/**
 * Cluster Restart API Endpoint
 * Handles graceful rolling restart of worker processes
 */

import cluster from 'node:cluster';

export const POST: RequestHandler = async ({ request }) => {
  try {
    // Verify we're in primary process
    if (!cluster.isPrimary) {
      return json({
        error: 'Cluster restart only available from primary process'
      }, { status: 403 });
    }

    // Get cluster manager instance
    const clusterManager = globalThis.clusterManager;
    
    if (!clusterManager) {
      return json({
        error: 'Cluster manager not available'
      }, { status: 503 });
    }

    // Check if already restarting
    if (globalThis.clusterRestarting) {
      return json({
        error: 'Rolling restart already in progress',
        message: 'Please wait for the current restart to complete'
      }, { status: 409 });
    }

    // Parse optional parameters
    const body = await request.json().catch(() => ({}));
    const { force = false, timeout = 30000 } = body;

    // Get pre-restart state
    const preRestartWorkers = clusterManager.getWorkerMetrics();
    const preRestartHealth = clusterManager.getHealth();

    console.log('🔄 Initiating rolling restart of cluster...');
    console.log(`📊 Pre-restart: ${preRestartWorkers.length} workers, ${preRestartHealth.healthyWorkers} healthy`);

    // Set restart flag
    globalThis.clusterRestarting = true;

    // Start rolling restart in background
    const restartPromise = performRollingRestart(clusterManager, {
      force,
      timeout
    });

    // Don't await the full restart - return immediately
    restartPromise
      .then(() => {
        console.log('✅ Rolling restart completed successfully');
        globalThis.clusterRestarting = false;
      })
      .catch((error) => {
        console.error('❌ Rolling restart failed:', error);
        globalThis.clusterRestarting = false;
      });

    // Log restart action for audit
    const auditLog = {
      timestamp: new Date().toISOString(),
      action: 'cluster_rolling_restart',
      preRestartWorkers: preRestartWorkers.length,
      healthyWorkers: preRestartHealth.healthyWorkers,
      force,
      timeout,
      initiator: 'admin_api'
    };
    
    console.log('📝 Restart audit log:', auditLog);

    return json({
      success: true,
      message: 'Rolling restart initiated',
      workersToRestart: preRestartWorkers.length,
      estimatedDuration: `${Math.ceil(preRestartWorkers.length * 3)}s`,
      force,
      timeout,
      timestamp: Date.now()
    });

  } catch (error: any) {
    console.error('Cluster restart error:', error);
    globalThis.clusterRestarting = false;
    
    return json({
      error: 'Failed to initiate rolling restart',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async () => {
  try {
    // Get restart status
    const clusterManager = globalThis.clusterManager;
    
    if (!clusterManager) {
      return json({
        error: 'Cluster manager not available'
      }, { status: 503 });
    }

    const isRestarting = globalThis.clusterRestarting || false;
    const workers = clusterManager.getWorkerMetrics();
    const health = clusterManager.getHealth();

    return json({
      isRestarting,
      currentWorkers: workers.length,
      healthyWorkers: health.healthyWorkers,
      lastRestart: globalThis.lastRestartTime || null,
      restartHistory: globalThis.restartHistory || [],
      canRestart: !isRestarting && health.healthyWorkers > 0
    });

  } catch (error: any) {
    console.error('Cluster restart status error:', error);
    
    return json({
      error: 'Failed to get restart status',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

/**
 * Perform rolling restart of all workers
 */
async function performRollingRestart(
  clusterManager: any, 
  options: { force: boolean; timeout: number }
): Promise<void> {
  const startTime = Date.now();
  const workers = clusterManager.getWorkerMetrics();
  
  console.log(`🔄 Starting rolling restart of ${workers.length} workers`);

  for (let i = 0; i < workers.length; i++) {
    const worker = workers[i];
    const clusterWorker = cluster.workers![worker.workerId];
    
    if (!clusterWorker || clusterWorker.isDead()) {
      console.log(`⏭️ Skipping dead worker ${worker.workerId}`);
      continue;
    }

    console.log(`🔄 Restarting worker ${worker.workerId} (${i + 1}/${workers.length})`);

    try {
      // Gracefully disconnect worker
      clusterWorker.disconnect();
      
      // Wait for graceful shutdown or force kill after timeout
      await Promise.race([
        waitForWorkerExit(clusterWorker),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), options.timeout)
        )
      ]);

    } catch (error: any) {
      if (options.force) {
        console.log(`💀 Force killing worker ${worker.workerId}`);
        clusterWorker.kill();
      } else {
        throw new Error(`Failed to restart worker ${worker.workerId}: ${error}`);
      }
    }

    // Wait for replacement worker to be ready
    await waitForHealthyWorkers(clusterManager, workers.length);
    
    // Brief pause between restarts
    await new Promise((resolve: any) => setTimeout(resolve, 1000));
  }

  const duration = Date.now() - startTime;
  console.log(`✅ Rolling restart completed in ${duration}ms`);

  // Record restart in history
  const restartRecord = {
    timestamp: new Date().toISOString(),
    duration,
    workersRestarted: workers.length,
    success: true
  };

  globalThis.lastRestartTime = Date.now();
  globalThis.restartHistory = globalThis.restartHistory || [];
  globalThis.restartHistory.unshift(restartRecord);
  
  // Keep only last 10 restart records
  if (globalThis.restartHistory.length > 10) {
    globalThis.restartHistory = globalThis.restartHistory.slice(0, 10);
  }
}

/**
 * Wait for a worker to exit
 */
function waitForWorkerExit(worker: Worker): Promise<void> {
  return new Promise((resolve) => {
    const checkExit = () => {
      if (worker.isDead()) {
        resolve();
      } else {
        setTimeout(checkExit, 100);
      }
    };
    checkExit();
  });
}

/**
 * Wait for cluster to have the expected number of healthy workers
 */
function waitForHealthyWorkers(
  clusterManager: any, 
  expectedCount: number, 
  maxWait = 10000
): Promise<void> {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    const checkHealth = () => {
      const health = clusterManager.getHealth();
      
      if (health.healthyWorkers >= expectedCount) {
        resolve();
      } else if (Date.now() - startTime > maxWait) {
        reject(new Error(`Timeout waiting for ${expectedCount} healthy workers`));
      } else {
        setTimeout(checkHealth, 500);
      }
    };
    
    checkHealth();
  });
}