#!/usr/bin/env node

/**
 * SvelteKit 2 Cluster Entry Point
 * Production-ready Node.js clustering with intelligent load balancing
 */

import { createSvelteKitCluster } from './src/lib/services/nodejs-cluster-architecture.js';
import { readFileSync } from 'fs';
import { join } from 'path';

// Load configuration
const config = loadClusterConfig();

// Create and start cluster manager
const clusterManager = createSvelteKitCluster(config);

// Setup monitoring and logging
setupMonitoring(clusterManager);

console.log('🚀 SvelteKit Cluster Manager started');
console.log(`📊 Configuration: ${JSON.stringify(config, null, 2)}`);

/**
 * Load cluster configuration from environment and config files
 */
function loadClusterConfig() {
  const defaultConfig = {
    workers: parseInt(process.env.CLUSTER_WORKERS) || undefined,
    port: parseInt(process.env.PORT) || 3000,
    host: process.env.HOST || '0.0.0.0',
    gracefulShutdownTimeout: parseInt(process.env.GRACEFUL_SHUTDOWN_TIMEOUT) || 10000,
    healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL) || 5000,
    maxMemoryUsage: parseInt(process.env.MAX_MEMORY_USAGE) || 512,
    restartOnHighMemory: process.env.RESTART_ON_HIGH_MEMORY !== 'false',
    loadBalancingStrategy: process.env.LOAD_BALANCING_STRATEGY || 'round-robin',
    enableStickySession: process.env.ENABLE_STICKY_SESSION === 'true',
    redisUrl: process.env.REDIS_URL
  };

  // Try to load from config file
  try {
    const configPath = process.env.CLUSTER_CONFIG_PATH || './cluster.config.json';
    const fileConfig = JSON.parse(readFileSync(configPath, 'utf8'));
    return { ...defaultConfig, ...fileConfig };
  } catch (error) {
    console.log('📝 Using default configuration (no config file found)');
    return defaultConfig;
  }
}

/**
 * Setup monitoring and health reporting
 */
function setupMonitoring(clusterManager) {
  // Log cluster health every 30 seconds
  setInterval(() => {
    const health = clusterManager.getHealth();
    console.log(`📊 Cluster Health: ${health.healthyWorkers}/${health.totalWorkers} workers, ` +
                `${health.totalRequests} requests, ` +
                `${(health.memoryUsage.average / 1024 / 1024).toFixed(2)}MB avg memory`);
  }, 30000);

  // Handle cluster scaling via signals
  process.on('SIGUSR1', () => {
    console.log('📈 Scaling up cluster...');
    const currentWorkers = clusterManager.getWorkerMetrics().length;
    clusterManager.scaleCluster(currentWorkers + 1);
  });

  process.on('SIGUSR2', () => {
    console.log('📉 Scaling down cluster...');
    const currentWorkers = clusterManager.getWorkerMetrics().length;
    if (currentWorkers > 1) {
      clusterManager.scaleCluster(currentWorkers - 1);
    }
  });

  // Handle graceful restart
  process.on('SIGHUP', async () => {
    console.log('🔄 Rolling restart initiated...');
    await clusterManager.rollingRestart();
  });
}

/**
 * Export cluster manager for programmatic access
 */
export default clusterManager;