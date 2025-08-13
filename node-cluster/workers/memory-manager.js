const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { performance } = require('perf_hooks');

/**
 * Memory Management Service Worker
 * Handles OOM prevention, garbage collection coordination, and memory optimization
 */
class MemoryManagerWorker {
  constructor(data) {
    this.workerId = data.workerId;
    this.services = data.services;
    this.startTime = Date.now();
    
    // Memory management configuration
    this.config = {
      memoryThresholds: {
        warning: 0.75,      // 75% memory usage warning
        critical: 0.85,     // 85% memory usage critical
        emergency: 0.95     // 95% memory usage emergency
      },
      heapThresholds: {
        warning: 1.5 * 1024 * 1024 * 1024,  // 1.5GB heap warning
        critical: 2 * 1024 * 1024 * 1024,   // 2GB heap critical
        emergency: 2.5 * 1024 * 1024 * 1024 // 2.5GB heap emergency
      },
      gcIntervals: {
        normal: 30000,      // 30 seconds normal GC
        stressed: 10000,    // 10 seconds under stress
        emergency: 5000     // 5 seconds emergency GC
      },
      checkInterval: 5000,  // Memory check every 5 seconds
      historySize: 100,     // Keep 100 memory readings
      webgpuMemoryLimit: 4 * 1024 * 1024 * 1024, // 4GB WebGPU limit
      tensorCacheLimit: 512 * 1024 * 1024        // 512MB tensor cache
    };
    
    // Memory tracking
    this.memoryHistory = [];
    this.gcHistory = [];
    this.lastGC = Date.now();
    this.gcCount = 0;
    this.oomPreventions = 0;
    
    // Performance monitoring
    this.performanceMetrics = {
      avgGCTime: 0,
      totalGCTime: 0,
      memoryLeakDetected: false,
      lastLeakCheck: Date.now(),
      tensorAllocations: new Map(),
      webgpuBuffers: new Map()
    };
    
    this.init();
  }
  
  async init() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Memory manager worker starting`);
    
    // Setup message handling
    this.setupMessageHandling();
    
    // Start memory monitoring
    this.startMemoryMonitoring();
    
    // Start WebGPU memory tracking
    this.startWebGPUMonitoring();
    
    // Setup emergency OOM prevention
    this.setupOOMPrevention();
    
    this.sendMessage({
      type: 'worker-ready',
      worker: 'memory-manager',
      pid: process.pid
    });
  }
  
  setupMessageHandling() {
    parentPort.on('message', async (message) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        console.error(`[MEMORY-MANAGER-${this.workerId}] Message handling error:`, error);
        this.sendMessage({
          type: 'error',
          worker: 'memory-manager',
          error: error.message,
          timestamp: Date.now()
        });
      }
    });
  }
  
  async handleMessage(message) {
    switch (message.type) {
      case 'force-gc':
        await this.forceGarbageCollection();
        break;
        
      case 'memory-report':
        this.sendMemoryReport();
        break;
        
      case 'optimize-memory':
        await this.optimizeMemory();
        break;
        
      case 'register-tensor':
        this.registerTensorAllocation(message.data);
        break;
        
      case 'release-tensor':
        this.releaseTensorAllocation(message.data.id);
        break;
        
      case 'webgpu-allocate':
        await this.handleWebGPUAllocation(message.data);
        break;
        
      case 'webgpu-deallocate':
        await this.handleWebGPUDeallocation(message.data);
        break;
        
      case 'memory-pressure':
        await this.handleMemoryPressure(message.data.level);
        break;
        
      case 'leak-detection':
        await this.performLeakDetection();
        break;
        
      case 'health-check':
        this.sendHealthReport();
        break;
        
      default:
        console.log(`[MEMORY-MANAGER-${this.workerId}] Unknown message type: ${message.type}`);
    }
  }
  
  startMemoryMonitoring() {
    setInterval(() => {
      this.checkMemoryUsage();
    }, this.config.checkInterval);
    
    // Also check on each event loop iteration for critical situations
    setImmediate(() => this.continuousMonitoring());
  }
  
  continuousMonitoring() {
    const memUsage = process.memoryUsage();
    const heapUsage = memUsage.heapUsed / memUsage.heapTotal;
    
    // Emergency GC if heap is critically full
    if (heapUsage > this.config.memoryThresholds.emergency) {
      this.emergencyGarbageCollection();
    }
    
    // Schedule next check
    setImmediate(() => this.continuousMonitoring());
  }
  
  checkMemoryUsage() {
    const memUsage = process.memoryUsage();
    const timestamp = Date.now();
    
    // Calculate memory percentages
    const heapUsagePercent = memUsage.heapUsed / memUsage.heapTotal;
    const rssUsagePercent = memUsage.rss / this.getSystemMemory();
    
    // Create memory snapshot
    const snapshot = {
      timestamp: timestamp,
      heapUsed: memUsage.heapUsed,
      heapTotal: memUsage.heapTotal,
      rss: memUsage.rss,
      external: memUsage.external,
      arrayBuffers: memUsage.arrayBuffers,
      heapUsagePercent: heapUsagePercent,
      rssUsagePercent: rssUsagePercent,
      gcInfo: this.getLastGCInfo()
    };
    
    // Add to history
    this.memoryHistory.push(snapshot);
    if (this.memoryHistory.length > this.config.historySize) {
      this.memoryHistory.shift();
    }
    
    // Check thresholds and trigger actions
    this.evaluateMemoryThresholds(snapshot);
    
    // Log critical memory situations
    if (heapUsagePercent > this.config.memoryThresholds.warning) {
      console.warn(`[MEMORY-MANAGER-${this.workerId}] High memory usage: ${(heapUsagePercent * 100).toFixed(1)}%`);
    }
  }
  
  evaluateMemoryThresholds(snapshot) {
    const heapPercent = snapshot.heapUsagePercent;
    const rssPercent = snapshot.rssUsagePercent;
    
    if (heapPercent > this.config.memoryThresholds.emergency || 
        rssPercent > this.config.memoryThresholds.emergency) {
      this.handleEmergencyMemory(snapshot);
    } else if (heapPercent > this.config.memoryThresholds.critical || 
               rssPercent > this.config.memoryThresholds.critical) {
      this.handleCriticalMemory(snapshot);
    } else if (heapPercent > this.config.memoryThresholds.warning || 
               rssPercent > this.config.memoryThresholds.warning) {
      this.handleWarningMemory(snapshot);
    }
  }
  
  async handleEmergencyMemory(snapshot) {
    console.error(`[MEMORY-MANAGER-${this.workerId}] EMERGENCY: Memory usage critical!`, {
      heapPercent: (snapshot.heapUsagePercent * 100).toFixed(1) + '%',
      rssPercent: (snapshot.rssUsagePercent * 100).toFixed(1) + '%'
    });
    
    // Immediate emergency actions
    await this.emergencyGarbageCollection();
    await this.releaseNonCriticalMemory();
    await this.clearWebGPUBuffers();
    
    // Notify all workers and main process
    this.sendMessage({
      type: 'memory-emergency',
      data: {
        snapshot: snapshot,
        actions: ['emergency-gc', 'clear-caches', 'release-buffers'],
        timestamp: Date.now()
      }
    });
    
    this.oomPreventions++;
  }
  
  async handleCriticalMemory(snapshot) {
    console.warn(`[MEMORY-MANAGER-${this.workerId}] CRITICAL: High memory usage`, {
      heapPercent: (snapshot.heapUsagePercent * 100).toFixed(1) + '%'
    });
    
    // Aggressive cleanup
    await this.forceGarbageCollection();
    await this.clearTensorCache();
    await this.optimizeWebGPUMemory();
    
    this.sendMessage({
      type: 'memory-critical',
      data: {
        snapshot: snapshot,
        actions: ['force-gc', 'clear-tensor-cache'],
        timestamp: Date.now()
      }
    });
  }
  
  async handleWarningMemory(snapshot) {
    console.log(`[MEMORY-MANAGER-${this.workerId}] WARNING: Elevated memory usage`, {
      heapPercent: (snapshot.heapUsagePercent * 100).toFixed(1) + '%'
    });
    
    // Gentle cleanup
    if (Date.now() - this.lastGC > this.config.gcIntervals.stressed) {
      await this.forceGarbageCollection();
    }
    
    await this.cleanupOldTensors();
  }
  
  async forceGarbageCollection() {
    if (!global.gc) {
      console.warn(`[MEMORY-MANAGER-${this.workerId}] Garbage collection not available`);
      return;
    }
    
    const startTime = performance.now();
    const beforeMemory = process.memoryUsage();
    
    // Run garbage collection
    global.gc();
    
    const endTime = performance.now();
    const afterMemory = process.memoryUsage();
    const gcTime = endTime - startTime;
    
    // Update GC metrics
    this.gcCount++;
    this.lastGC = Date.now();
    this.performanceMetrics.totalGCTime += gcTime;
    this.performanceMetrics.avgGCTime = this.performanceMetrics.totalGCTime / this.gcCount;
    
    // Log GC results
    const memoryFreed = beforeMemory.heapUsed - afterMemory.heapUsed;
    console.log(`[MEMORY-MANAGER-${this.workerId}] GC completed in ${gcTime.toFixed(2)}ms, freed ${this.formatBytes(memoryFreed)}`);
    
    // Add to GC history
    this.gcHistory.push({
      timestamp: Date.now(),
      duration: gcTime,
      memoryBefore: beforeMemory.heapUsed,
      memoryAfter: afterMemory.heapUsed,
      memoryFreed: memoryFreed
    });
    
    if (this.gcHistory.length > 50) {
      this.gcHistory.shift();
    }
    
    this.sendMessage({
      type: 'gc-completed',
      data: {
        duration: gcTime,
        memoryFreed: memoryFreed,
        gcCount: this.gcCount,
        timestamp: Date.now()
      }
    });
  }
  
  async emergencyGarbageCollection() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Emergency garbage collection initiated`);
    
    // Multiple GC passes for emergency cleanup
    for (let i = 0; i < 3; i++) {
      await this.forceGarbageCollection();
      // Small delay between passes
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  startWebGPUMonitoring() {
    // Monitor WebGPU memory usage if available
    setInterval(() => {
      this.checkWebGPUMemory();
    }, 10000); // Check every 10 seconds
  }
  
  checkWebGPUMemory() {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      // WebGPU memory tracking (simplified for this example)
      const totalAllocated = Array.from(this.performanceMetrics.webgpuBuffers.values())
        .reduce((sum, buffer) => sum + buffer.size, 0);
      
      if (totalAllocated > this.config.webgpuMemoryLimit * 0.8) {
        console.warn(`[MEMORY-MANAGER-${this.workerId}] WebGPU memory usage high: ${this.formatBytes(totalAllocated)}`);
        this.optimizeWebGPUMemory();
      }
    }
  }
  
  async handleWebGPUAllocation(data) {
    const { id, size, type, priority = 'normal' } = data;
    
    // Check if allocation would exceed limits
    const currentUsage = Array.from(this.performanceMetrics.webgpuBuffers.values())
      .reduce((sum, buffer) => sum + buffer.size, 0);
    
    if (currentUsage + size > this.config.webgpuMemoryLimit) {
      // Try to free some memory first
      await this.optimizeWebGPUMemory();
      
      // Check again
      const newUsage = Array.from(this.performanceMetrics.webgpuBuffers.values())
        .reduce((sum, buffer) => sum + buffer.size, 0);
      
      if (newUsage + size > this.config.webgpuMemoryLimit) {
        this.sendMessage({
          type: 'webgpu-allocation-failed',
          data: {
            id: id,
            requestedSize: size,
            currentUsage: newUsage,
            limit: this.config.webgpuMemoryLimit,
            reason: 'Memory limit exceeded'
          }
        });
        return;
      }
    }
    
    // Track the allocation
    this.performanceMetrics.webgpuBuffers.set(id, {
      id: id,
      size: size,
      type: type,
      priority: priority,
      allocatedAt: Date.now()
    });
    
    console.log(`[MEMORY-MANAGER-${this.workerId}] WebGPU allocation: ${id} (${this.formatBytes(size)})`);
    
    this.sendMessage({
      type: 'webgpu-allocated',
      data: { id: id, size: size }
    });
  }
  
  async handleWebGPUDeallocation(data) {
    const { id } = data;
    
    const buffer = this.performanceMetrics.webgpuBuffers.get(id);
    if (buffer) {
      this.performanceMetrics.webgpuBuffers.delete(id);
      console.log(`[MEMORY-MANAGER-${this.workerId}] WebGPU deallocation: ${id} (${this.formatBytes(buffer.size)})`);
      
      this.sendMessage({
        type: 'webgpu-deallocated',
        data: { id: id, size: buffer.size }
      });
    }
  }
  
  async optimizeWebGPUMemory() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Optimizing WebGPU memory`);
    
    const buffers = Array.from(this.performanceMetrics.webgpuBuffers.entries());
    const now = Date.now();
    
    // Sort by priority and age
    buffers.sort((a, b) => {
      const [idA, bufferA] = a;
      const [idB, bufferB] = b;
      
      // Lower priority first, then older buffers
      if (bufferA.priority !== bufferB.priority) {
        const priorityOrder = { 'low': 0, 'normal': 1, 'high': 2, 'critical': 3 };
        return priorityOrder[bufferA.priority] - priorityOrder[bufferB.priority];
      }
      
      return bufferA.allocatedAt - bufferB.allocatedAt;
    });
    
    // Free low priority and old buffers
    let freedMemory = 0;
    const toFree = [];
    
    for (const [id, buffer] of buffers) {
      if (buffer.priority === 'low' || (now - buffer.allocatedAt > 300000)) { // 5 minutes
        toFree.push(id);
        freedMemory += buffer.size;
        
        // Stop if we've freed enough
        if (freedMemory > this.config.webgpuMemoryLimit * 0.2) { // Free 20%
          break;
        }
      }
    }
    
    // Actually free the buffers
    for (const id of toFree) {
      this.performanceMetrics.webgpuBuffers.delete(id);
      
      this.sendMessage({
        type: 'webgpu-force-deallocate',
        data: { id: id, reason: 'memory-optimization' }
      });
    }
    
    if (toFree.length > 0) {
      console.log(`[MEMORY-MANAGER-${this.workerId}] WebGPU optimization freed ${this.formatBytes(freedMemory)} from ${toFree.length} buffers`);
    }
  }
  
  async clearWebGPUBuffers() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Emergency WebGPU buffer clearing`);
    
    const bufferIds = Array.from(this.performanceMetrics.webgpuBuffers.keys());
    let freedMemory = 0;
    
    for (const id of bufferIds) {
      const buffer = this.performanceMetrics.webgpuBuffers.get(id);
      if (buffer && buffer.priority !== 'critical') {
        freedMemory += buffer.size;
        this.performanceMetrics.webgpuBuffers.delete(id);
        
        this.sendMessage({
          type: 'webgpu-force-deallocate',
          data: { id: id, reason: 'emergency-cleanup' }
        });
      }
    }
    
    console.log(`[MEMORY-MANAGER-${this.workerId}] Emergency cleared ${this.formatBytes(freedMemory)} WebGPU memory`);
  }
  
  registerTensorAllocation(data) {
    const { id, size, type, shape } = data;
    
    this.performanceMetrics.tensorAllocations.set(id, {
      id: id,
      size: size,
      type: type,
      shape: shape,
      allocatedAt: Date.now(),
      lastAccessed: Date.now()
    });
    
    // Check tensor cache limit
    const totalTensorMemory = Array.from(this.performanceMetrics.tensorAllocations.values())
      .reduce((sum, tensor) => sum + tensor.size, 0);
    
    if (totalTensorMemory > this.config.tensorCacheLimit) {
      this.cleanupOldTensors();
    }
  }
  
  releaseTensorAllocation(id) {
    const tensor = this.performanceMetrics.tensorAllocations.get(id);
    if (tensor) {
      this.performanceMetrics.tensorAllocations.delete(id);
      console.log(`[MEMORY-MANAGER-${this.workerId}] Released tensor: ${id} (${this.formatBytes(tensor.size)})`);
    }
  }
  
  async cleanupOldTensors() {
    const now = Date.now();
    const maxAge = 600000; // 10 minutes
    const toDelete = [];
    
    this.performanceMetrics.tensorAllocations.forEach((tensor, id) => {
      if (now - tensor.lastAccessed > maxAge) {
        toDelete.push(id);
      }
    });
    
    for (const id of toDelete) {
      this.releaseTensorAllocation(id);
      
      this.sendMessage({
        type: 'tensor-cleanup',
        data: { id: id, reason: 'age-limit' }
      });
    }
    
    if (toDelete.length > 0) {
      console.log(`[MEMORY-MANAGER-${this.workerId}] Cleaned up ${toDelete.length} old tensors`);
    }
  }
  
  async clearTensorCache() {
    const tensorIds = Array.from(this.performanceMetrics.tensorAllocations.keys());
    
    for (const id of tensorIds) {
      this.releaseTensorAllocation(id);
      
      this.sendMessage({
        type: 'tensor-cleanup',
        data: { id: id, reason: 'cache-clear' }
      });
    }
    
    console.log(`[MEMORY-MANAGER-${this.workerId}] Cleared tensor cache: ${tensorIds.length} tensors`);
  }
  
  async releaseNonCriticalMemory() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Releasing non-critical memory`);
    
    // Clear various caches and temporary data
    await this.clearTensorCache();
    await this.cleanupOldTensors();
    
    // Trim memory history to minimal size
    if (this.memoryHistory.length > 20) {
      this.memoryHistory = this.memoryHistory.slice(-20);
    }
    
    if (this.gcHistory.length > 10) {
      this.gcHistory = this.gcHistory.slice(-10);
    }
    
    this.sendMessage({
      type: 'memory-released',
      data: {
        actions: ['tensor-cache-cleared', 'history-trimmed'],
        timestamp: Date.now()
      }
    });
  }
  
  async performLeakDetection() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Performing memory leak detection`);
    
    if (this.memoryHistory.length < 10) {
      return; // Not enough data
    }
    
    // Check for consistent memory growth
    const recentHistory = this.memoryHistory.slice(-10);
    const trend = this.calculateMemoryTrend(recentHistory);
    
    // Check for steady increase without corresponding decreases
    if (trend.slope > 0 && trend.correlation > 0.8) {
      const projectedMemory = recentHistory[recentHistory.length - 1].heapUsed + (trend.slope * 60000); // 1 minute projection
      
      if (projectedMemory > this.config.heapThresholds.critical) {
        this.performanceMetrics.memoryLeakDetected = true;
        
        console.warn(`[MEMORY-MANAGER-${this.workerId}] Potential memory leak detected!`, {
          trend: trend,
          projectedMemory: this.formatBytes(projectedMemory)
        });
        
        this.sendMessage({
          type: 'memory-leak-detected',
          data: {
            trend: trend,
            projectedMemory: projectedMemory,
            recommendation: 'Review recent allocations and ensure proper cleanup',
            timestamp: Date.now()
          }
        });
      }
    }
    
    this.performanceMetrics.lastLeakCheck = Date.now();
  }
  
  calculateMemoryTrend(history) {
    const n = history.length;
    const x = history.map((_, i) => i);
    const y = history.map(h => h.heapUsed);
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate correlation coefficient
    const meanX = sumX / n;
    const meanY = sumY / n;
    const numerator = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
    const denomX = Math.sqrt(x.reduce((sum, xi) => sum + (xi - meanX) ** 2, 0));
    const denomY = Math.sqrt(y.reduce((sum, yi) => sum + (yi - meanY) ** 2, 0));
    const correlation = numerator / (denomX * denomY);
    
    return { slope, intercept, correlation };
  }
  
  setupOOMPrevention() {
    // Set up emergency handlers
    process.on('warning', (warning) => {
      if (warning.name === 'MaxListenersExceededWarning' || 
          warning.message.includes('memory')) {
        console.warn(`[MEMORY-MANAGER-${this.workerId}] Memory warning: ${warning.message}`);
        this.handleMemoryPressure('high');
      }
    });
    
    // Monitor for uncaught exceptions that might indicate memory issues
    process.on('uncaughtException', (error) => {
      if (error.message.includes('out of memory') || error.code === 'ERR_OUT_OF_MEMORY') {
        console.error(`[MEMORY-MANAGER-${this.workerId}] OOM Exception caught:`, error.message);
        this.emergencyGarbageCollection();
      }
    });
  }
  
  async handleMemoryPressure(level) {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Handling memory pressure: ${level}`);
    
    switch (level) {
      case 'low':
        if (Date.now() - this.lastGC > this.config.gcIntervals.normal) {
          await this.forceGarbageCollection();
        }
        break;
        
      case 'medium':
        await this.forceGarbageCollection();
        await this.cleanupOldTensors();
        break;
        
      case 'high':
        await this.forceGarbageCollection();
        await this.clearTensorCache();
        await this.optimizeWebGPUMemory();
        break;
        
      case 'critical':
        await this.emergencyGarbageCollection();
        await this.releaseNonCriticalMemory();
        await this.clearWebGPUBuffers();
        break;
    }
    
    this.sendMessage({
      type: 'memory-pressure-handled',
      data: { level: level, timestamp: Date.now() }
    });
  }
  
  async optimizeMemory() {
    console.log(`[MEMORY-MANAGER-${this.workerId}] Starting memory optimization`);
    
    const beforeMemory = process.memoryUsage();
    
    // Comprehensive optimization
    await this.forceGarbageCollection();
    await this.cleanupOldTensors();
    await this.optimizeWebGPUMemory();
    await this.performLeakDetection();
    
    const afterMemory = process.memoryUsage();
    const memoryFreed = beforeMemory.heapUsed - afterMemory.heapUsed;
    
    console.log(`[MEMORY-MANAGER-${this.workerId}] Memory optimization completed, freed ${this.formatBytes(memoryFreed)}`);
    
    this.sendMessage({
      type: 'memory-optimized',
      data: {
        memoryBefore: beforeMemory,
        memoryAfter: afterMemory,
        memoryFreed: memoryFreed,
        timestamp: Date.now()
      }
    });
  }
  
  getSystemMemory() {
    // Simplified system memory detection
    const os = require('os');
    return os.totalmem();
  }
  
  getLastGCInfo() {
    if (this.gcHistory.length > 0) {
      return this.gcHistory[this.gcHistory.length - 1];
    }
    return null;
  }
  
  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  sendMemoryReport() {
    const currentMemory = process.memoryUsage();
    const systemMemory = this.getSystemMemory();
    
    const report = {
      current: currentMemory,
      system: {
        total: systemMemory,
        free: require('os').freemem(),
        usage: (systemMemory - require('os').freemem()) / systemMemory
      },
      history: this.memoryHistory.slice(-10), // Last 10 readings
      gc: {
        count: this.gcCount,
        lastGC: this.lastGC,
        avgTime: this.performanceMetrics.avgGCTime,
        totalTime: this.performanceMetrics.totalGCTime,
        history: this.gcHistory.slice(-5) // Last 5 GC runs
      },
      tensors: {
        count: this.performanceMetrics.tensorAllocations.size,
        totalSize: Array.from(this.performanceMetrics.tensorAllocations.values())
          .reduce((sum, tensor) => sum + tensor.size, 0)
      },
      webgpu: {
        bufferCount: this.performanceMetrics.webgpuBuffers.size,
        totalSize: Array.from(this.performanceMetrics.webgpuBuffers.values())
          .reduce((sum, buffer) => sum + buffer.size, 0),
        limit: this.config.webgpuMemoryLimit
      },
      performance: {
        oomPreventions: this.oomPreventions,
        memoryLeakDetected: this.performanceMetrics.memoryLeakDetected,
        lastLeakCheck: this.performanceMetrics.lastLeakCheck
      },
      timestamp: Date.now()
    };
    
    this.sendMessage({
      type: 'memory-report',
      data: report
    });
  }
  
  sendHealthReport() {
    const health = {
      worker: 'memory-manager',
      workerId: this.workerId,
      pid: process.pid,
      uptime: Date.now() - this.startTime,
      memoryUsage: process.memoryUsage(),
      gcStats: {
        count: this.gcCount,
        avgTime: this.performanceMetrics.avgGCTime,
        lastGC: this.lastGC
      },
      oomPreventions: this.oomPreventions,
      monitoringActive: true,
      timestamp: Date.now()
    };
    
    this.sendMessage({
      type: 'health-report',
      data: health
    });
  }
  
  sendMessage(message) {
    try {
      parentPort.postMessage(message);
    } catch (error) {
      console.error(`[MEMORY-MANAGER-${this.workerId}] Failed to send message:`, error);
    }
  }
}

// Initialize worker if running in worker thread
if (!isMainThread) {
  new MemoryManagerWorker(workerData);
}

module.exports = MemoryManagerWorker;