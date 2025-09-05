#!/usr/bin/env zx

/**
 * Cluster Multicore Manager - Advanced Worker Coordination
 * Manages GPU contexts, SIMD processing, and resource allocation
 * Integrates with full-stack-workflow.ts and agent orchestration
 */

import 'zx/globals'
import cluster from 'cluster'
import os from 'os'
import { EventEmitter } from 'events'
import { performance } from 'perf_hooks'
import chalk from 'chalk'
import fs from 'fs'
import net from 'net'

export class ClusterMulticoreManager extends EventEmitter {
  /**
   * Reserve ports for a map of services and ranges
   * serviceMap: { serviceName: {start, end} }
   */
  async reservePorts(serviceMap) {
    const allocations = {};
    for (const [name, range] of Object.entries(serviceMap)) {
      allocations[name] = await this.findFreePortInRange(range.start, range.end);
      this.portAssignments[name] = allocations[name];
    }
    this.writePortAssignments();
    return allocations;
  }

  /**
   * Find a free port in a custom range
   */
  async findFreePortInRange(start, end) {
    for (let port = start; port <= end; port++) {
      if (this.allocatedPorts.has(port)) continue;
      const isFree = await this.isPortFree(port);
      if (isFree) {
        this.allocatedPorts.add(port);
        return port;
      }
    }
    throw new Error(`No free ports available in range ${start}-${end}`);
  }

  /**
   * Get assigned port for a service
   */
  getPort(serviceName) {
    return this.portAssignments[serviceName];
  }
  constructor(options = {}) {
    super()

    this.config = {
      maxWorkers: options.maxWorkers || os.cpus().length,
      gpuContextsPerWorker: options.gpuContextsPerWorker || 2,
      memoryLimitMB: options.memoryLimitMB || 512,
      enableWebGPU: options.enableWebGPU ?? true,
      enableSIMD: options.enableSIMD ?? true,
      taskTimeout: options.taskTimeout || 30000,
      healthCheckInterval: options.healthCheckInterval || 5000,
  portRange: options.portRange || { start: 9100, end: 9200 },
  portAssignmentsFile: options.portAssignmentsFile || 'cluster-status.json',
      ...options
    }

    this.workers = new Map()
    this.taskQueue = []
    this.activeTasks = new Map()
    this.metrics = {
      tasksCompleted: 0,
      tasksFaileds: 0,
      totalProcessingTime: 0,
      avgWorkerCPU: 0,
      avgWorkerMemory: 0,
      gpuUtilization: 0
    }

    this.resourcePools = {
      gpu: Array.from({ length: this.config.maxWorkers * this.config.gpuContextsPerWorker },
        (_, i) => ({ id: i, available: true, workerId: Math.floor(i / this.config.gpuContextsPerWorker) })),
      memory: new Map(),
      cpu: new Map()
    }

  // Port allocator state
  this.portAssignments = {}
  this.allocatedPorts = new Set()

    this.isInitialized = false
  }

  /**
   * Initialize the cluster manager
   */
  async initialize() {
    if (this.isInitialized) return

    console.log(chalk.cyan('🚀 Initializing Cluster Multicore Manager'))
    console.log(chalk.blue(`📊 Configuration:`))
    console.log(chalk.blue(`   Workers: ${this.config.maxWorkers}`))
  console.log(chalk.blue(`   Port Range: ${this.config.portRange.start}-${this.config.portRange.end}`))
    console.log(chalk.blue(`   GPU Contexts per Worker: ${this.config.gpuContextsPerWorker}`))
    console.log(chalk.blue(`   Memory Limit: ${this.config.memoryLimitMB}MB per worker`))
    console.log(chalk.blue(`   WebGPU: ${this.config.enableWebGPU ? 'Enabled' : 'Disabled'}`))
    console.log(chalk.blue(`   SIMD: ${this.config.enableSIMD ? 'Enabled' : 'Disabled'}`))

    if (cluster.isPrimary) {
      await this.initializePrimary()
    } else {
      await this.initializeWorker()
    }

    this.isInitialized = true
    this.emit('initialized', { config: this.config })
  }

  /**
   * Initialize primary process
   */
  async initializePrimary() {
    console.log(chalk.green('🎯 Primary process initializing worker coordination'))

    // Fork workers
    for (let i = 0; i < this.config.maxWorkers; i++) {
      await this.createWorker(i)
    }

  // Initialize port assignments
  await this.initializePortAllocator()

    // Set up health monitoring
    this.startHealthMonitoring()

    // Handle worker events
    cluster.on('exit', this.handleWorkerExit.bind(this))
    cluster.on('message', this.handleWorkerMessage.bind(this))

    console.log(chalk.green(`✅ ${this.workers.size} workers initialized and ready`))
  }

  /**
   * Create and configure a worker
   */
  async createWorker(workerId) {
    const worker = cluster.fork({
      WORKER_ID: workerId,
      GPU_CONTEXTS: this.config.gpuContextsPerWorker,
      MEMORY_LIMIT_MB: this.config.memoryLimitMB,
      ENABLE_WEBGPU: this.config.enableWebGPU,
      ENABLE_SIMD: this.config.enableSIMD
    })

    const workerInfo = {
      id: workerId,
      worker,
      pid: worker.process.pid,
      status: 'initializing',
      activeTasks: 0,
      totalTasks: 0,
      cpu: 0,
      memory: 0,
      gpu: this.resourcePools.gpu.filter(gpu => gpu.workerId === workerId),
      startTime: Date.now()
    }

    this.workers.set(workerId, workerInfo)

    // Setup worker message handling
    worker.on('message', (message) => {
      this.handleWorkerMessage(workerId, message)
    })

    worker.on('online', () => {
      workerInfo.status = 'ready'
      console.log(chalk.green(`👷 Worker ${workerId} (PID: ${worker.process.pid}) is online`))
    })

    worker.on('exit', (code, signal) => {
      this.handleWorkerExit(worker, code, signal)
    })

    return workerInfo
  }

  /**
   * Central Port Allocator
   */
  async initializePortAllocator() {
    // Predefined services to assign ports for
    const services = [
      'cluster-manager',
      'load-balancer', 
      'frontend',
      'enhanced-rag',
      'quic-gateway'
    ];
    for (const service of services) {
      this.portAssignments[service] = await this.findFreePort();
    }
    this.writePortAssignments();
    console.log(chalk.green('🛡️ Port assignments:'), this.portAssignments);
  }

  async findFreePort() {
    for (let port = this.config.portRange.start; port <= this.config.portRange.end; port++) {
      if (this.allocatedPorts.has(port)) continue;
      const isFree = await this.isPortFree(port);
      if (isFree) {
        this.allocatedPorts.add(port);
        return port;
      }
    }
    throw new Error('No free ports available in range');
  }

  async isPortFree(port) {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.once('error', () => resolve(false));
      server.once('listening', () => {
        server.close(() => resolve(true));
      });
      server.listen(port, '0.0.0.0');
    });
  }

  writePortAssignments() {
    fs.writeFileSync(this.config.portAssignmentsFile, JSON.stringify(this.portAssignments, null, 2));
  }

  getPortAssignments() {
    return this.portAssignments;
  }

  /**
   * Execute tasks concurrently across the cluster
   */
  async executeTasksConcurrently(tasks) {
    if (!this.isInitialized) {
      throw new Error('Cluster manager not initialized')
    }

    console.log(chalk.cyan(`🔄 Executing ${tasks.length} tasks concurrently`))

    const taskPromises = tasks.map(async (task) => {
      return this.executeTask(task)
    })

    const startTime = performance.now()
    const results = await Promise.allSettled(taskPromises)
    const totalTime = performance.now() - startTime

    // Update metrics
    this.metrics.totalProcessingTime += totalTime
    results.forEach(result => {
      if (result.status === 'fulfilled') {
        this.metrics.tasksCompleted++
      } else {
        this.metrics.tasksFaileds++
      }
    })

    const summary = {
      totalTasks: tasks.length,
      successful: results.filter(r => r.status === 'fulfilled').length,
      failed: results.filter(r => r.status === 'rejected').length,
      totalTime,
      averageTime: totalTime / tasks.length
    }

    console.log(chalk.green(`✅ Concurrent execution completed:`))
    console.log(chalk.blue(`   Successful: ${summary.successful}/${summary.totalTasks}`))
    console.log(chalk.blue(`   Total Time: ${summary.totalTime.toFixed(2)}ms`))
    console.log(chalk.blue(`   Average Time: ${summary.averageTime.toFixed(2)}ms`))

    return { results, summary }
  }

  /**
   * Execute a single task
   */
  async executeTask(task) {
    const taskId = this.generateTaskId()
    const startTime = performance.now()

    console.log(chalk.yellow(`🎯 Executing task: ${task.type} (ID: ${taskId})`))

    try {
      // Select optimal worker for task
      const worker = await this.selectOptimalWorker(task)

      if (!worker) {
        throw new Error('No available workers for task execution')
      }

      // Allocate resources
      const resources = await this.allocateResources(task, worker.id)

      // Execute task
      const result = await this.sendTaskToWorker(worker, {
        ...task,
        taskId,
        resources,
        timestamp: Date.now()
      })

      // Release resources
      await this.releaseResources(resources)

      const endTime = performance.now()
      const processingTime = endTime - startTime

      console.log(chalk.green(`✨ Task ${taskId} completed in ${processingTime.toFixed(2)}ms`))

      return {
        taskId,
        type: task.type,
        result,
        processingTime,
        workerId: worker.id,
        success: true
      }

    } catch (error) {
      const endTime = performance.now()
      const processingTime = endTime - startTime

      console.error(chalk.red(`💥 Task ${taskId} failed after ${processingTime.toFixed(2)}ms:`), error)

      return {
        taskId,
        type: task.type,
        error: error.message,
        processingTime,
        success: false
      }
    }
  }

  /**
   * Select optimal worker based on task requirements and current load
   */
  async selectOptimalWorker(task) {
    const availableWorkers = Array.from(this.workers.values())
      .filter(w => w.status === 'ready')
      .sort((a, b) => {
        // Priority scoring algorithm
        let scoreA = 0, scoreB = 0

        // Lower active tasks = higher priority
        scoreA += (10 - a.activeTasks) * 0.4
        scoreB += (10 - b.activeTasks) * 0.4

        // Lower CPU usage = higher priority
        scoreA += (100 - a.cpu) * 0.3
        scoreB += (100 - b.cpu) * 0.3

        // GPU availability for GPU tasks
        if (task.requiresGPU) {
          const gpuAvailableA = a.gpu.filter(g => g.available).length
          const gpuAvailableB = b.gpu.filter(g => g.available).length
          scoreA += gpuAvailableA * 0.3
          scoreB += gpuAvailableB * 0.3
        }

        return scoreB - scoreA
      })

    return availableWorkers[0] || null
  }

  /**
   * Allocate resources for task execution
   */
  async allocateResources(task, workerId) {
    const allocated = {
      worker: workerId,
      gpu: [],
      memory: 0
    }

    // Allocate GPU contexts if needed
    if (task.requiresGPU) {
      const requiredGPU = task.gpuContexts || 1
      const availableGPU = this.resourcePools.gpu
        .filter(gpu => gpu.workerId === workerId && gpu.available)
        .slice(0, requiredGPU)

      if (availableGPU.length < requiredGPU) {
        throw new Error(`Insufficient GPU contexts available (need ${requiredGPU}, have ${availableGPU.length})`)
      }

      availableGPU.forEach(gpu => {
        gpu.available = false
        allocated.gpu.push(gpu.id)
      })
    }

    // Allocate memory
    if (task.memoryMB) {
      const currentMemory = this.resourcePools.memory.get(workerId) || 0
      if (currentMemory + task.memoryMB > this.config.memoryLimitMB) {
        throw new Error(`Insufficient memory available (need ${task.memoryMB}MB, available ${this.config.memoryLimitMB - currentMemory}MB)`)
      }

      this.resourcePools.memory.set(workerId, currentMemory + task.memoryMB)
      allocated.memory = task.memoryMB
    }

    return allocated
  }

  /**
   * Release allocated resources
   */
  async releaseResources(resources) {
    // Release GPU contexts
    resources.gpu.forEach(gpuId => {
      const gpu = this.resourcePools.gpu.find(g => g.id === gpuId)
      if (gpu) gpu.available = true
    })

    // Release memory
    if (resources.memory > 0) {
      const currentMemory = this.resourcePools.memory.get(resources.worker) || 0
      this.resourcePools.memory.set(resources.worker, Math.max(0, currentMemory - resources.memory))
    }
  }

  /**
   * Send task to worker and wait for result
   */
  async sendTaskToWorker(workerInfo, task) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Task ${task.taskId} timed out`))
      }, this.config.taskTimeout)

      const messageHandler = (message) => {
        if (message.taskId === task.taskId) {
          clearTimeout(timeout)
          workerInfo.worker.removeListener('message', messageHandler)

          if (message.type === 'task-result') {
            resolve(message.result)
          } else if (message.type === 'task-error') {
            reject(new Error(message.error))
          }
        }
      }

      workerInfo.worker.on('message', messageHandler)
      workerInfo.worker.send(task)

      // Update worker stats
      workerInfo.activeTasks++
      workerInfo.totalTasks++
    })
  }

  /**
   * Handle worker messages
   */
  handleWorkerMessage(workerId, message) {
    const worker = this.workers.get(workerId)
    if (!worker) return

    switch (message.type) {
      case 'health-update':
        worker.cpu = message.cpu
        worker.memory = message.memory
        this.updateClusterMetrics()
        break

      case 'task-completed':
        worker.activeTasks = Math.max(0, worker.activeTasks - 1)
        break

      case 'task-failed':
        worker.activeTasks = Math.max(0, worker.activeTasks - 1)
        break

      case 'gpu-status':
        this.updateGPUStatus(workerId, message.gpuStatus)
        break

      default:
        // Handle other message types
        this.emit('worker-message', { workerId, message })
    }
  }

  /**
   * Handle worker exit
   */
  handleWorkerExit(worker, code, signal) {
    console.log(chalk.red(`⚠️ Worker PID ${worker.process.pid} exited with code ${code} and signal ${signal}`))

    // Find and remove worker
    for (const [workerId, workerInfo] of this.workers.entries()) {
      if (workerInfo.worker === worker) {
        this.workers.delete(workerId)

        // Release resources
        this.releaseWorkerResources(workerId)

        // Respawn if needed
        if (code !== 0 && !worker.exitedAfterDisconnect) {
          console.log(chalk.yellow(`🔄 Respawning worker ${workerId}`))
          setTimeout(() => this.createWorker(workerId), 1000)
        }
        break
      }
    }
  }

  /**
   * Release all resources for a worker
   */
  releaseWorkerResources(workerId) {
    // Release GPU contexts
    this.resourcePools.gpu
      .filter(gpu => gpu.workerId === workerId)
      .forEach(gpu => gpu.available = true)

    // Release memory
    this.resourcePools.memory.delete(workerId)
  }

  /**
   * Start health monitoring
   */
  startHealthMonitoring() {
    setInterval(() => {
      this.performHealthCheck()
    }, this.config.healthCheckInterval)
  }

  /**
   * Perform cluster health check
   */
  async performHealthCheck() {
    const healthPromises = Array.from(this.workers.values()).map(worker => {
      return new Promise((resolve) => {
        const timeout = setTimeout(() => resolve(null), 2000)

        const messageHandler = (message) => {
          if (message.type === 'health-response') {
            clearTimeout(timeout)
            worker.worker.removeListener('message', messageHandler)
            resolve(message.health)
          }
        }

        worker.worker.on('message', messageHandler)
        worker.worker.send({ type: 'health-check' })
      })
    })

    const healthResults = await Promise.allSettled(healthPromises)
    this.updateClusterHealth(healthResults)
  }

  /**
   * Update cluster metrics
   */
  updateClusterMetrics() {
    const workers = Array.from(this.workers.values())

    this.metrics.avgWorkerCPU = workers.reduce((sum, w) => sum + w.cpu, 0) / workers.length
    this.metrics.avgWorkerMemory = workers.reduce((sum, w) => sum + w.memory, 0) / workers.length

    const totalGPU = this.resourcePools.gpu.length
    const availableGPU = this.resourcePools.gpu.filter(g => g.available).length
    this.metrics.gpuUtilization = ((totalGPU - availableGPU) / totalGPU) * 100

    this.emit('metrics-updated', this.metrics)
  }

  /**
   * Get cluster status
   */
  getClusterStatus() {
    return {
      isInitialized: this.isInitialized,
      totalWorkers: this.workers.size,
      readyWorkers: Array.from(this.workers.values()).filter(w => w.status === 'ready').length,
      activeTasks: Array.from(this.workers.values()).reduce((sum, w) => sum + w.activeTasks, 0),
      metrics: this.metrics,
      resourcePools: {
        gpu: {
          total: this.resourcePools.gpu.length,
          available: this.resourcePools.gpu.filter(g => g.available).length
        },
        memory: {
          totalMB: this.config.memoryLimitMB * this.workers.size,
          usedMB: Array.from(this.resourcePools.memory.values()).reduce((sum, m) => sum + m, 0)
        }
      },
      portAssignments: this.portAssignments
    };
  }

  /**
   * Initialize worker process
   */
  async initializeWorker() {
    const workerId = parseInt(process.env.WORKER_ID)
    const gpuContexts = parseInt(process.env.GPU_CONTEXTS)
    const memoryLimit = parseInt(process.env.MEMORY_LIMIT_MB)

    console.log(chalk.blue(`👷 Worker ${workerId} initializing with ${gpuContexts} GPU contexts, ${memoryLimit}MB memory`))

    // Initialize worker resources (simulated)
    await this.initializeWorkerResources(workerId, gpuContexts, memoryLimit)

    // Handle task messages
    process.on('message', async (message) => {
      await this.handleWorkerTask(message)
    })

    // Send ready signal
    process.send({ type: 'worker-ready', workerId })

    // Start health reporting
    this.startWorkerHealthReporting()
  }

  /**
   * Initialize worker-specific resources
   */
  async initializeWorkerResources(workerId, gpuContexts, memoryLimit) {
    // Simulate resource initialization
    if (this.config.enableWebGPU) {
      console.log(chalk.magenta(`🚀 Worker ${workerId}: WebGPU contexts initialized`))
    }

    if (this.config.enableSIMD) {
      console.log(chalk.cyan(`💾 Worker ${workerId}: SIMD buffers allocated (${memoryLimit}MB)`))
    }

    console.log(chalk.green(`🎮 Worker ${workerId}: ${gpuContexts} GPU contexts ready`))
  }

  /**
   * Handle task in worker process
   */
  async handleWorkerTask(message) {
    if (message.type === 'health-check') {
      const health = {
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        status: 'healthy',
        uptime: process.uptime()
      }
      process.send({ type: 'health-response', health })
      return
    }

    // Handle actual task
    const { taskId, type, resources } = message

    try {
      const result = await this.processWorkerTask(message)
      process.send({ type: 'task-result', taskId, result })
      process.send({ type: 'task-completed', taskId })

    } catch (error) {
      process.send({ type: 'task-error', taskId, error: error.message })
      process.send({ type: 'task-failed', taskId })
    }
  }

  /**
   * Process specific task types in worker
   */
  async processWorkerTask(task) {
    const processingTime = Math.random() * 1000 + 500
    await new Promise(resolve => setTimeout(resolve, processingTime))

    switch (task.type) {
      case 'gpu-acceleration':
        return {
          processed: true,
          gpuContextsUsed: task.resources.gpu.length,
          processingTime,
          shadersCompiled: Math.floor(Math.random() * 5) + 1
        }

      case 'simd-processing':
        return {
          processed: true,
          documentsProcessed: task.batchSize || 100,
          processingTime,
          memoryUsed: task.resources.memory
        }

      case 'vector-embedding':
        return {
          processed: true,
          embeddingsGenerated: Math.floor(Math.random() * 500) + 100,
          processingTime,
          dimensions: task.dimensions || 384
        }

      default:
        return {
          processed: true,
          processingTime,
          type: task.type
        }
    }
  }

  /**
   * Start worker health reporting
   */
  startWorkerHealthReporting() {
    setInterval(() => {
      const health = {
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        pid: process.pid
      }
      process.send({ type: 'health-update', ...health })
    }, 3000)
  }

  /**
   * Utility functions
   */
  generateTaskId() {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  async shutdown() {
    console.log(chalk.yellow('🛑 Shutting down cluster manager...'))

    if (cluster.isPrimary) {
      // Kill all workers
      for (const [workerId, workerInfo] of this.workers) {
        workerInfo.worker.kill('SIGTERM')
      }
    }

    this.emit('shutdown')
    process.exit(0)
  }
}

// Export for use (already exported above in class definition)

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const manager = new ClusterMulticoreManager({
    maxWorkers: 4,
    gpuContextsPerWorker: 2,
    enableWebGPU: true,
    enableSIMD: true
  })

  await manager.initialize()

  // Example concurrent task execution
  const tasks = [
    { type: 'gpu-acceleration', requiresGPU: true, gpuContexts: 1 },
    { type: 'simd-processing', batchSize: 1024, memoryMB: 100 },
    { type: 'vector-embedding', dimensions: 384, memoryMB: 50 },
    { type: 'gpu-acceleration', requiresGPU: true, gpuContexts: 2 }
  ]

  const { results, summary } = await manager.executeTasksConcurrently(tasks)

  console.log(chalk.green('\n📊 Final Results:'))
  console.log(summary)

  setTimeout(() => manager.shutdown(), 2000)
}