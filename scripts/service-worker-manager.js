#!/usr/bin/env node

/**
 * Service Worker Manager - Background Processing & Event Loop Thread Assignment
 * Handles GPU parsing, indexing, embedding, metadata search, and error-to-vector processing
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import cluster from 'cluster';
import os from 'os';
import fs from 'fs';
import path from 'path';
import EventEmitter from 'events';
import { exec } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ServiceWorkerManager extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            maxWorkers: options.maxWorkers || os.cpus().length,
            workerTypes: options.workerTypes || ['gpu-parser', 'indexer', 'embedder', 'error-processor'],
            gpuEnabled: process.env.CUDA_ENABLED === 'true',
            threadPoolSize: options.threadPoolSize || os.cpus().length * 2,
            ...options
        };
        
        this.workers = new Map();
        this.workerPool = [];
        this.taskQueue = [];
        this.processedTasks = 0;
        this.activeThreads = 0;
        
        // Thread assignment strategy
        this.threadAssignment = {
            'gpu-parser': Math.ceil(this.config.maxWorkers * 0.4),
            'indexer': Math.ceil(this.config.maxWorkers * 0.3),
            'embedder': Math.ceil(this.config.maxWorkers * 0.2),
            'error-processor': Math.ceil(this.config.maxWorkers * 0.1)
        };
        
        console.log('🔧 Service Worker Manager initialized');
        console.log(`   - Max Workers: ${this.config.maxWorkers}`);
        console.log(`   - GPU Enabled: ${this.config.gpuEnabled}`);
        console.log(`   - Thread Pool Size: ${this.config.threadPoolSize}`);
        console.log(`   - Thread Assignment:`, this.threadAssignment);
    }
    
    /**
     * Initialize and start all service workers with optimal thread assignment
     */
    async start() {
        console.log('🚀 Starting Service Worker Manager...');
        
        // Assign threads to event loop for each worker type
        await this.assignThreadsToEventLoop();
        
        // Create worker pools for each type
        await this.createWorkerPools();
        
        // Start background processing
        this.startBackgroundProcessing();
        
        // Setup health monitoring
        this.setupHealthMonitoring();
        
        console.log('✅ All service workers started with optimal thread assignment');
        
        return this;
    }
    
    /**
     * Assign threads to event loop for optimal performance
     */
    async assignThreadsToEventLoop() {
        console.log('🔄 Assigning threads to event loop...');
        
        // Configure UV thread pool size for better async I/O
        process.env.UV_THREADPOOL_SIZE = this.config.threadPoolSize.toString();
        
        // Set V8 flags for optimal performance
        const v8Flags = [
            '--max-old-space-size=4096',
            '--optimize-for-size',
            '--expose-gc'
        ];
        
        if (this.config.gpuEnabled) {
            v8Flags.push('--experimental-worker');
        }
        
        // Apply thread affinity if possible
        if (process.platform !== 'win32') {
            try {
                const cpuCount = os.cpus().length;
                
                // Distribute threads across CPU cores
                for (let i = 0; i < cpuCount; i++) {
                    const affinity = Math.floor(i / 2); // 2 threads per core
                    exec(`taskset -cp ${affinity} ${process.pid}`, (err) => {
                        if (err) console.warn('Thread affinity setting failed:', err.message);
                    });
                }
            } catch (err) {
                console.warn('Thread affinity not supported:', err.message);
            }
        }
        
        console.log(`✅ Event loop configured with ${this.config.threadPoolSize} threads`);
    }
    
    /**
     * Create specialized worker pools for different tasks
     */
    async createWorkerPools() {
        console.log('👥 Creating specialized worker pools...');
        
        for (const [workerType, threadCount] of Object.entries(this.threadAssignment)) {
            const workerPool = [];
            
            for (let i = 0; i < threadCount; i++) {
                const worker = await this.createWorker(workerType, i);
                workerPool.push(worker);
                
                // Setup worker communication
                worker.on('message', (message) => {
                    this.handleWorkerMessage(workerType, i, message);
                });
                
                worker.on('error', (error) => {
                    console.error(`❌ Worker ${workerType}-${i} error:`, error);
                    this.restartWorker(workerType, i);
                });
                
                worker.on('exit', (code) => {
                    if (code !== 0) {
                        console.warn(`⚠️ Worker ${workerType}-${i} exited with code ${code}`);
                        this.restartWorker(workerType, i);
                    }
                });
            }
            
            this.workers.set(workerType, workerPool);
            console.log(`   ✅ ${workerType}: ${threadCount} workers created`);
        }
    }
    
    /**
     * Create a specialized worker for a specific task type
     */
    async createWorker(workerType, workerId) {
        const workerScript = this.getWorkerScript(workerType);
        
        const worker = new Worker(workerScript, {
            workerData: {
                workerId,
                workerType,
                gpuEnabled: this.config.gpuEnabled,
                config: this.config
            }
        });
        
        // Initialize worker
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Worker ${workerType}-${workerId} initialization timeout`));
            }, 10000);
            
            worker.once('message', (message) => {
                if (message.type === 'ready') {
                    clearTimeout(timeout);
                    resolve();
                } else if (message.type === 'error') {
                    clearTimeout(timeout);
                    reject(new Error(message.error));
                }
            });
        });
        
        return worker;
    }
    
    /**
     * Get the appropriate worker script for each worker type
     */
    getWorkerScript(workerType) {
        const workerScripts = {
            'gpu-parser': path.join(__dirname, 'workers', 'gpu-parser-worker.js'),
            'indexer': path.join(__dirname, 'workers', 'indexer-worker.js'),
            'embedder': path.join(__dirname, 'workers', 'embedder-worker.js'),
            'error-processor': path.join(__dirname, 'workers', 'error-processor-worker.js')
        };
        
        return workerScripts[workerType] || this.createGenericWorker(workerType);
    }
    
    /**
     * Create generic worker script if specific one doesn't exist
     */
    createGenericWorker(workerType) {
        const workerScript = `
import { parentPort, workerData } from 'worker_threads';

class ${workerType.charAt(0).toUpperCase() + workerType.slice(1).replace('-', '')}Worker {
    constructor(config) {
        this.config = config;
        this.workerId = config.workerId;
        this.workerType = config.workerType;
    }
    
    async processTask(task) {
        // Generic task processing
        console.log(\`🔧 \${this.workerType}-\${this.workerId} processing task: \${task.type}\`);
        
        try {
            // Simulate work based on task type
            await new Promise(resolve => setTimeout(resolve, Math.random() * 1000));
            
            return {
                success: true,
                result: \`Task \${task.id} completed by \${this.workerType}-\${this.workerId}\`,
                processedAt: new Date().toISOString()
            };
        } catch (error) {
            throw new Error(\`Task processing failed: \${error.message}\`);
        }
    }
}

const worker = new ${workerType.charAt(0).toUpperCase() + workerType.slice(1).replace('-', '')}Worker(workerData);

parentPort.on('message', async (task) => {
    try {
        const result = await worker.processTask(task);
        parentPort.postMessage({
            type: 'result',
            taskId: task.id,
            result
        });
    } catch (error) {
        parentPort.postMessage({
            type: 'error',
            taskId: task.id,
            error: error.message
        });
    }
});

parentPort.postMessage({ type: 'ready' });
`;
        
        const scriptPath = path.join(__dirname, 'workers', `${workerType}-worker.js`);
        
        // Ensure workers directory exists
        const workersDir = path.dirname(scriptPath);
        if (!fs.existsSync(workersDir)) {
            fs.mkdirSync(workersDir, { recursive: true });
        }
        
        fs.writeFileSync(scriptPath, workerScript);
        return scriptPath;
    }
    
    /**
     * Handle messages from workers
     */
    handleWorkerMessage(workerType, workerId, message) {
        switch (message.type) {
            case 'result':
                this.emit('taskCompleted', {
                    workerType,
                    workerId,
                    taskId: message.taskId,
                    result: message.result
                });
                this.processedTasks++;
                break;
                
            case 'error':
                this.emit('taskError', {
                    workerType,
                    workerId,
                    taskId: message.taskId,
                    error: message.error
                });
                break;
                
            case 'metrics':
                this.emit('workerMetrics', {
                    workerType,
                    workerId,
                    metrics: message.metrics
                });
                break;
        }
    }
    
    /**
     * Start background processing loop
     */
    startBackgroundProcessing() {
        console.log('⚡ Starting background processing loop...');
        
        // Process task queue continuously
        setInterval(() => {
            this.processTaskQueue();
        }, 100); // Check every 100ms
        
        // Performance monitoring
        setInterval(() => {
            this.reportPerformanceMetrics();
        }, 5000); // Report every 5 seconds
        
        // Garbage collection trigger
        if (global.gc) {
            setInterval(() => {
                if (this.processedTasks % 1000 === 0) {
                    global.gc();
                    console.log('🗑️ Triggered garbage collection');
                }
            }, 30000); // Every 30 seconds
        }
    }
    
    /**
     * Process queued tasks by distributing to appropriate workers
     */
    processTaskQueue() {
        while (this.taskQueue.length > 0) {
            const task = this.taskQueue.shift();
            const workerType = this.determineWorkerType(task);
            const worker = this.getAvailableWorker(workerType);
            
            if (worker) {
                worker.postMessage(task);
                this.activeThreads++;
            } else {
                // No available workers, put task back in queue
                this.taskQueue.unshift(task);
                break;
            }
        }
    }
    
    /**
     * Determine appropriate worker type for a task
     */
    determineWorkerType(task) {
        if (task.type.includes('parse') || task.type.includes('gpu')) {
            return 'gpu-parser';
        } else if (task.type.includes('index')) {
            return 'indexer';
        } else if (task.type.includes('embed')) {
            return 'embedder';
        } else if (task.type.includes('error')) {
            return 'error-processor';
        }
        
        // Default to gpu-parser for unknown tasks
        return 'gpu-parser';
    }
    
    /**
     * Get an available worker of the specified type
     */
    getAvailableWorker(workerType) {
        const workers = this.workers.get(workerType);
        if (!workers) return null;
        
        // Simple round-robin selection (could be improved with load balancing)
        const worker = workers.find(w => !w.busy);
        if (worker) {
            worker.busy = true;
            
            // Mark as available after some time
            setTimeout(() => {
                worker.busy = false;
                this.activeThreads--;
            }, 1000);
        }
        
        return worker;
    }
    
    /**
     * Add task to processing queue
     */
    addTask(task) {
        task.id = task.id || Date.now() + Math.random();
        task.createdAt = new Date().toISOString();
        
        this.taskQueue.push(task);
        
        this.emit('taskQueued', task);
        
        return task.id;
    }
    
    /**
     * Report performance metrics
     */
    reportPerformanceMetrics() {
        const metrics = {
            timestamp: new Date().toISOString(),
            queueLength: this.taskQueue.length,
            processedTasks: this.processedTasks,
            activeThreads: this.activeThreads,
            maxThreads: this.config.threadPoolSize,
            memoryUsage: process.memoryUsage(),
            cpuUsage: process.cpuUsage()
        };
        
        console.log('📊 Performance Metrics:', JSON.stringify(metrics, null, 2));
        
        this.emit('performanceMetrics', metrics);
    }
    
    /**
     * Setup health monitoring for workers
     */
    setupHealthMonitoring() {
        console.log('💓 Setting up health monitoring...');
        
        setInterval(() => {
            this.checkWorkerHealth();
        }, 30000); // Check every 30 seconds
    }
    
    /**
     * Check health of all workers
     */
    checkWorkerHealth() {
        for (const [workerType, workers] of this.workers.entries()) {
            workers.forEach((worker, index) => {
                if (worker.exitCode !== null || worker.exitCode !== undefined) {
                    console.warn(`⚠️ Unhealthy worker detected: ${workerType}-${index}`);
                    this.restartWorker(workerType, index);
                }
            });
        }
    }
    
    /**
     * Restart a worker
     */
    async restartWorker(workerType, workerId) {
        console.log(`🔄 Restarting worker: ${workerType}-${workerId}`);
        
        try {
            const workers = this.workers.get(workerType);
            const oldWorker = workers[workerId];
            
            if (oldWorker) {
                await oldWorker.terminate();
            }
            
            const newWorker = await this.createWorker(workerType, workerId);
            workers[workerId] = newWorker;
            
            console.log(`✅ Worker restarted: ${workerType}-${workerId}`);
        } catch (error) {
            console.error(`❌ Failed to restart worker ${workerType}-${workerId}:`, error);
        }
    }
    
    /**
     * Graceful shutdown
     */
    async shutdown() {
        console.log('🛑 Shutting down Service Worker Manager...');
        
        for (const [workerType, workers] of this.workers.entries()) {
            console.log(`   Terminating ${workerType} workers...`);
            
            await Promise.all(workers.map(worker => worker.terminate()));
        }
        
        console.log('✅ Service Worker Manager shutdown complete');
    }
}

// Main execution
async function main() {
    if (cluster.isMaster) {
        console.log('🚀 Starting Service Worker Manager (Master Process)');
        
        const manager = new ServiceWorkerManager({
            maxWorkers: parseInt(process.env.MAX_WORKERS) || os.cpus().length,
            gpuEnabled: process.env.CUDA_ENABLED === 'true'
        });
        
        // Start the service worker manager
        await manager.start();
        
        // Example tasks for testing
        setInterval(() => {
            manager.addTask({
                type: 'gpu-parse',
                data: { content: 'Sample legal document for parsing' }
            });
            
            manager.addTask({
                type: 'index-document',
                data: { id: 'doc_' + Date.now(), content: 'Legal case document' }
            });
            
            manager.addTask({
                type: 'embed-text',
                data: { text: 'Contract clause analysis' }
            });
            
            manager.addTask({
                type: 'error-analysis',
                data: { errorLog: 'TypeScript compilation error' }
            });
        }, 5000); // Add test tasks every 5 seconds
        
        // Graceful shutdown handling
        process.on('SIGINT', async () => {
            console.log('\n🛑 Received SIGINT, shutting down gracefully...');
            await manager.shutdown();
            process.exit(0);
        });
        
        process.on('SIGTERM', async () => {
            console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
            await manager.shutdown();
            process.exit(0);
        });
        
    } else {
        // Worker process (if using cluster module)
        console.log(`🔧 Worker process ${process.pid} started`);
    }
}

// Export for use as module
export default ServiceWorkerManager;

// Run if executed directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    main().catch(console.error);
}
