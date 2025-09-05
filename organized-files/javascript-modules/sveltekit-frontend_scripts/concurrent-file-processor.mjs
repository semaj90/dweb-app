/**
 * Concurrent File Processor Worker
 * High-performance parallel file processing for development and production
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { readFile, writeFile, readdir, stat } from 'fs/promises';
import { join, extname, resolve, dirname } from 'path';
import { performance } from 'perf_hooks';
import { createHash } from 'crypto';
import { fileURLToPath } from 'url';
import cluster from 'cluster';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// File processing configuration
const PROCESSING_CONFIG = {
  maxConcurrency: 4,
  chunkSize: 10,
  supportedExtensions: ['.ts', '.js', '.svelte', '.css', '.json', '.md'],
  excludePatterns: [
    /node_modules/,
    /\.git/,
    /dist/,
    /build/,
    /\.svelte-kit/,
    /coverage/
  ],
  processingTimeout: 30000,
  memoryLimit: 256 * 1024 * 1024 // 256MB
};

/**
 * High-performance file processor with worker thread support
 */
class ConcurrentFileProcessor {
  constructor(options = {}) {
    this.options = {
      ...PROCESSING_CONFIG,
      ...options
    };
    
    this.workers = [];
    this.processingQueue = [];
    this.results = new Map();
    this.metrics = {
      filesProcessed: 0,
      totalProcessingTime: 0,
      errorCount: 0,
      startTime: performance.now()
    };
    
    // Worker management
    this.workerPool = [];
    this.availableWorkers = [];
    
    if (isMainThread) {
      this.initializeWorkerPool();
    }
  }
  
  /**
   * Initialize worker pool for parallel processing
   */
  initializeWorkerPool() {
    for (let i = 0; i < this.options.maxConcurrency; i++) {
      const worker = new Worker(__filename, {
        workerData: {
          workerId: i,
          isWorker: true,
          options: this.options
        }
      });
      
      worker.on('message', (result) => {
        this.handleWorkerMessage(worker, result);
      });
      
      worker.on('error', (error) => {
        console.error(`Worker ${i} error:`, error);
        this.metrics.errorCount++;
      });
      
      worker.on('exit', (code) => {
        if (code !== 0) {
          console.error(`Worker ${i} stopped with exit code ${code}`);
        }
      });
      
      this.workerPool.push(worker);
      this.availableWorkers.push(worker);
    }
    
    console.log(`✅ Initialized file processor with ${this.options.maxConcurrency} workers`);
  }
  
  /**
   * Process directory recursively with parallel workers
   */
  async processDirectory(directoryPath, processingFunction) {
    const startTime = performance.now();
    console.log(`🔄 Processing directory: ${directoryPath}`);
    
    try {
      const files = await this.discoverFiles(directoryPath);
      const filteredFiles = this.filterFiles(files);
      
      console.log(`📁 Discovered ${filteredFiles.length} files to process`);
      
      // Split files into chunks for parallel processing
      const chunks = this.chunkArray(filteredFiles, this.options.chunkSize);
      const processingPromises = [];
      
      for (const chunk of chunks) {
        const promise = this.processFileChunk(chunk, processingFunction);
        processingPromises.push(promise);
      }
      
      const results = await Promise.all(processingPromises);
      const flatResults = results.flat();
      
      const processingTime = performance.now() - startTime;
      this.metrics.filesProcessed += filteredFiles.length;
      this.metrics.totalProcessingTime += processingTime;
      
      console.log(`✅ Processed ${filteredFiles.length} files in ${processingTime.toFixed(2)}ms`);
      
      return {
        files: flatResults,
        metrics: this.getMetrics(),
        processingTime
      };
      
    } catch (error) {
      console.error(`❌ Directory processing error:`, error);
      this.metrics.errorCount++;
      throw error;
    }
  }
  
  /**
   * Discover files recursively
   */
  async discoverFiles(directoryPath, files = []) {
    try {
      const entries = await readdir(directoryPath);
      
      const promises = entries.map(async (entry) => {
        const fullPath = join(directoryPath, entry);
        const stats = await stat(fullPath);
        
        if (stats.isDirectory()) {
          if (!this.shouldExcludePath(fullPath)) {
            await this.discoverFiles(fullPath, files);
          }
        } else if (stats.isFile()) {
          if (this.shouldProcessFile(fullPath)) {
            files.push({
              path: fullPath,
              size: stats.size,
              modified: stats.mtime
            });
          }
        }
      });
      
      await Promise.all(promises);
      return files;
      
    } catch (error) {
      console.error(`Error discovering files in ${directoryPath}:`, error);
      return files;
    }
  }
  
  /**
   * Filter files based on configuration
   */
  filterFiles(files) {
    return files.filter(file => {
      // Size limit check
      if (file.size > this.options.memoryLimit) {
        console.warn(`⚠️ Skipping large file: ${file.path} (${file.size} bytes)`);
        return false;
      }
      
      // Extension check
      const ext = extname(file.path).toLowerCase();
      if (!this.options.supportedExtensions.includes(ext)) {
        return false;
      }
      
      return true;
    });
  }
  
  /**
   * Process chunk of files using available worker
   */
  async processFileChunk(files, processingFunction) {
    return new Promise((resolve, reject) => {
      const worker = this.getAvailableWorker();
      
      if (!worker) {
        // Fallback to main thread processing
        return this.processFilesMainThread(files, processingFunction)
          .then(resolve)
          .catch(reject);
      }
      
      const timeout = setTimeout(() => {
        reject(new Error(`Worker processing timeout for chunk of ${files.length} files`));
      }, this.options.processingTimeout);
      
      worker.postMessage({
        type: 'process_files',
        files: files,
        processingFunction: processingFunction.toString()
      });
      
      const messageHandler = (result) => {
        clearTimeout(timeout);
        worker.off('message', messageHandler);
        this.releaseWorker(worker);
        
        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result.data);
        }
      };
      
      worker.on('message', messageHandler);
    });
  }
  
  /**
   * Get available worker from pool
   */
  getAvailableWorker() {
    return this.availableWorkers.pop() || null;
  }
  
  /**
   * Release worker back to pool
   */
  releaseWorker(worker) {
    this.availableWorkers.push(worker);
  }
  
  /**
   * Fallback main thread processing
   */
  async processFilesMainThread(files, processingFunction) {
    const results = [];
    
    for (const file of files) {
      try {
        const result = await this.processFile(file, processingFunction);
        results.push(result);
      } catch (error) {
        console.error(`Error processing ${file.path}:`, error);
        this.metrics.errorCount++;
      }
    }
    
    return results;
  }
  
  /**
   * Process single file
   */
  async processFile(file, processingFunction) {
    const startTime = performance.now();
    
    try {
      // Read file content
      const content = await readFile(file.path, 'utf8');
      
      // Calculate file hash for caching
      const hash = createHash('md5').update(content).digest('hex');
      
      // Apply processing function
      let processed;
      if (typeof processingFunction === 'function') {
        processed = await processingFunction(content, file);
      } else if (typeof processingFunction === 'string') {
        // Eval the stringified function (for worker threads)
        const func = eval(`(${processingFunction})`);
        processed = await func(content, file);
      } else {
        processed = content; // No processing
      }
      
      const processingTime = performance.now() - startTime;
      
      return {
        path: file.path,
        originalSize: content.length,
        processedSize: processed ? processed.length : 0,
        hash,
        processingTime,
        processed: !!processed,
        content: processed || content
      };
      
    } catch (error) {
      throw new Error(`Failed to process ${file.path}: ${error.message}`);
    }
  }
  
  /**
   * Handle worker messages
   */
  handleWorkerMessage(worker, message) {
    if (message.type === 'progress') {
      console.log(`Worker progress: ${message.data}`);
    } else if (message.type === 'metrics') {
      // Aggregate worker metrics
      this.metrics.filesProcessed += message.data.filesProcessed;
      this.metrics.totalProcessingTime += message.data.processingTime;
    }
  }
  
  /**
   * Utility functions
   */
  shouldExcludePath(path) {
    return this.options.excludePatterns.some(pattern => pattern.test(path));
  }
  
  shouldProcessFile(path) {
    const ext = extname(path).toLowerCase();
    return this.options.supportedExtensions.includes(ext) && 
           !this.shouldExcludePath(path);
  }
  
  chunkArray(array, chunkSize) {
    const chunks = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }
  
  /**
   * Get processing metrics
   */
  getMetrics() {
    const currentTime = performance.now();
    const totalTime = currentTime - this.metrics.startTime;
    
    return {
      ...this.metrics,
      totalTime,
      averageProcessingTime: this.metrics.filesProcessed > 0 
        ? this.metrics.totalProcessingTime / this.metrics.filesProcessed 
        : 0,
      filesPerSecond: this.metrics.filesProcessed > 0 
        ? (this.metrics.filesProcessed / totalTime) * 1000 
        : 0
    };
  }
  
  /**
   * Cleanup resources
   */
  async cleanup() {
    console.log('🧹 Cleaning up file processor...');
    
    const promises = this.workerPool.map(worker => 
      new Promise(resolve => {
        worker.terminate().then(resolve);
      })
    );
    
    await Promise.all(promises);
    console.log('✅ File processor cleanup complete');
  }
}

/**
 * Worker thread execution
 */
if (!isMainThread && workerData?.isWorker) {
  const processor = new ConcurrentFileProcessor(workerData.options);
  
  parentPort.on('message', async (message) => {
    try {
      if (message.type === 'process_files') {
        const results = [];
        
        for (const file of message.files) {
          const result = await processor.processFile(file, message.processingFunction);
          results.push(result);
          
          // Send progress updates
          parentPort.postMessage({
            type: 'progress',
            data: `Processed ${file.path}`
          });
        }
        
        parentPort.postMessage({
          type: 'result',
          data: results
        });
        
      } else {
        parentPort.postMessage({
          type: 'error',
          error: `Unknown message type: ${message.type}`
        });
      }
      
    } catch (error) {
      parentPort.postMessage({
        type: 'error',
        error: error.message
      });
    }
  });
  
  // Send periodic metrics
  setInterval(() => {
    parentPort.postMessage({
      type: 'metrics',
      data: processor.getMetrics()
    });
  }, 5000);
}

/**
 * Example processing functions
 */

// TypeScript/JavaScript processing
const processTypeScript = async (content, file) => {
  // Add type checking, linting, or transformations
  if (content.includes('export') || content.includes('import')) {
    // Process ES modules
    return content.replace(/console\.log\(/g, '// console.log(');
  }
  return content;
};

// Svelte component processing
const processSvelte = async (content, file) => {
  // Process Svelte components - check for Svelte 5 compatibility
  if (content.includes('export let')) {
    console.warn(`⚠️ Found Svelte 4 export let in ${file.path} - consider updating to $props`);
  }
  
  if (content.includes('<slot>') && content.includes('<script>')) {
    console.warn(`⚠️ Found <slot> usage in ${file.path} - consider updating to {@render ...}`);
  }
  
  return content;
};

// CSS processing
const processCSS = async (content, file) => {
  // Process CSS - add vendor prefixes, optimize, etc.
  return content.replace(/@import\s+["']([^"']+)["']/g, (match, url) => {
    return `@import "${url}" /* processed */`;
  });
};

// Export for usage in concurrent server
export default ConcurrentFileProcessor;
export { processTypeScript, processSvelte, processCSS };

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const processor = new ConcurrentFileProcessor();
  
  const sourceDir = process.argv[2] || './src';
  
  processor.processDirectory(resolve(sourceDir), processSvelte)
    .then(result => {
      console.log('\n📊 Processing Results:');
      console.log(`Files processed: ${result.files.length}`);
      console.log(`Total time: ${result.processingTime.toFixed(2)}ms`);
      console.log(`Average time per file: ${result.metrics.averageProcessingTime.toFixed(2)}ms`);
      console.log(`Files per second: ${result.metrics.filesPerSecond.toFixed(2)}`);
      
      if (result.metrics.errorCount > 0) {
        console.warn(`⚠️ Errors encountered: ${result.metrics.errorCount}`);
      }
    })
    .catch(error => {
      console.error('❌ Processing failed:', error);
      process.exit(1);
    })
    .finally(() => {
      processor.cleanup();
    });
}