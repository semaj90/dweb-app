#!/usr/bin/env zx

/**
 * Concurrent Codebase Indexer using zx
 * High-performance file processing with concurrent execution
 */

import { $, argv, glob, fs, os, chalk, sleep, spinner } from 'zx'
import { Worker } from 'worker_threads'
import { fileURLToPath } from 'url'
import path from 'path'

// Configuration
const CONFIG = {
  MAX_WORKERS: os.cpus().length,
  BATCH_SIZE: 100,
  MAX_FILE_SIZE: 1024 * 1024, // 1MB
  OLLAMA_URL: 'http://localhost:11434',
  GO_SERVICE_URL: 'http://localhost:8081',
  AUTOGEN_SERVICE_URL: 'http://localhost:8083',
  OUTPUT_DIR: './indexing-output',
  SUPPORTED_EXTENSIONS: ['.ts', '.js', '.svelte', '.go', '.py', '.md', '.json', '.sql', '.css', '.scss']
}

// Global state
let progressBar
let startTime = Date.now()
let totalFiles = 0
let processedFiles = 0
let failedFiles = 0
let results = []

$.verbose = false // Reduce noise

/**
 * Main indexing orchestration function
 */
async function main() {
  console.log(chalk.cyan('🚀 Starting Concurrent Codebase Indexing'))
  console.log(chalk.gray(`Max Workers: ${CONFIG.MAX_WORKERS}, Batch Size: ${CONFIG.BATCH_SIZE}`))
  
  try {
    // Parse command line arguments
    const rootPath = argv._[0] || process.cwd()
    const workers = parseInt(argv.workers) || CONFIG.MAX_WORKERS
    const batchSize = parseInt(argv.batch) || CONFIG.BATCH_SIZE
    
    console.log(chalk.blue(`📁 Indexing path: ${rootPath}`))
    
    // Phase 1: Service Health Checks
    await checkServices()
    
    // Phase 2: File Discovery
    const filePaths = await discoverFiles(rootPath)
    totalFiles = filePaths.length
    
    console.log(chalk.green(`✅ Discovered ${totalFiles} files`))
    
    // Phase 3: Concurrent Processing
    await processFilesConcurrently(filePaths, workers, batchSize)
    
    // Phase 4: Generate Reports
    await generateReports()
    
    console.log(chalk.green('🎉 Indexing completed successfully!'))
    
  } catch (error) {
    console.error(chalk.red(`❌ Indexing failed: ${error.message}`))
    process.exit(1)
  }
}

/**
 * Check if all required services are running
 */
async function checkServices() {
  const services = [
    { name: 'Ollama', url: CONFIG.OLLAMA_URL + '/api/tags' },
    { name: 'Go Service', url: CONFIG.GO_SERVICE_URL + '/api/health' },
  ]
  
  console.log(chalk.yellow('🔍 Checking service availability...'))
  
  for (const service of services) {
    try {
      const response = await fetch(service.url, { 
        method: 'GET',
        timeout: 5000 
      })
      
      if (response.ok) {
        console.log(chalk.green(`✅ ${service.name} is running`))
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (error) {
      console.log(chalk.red(`❌ ${service.name} is not available: ${error.message}`))
      
      // Try to start services
      if (service.name === 'Go Service') {
        console.log(chalk.yellow('🚀 Attempting to start Go service...'))
        await startGoService()
      }
    }
  }
}

/**
 * Start the Go indexing service
 */
async function startGoService() {
  try {
    // Build and start Go service in background
    await $`cd indexing-system && go mod tidy`
    
    const goProcess = $`cd indexing-system && go run async-indexer.go`
    goProcess.nothrow()
    
    // Wait a moment for service to start
    await sleep(3000)
    
    // Test if service is now available
    const response = await fetch(CONFIG.GO_SERVICE_URL + '/api/health')
    if (response.ok) {
      console.log(chalk.green('✅ Go service started successfully'))
    } else {
      console.log(chalk.yellow('⚠️ Go service may still be starting...'))
    }
  } catch (error) {
    console.log(chalk.red(`Failed to start Go service: ${error.message}`))
  }
}

/**
 * Discover all files to be processed
 */
async function discoverFiles(rootPath) {
  const spinnerInstance = spinner()
  spinnerInstance.start('🔍 Discovering files...')
  
  try {
    const allFiles = []
    
    // Use glob patterns for each supported extension
    for (const ext of CONFIG.SUPPORTED_EXTENSIONS) {
      const pattern = `${rootPath}/**/*${ext}`
      const files = await glob(pattern, {
        ignore: [
          '**/node_modules/**',
          '**/.git/**',
          '**/dist/**',
          '**/build/**',
          '**/.next/**',
          '**/target/**',
          '**/__pycache__/**'
        ]
      })
      allFiles.push(...files)
    }
    
    // Filter out files that are too large
    const validFiles = []
    for (const filePath of allFiles) {
      try {
        const stats = await fs.stat(filePath)
        if (stats.size <= CONFIG.MAX_FILE_SIZE && stats.isFile()) {
          validFiles.push(filePath)
        }
      } catch (error) {
        // Skip files we can't access
      }
    }
    
    spinnerInstance.stop()
    return validFiles
  } catch (error) {
    spinnerInstance.stop()
    throw error
  }
}

/**
 * Process files using concurrent workers
 */
async function processFilesConcurrently(filePaths, maxWorkers, batchSize) {
  console.log(chalk.blue(`⚡ Processing ${filePaths.length} files with ${maxWorkers} workers`))
  
  // Create output directory
  await fs.ensureDir(CONFIG.OUTPUT_DIR)
  
  // Initialize progress tracking
  progressBar = createProgressBar(filePaths.length)
  
  // Split files into batches
  const batches = []
  for (let i = 0; i < filePaths.length; i += batchSize) {
    batches.push(filePaths.slice(i, i + batchSize))
  }
  
  // Process batches concurrently with worker limit
  const semaphore = new Semaphore(maxWorkers)
  const promises = batches.map(async (batch, batchIndex) => {
    await semaphore.acquire()
    try {
      await processBatch(batch, batchIndex)
    } finally {
      semaphore.release()
    }
  })
  
  await Promise.all(promises)
  progressBar.stop()
}

/**
 * Process a single batch of files
 */
async function processBatch(filePaths, batchIndex) {
  const batchResults = []
  
  for (const filePath of filePaths) {
    try {
      const result = await processFile(filePath)
      batchResults.push(result)
      results.push(result)
      processedFiles++
      
      // Update progress
      progressBar.update(processedFiles)
      
    } catch (error) {
      failedFiles++
      console.log(chalk.red(`❌ Failed to process ${filePath}: ${error.message}`))
    }
  }
  
  // Save batch results
  const batchFile = path.join(CONFIG.OUTPUT_DIR, `batch-${batchIndex}.json`)
  await fs.writeJSON(batchFile, {
    batchIndex,
    processedAt: new Date().toISOString(),
    files: batchResults
  }, { spaces: 2 })
}

/**
 * Process a single file
 */
async function processFile(filePath) {
  const content = await fs.readFile(filePath, 'utf8')
  const stats = await fs.stat(filePath)
  
  const fileInfo = {
    path: filePath,
    size: stats.size,
    modified: stats.mtime,
    language: getLanguageFromPath(filePath),
    contentHash: await generateHash(content)
  }
  
  // Strategy 1: Try Go service for embedding
  let embedding = null
  try {
    const goResponse = await fetch(`${CONFIG.GO_SERVICE_URL}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: content.substring(0, 8000) }), // Limit content size
      timeout: 10000
    })
    
    if (goResponse.ok) {
      const data = await goResponse.json()
      embedding = data.embedding
    }
  } catch (error) {
    // Fallback strategies below
  }
  
  // Strategy 2: Direct Ollama call if Go service failed
  if (!embedding) {
    try {
      const ollamaResponse = await fetch(`${CONFIG.OLLAMA_URL}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: content.substring(0, 8000)
        }),
        timeout: 15000
      })
      
      if (ollamaResponse.ok) {
        const data = await ollamaResponse.json()
        embedding = data.embedding
      }
    } catch (error) {
      // Will remain null
    }
  }
  
  return {
    ...fileInfo,
    embedding: embedding,
    embeddingSize: embedding ? embedding.length : 0,
    processedAt: new Date().toISOString()
  }
}

/**
 * Generate comprehensive reports
 */
async function generateReports() {
  const endTime = Date.now()
  const processingTime = (endTime - startTime) / 1000
  
  console.log(chalk.yellow('📊 Generating reports...'))
  
  const summary = {
    metadata: {
      totalFiles: totalFiles,
      processedFiles: processedFiles,
      failedFiles: failedFiles,
      processingTimeSeconds: processingTime,
      filesPerSecond: processedFiles / processingTime,
      generatedAt: new Date().toISOString()
    },
    languageDistribution: getLanguageDistribution(results),
    embeddingStats: getEmbeddingStats(results),
    sizeDistribution: getSizeDistribution(results),
    topLargestFiles: getTopLargestFiles(results, 10)
  }
  
  // Save summary report
  await fs.writeJSON(
    path.join(CONFIG.OUTPUT_DIR, 'indexing-summary.json'),
    summary,
    { spaces: 2 }
  )
  
  // Generate HTML report
  await generateHTMLReport(summary)
  
  // Print console summary
  printConsoleSummary(summary)
}

/**
 * Generate HTML visualization report
 */
async function generateHTMLReport(summary) {
  const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codebase Indexing Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }
        .stat-value { font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 0.9em; opacity: 0.9; }
        .chart-container { margin-bottom: 40px; }
        .chart-container canvas { max-height: 400px; }
        .file-list { background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .file-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e9ecef; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Codebase Indexing Report</h1>
            <p>Generated on ${new Date().toLocaleDateString()}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${summary.metadata.totalFiles.toLocaleString()}</div>
                <div class="stat-label">Total Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${summary.metadata.processedFiles.toLocaleString()}</div>
                <div class="stat-label">Processed Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${summary.metadata.filesPerSecond.toFixed(1)}</div>
                <div class="stat-label">Files/Second</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${summary.embeddingStats.withEmbeddings}</div>
                <div class="stat-label">With Embeddings</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Language Distribution</h3>
            <canvas id="languageChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>File Size Distribution</h3>
            <canvas id="sizeChart"></canvas>
        </div>
        
        <div class="file-list">
            <h3>Largest Files</h3>
            ${summary.topLargestFiles.map(file => `
                <div class="file-item">
                    <span>${file.path}</span>
                    <span>${formatBytes(file.size)}</span>
                </div>
            `).join('')}
        </div>
    </div>
    
    <script>
        // Language Distribution Chart
        const languageCtx = document.getElementById('languageChart').getContext('2d');
        new Chart(languageCtx, {
            type: 'doughnut',
            data: {
                labels: ${JSON.stringify(Object.keys(summary.languageDistribution))},
                datasets: [{
                    data: ${JSON.stringify(Object.values(summary.languageDistribution))},
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        // Size Distribution Chart
        const sizeCtx = document.getElementById('sizeChart').getContext('2d');
        new Chart(sizeCtx, {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(Object.keys(summary.sizeDistribution))},
                datasets: [{
                    label: 'Number of Files',
                    data: ${JSON.stringify(Object.values(summary.sizeDistribution))},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
    </script>
</body>
</html>`
  
  await fs.writeFile(
    path.join(CONFIG.OUTPUT_DIR, 'indexing-report.html'),
    htmlContent
  )
  
  console.log(chalk.green(`📊 HTML report generated: ${CONFIG.OUTPUT_DIR}/indexing-report.html`))
}

/**
 * Utility Functions
 */

function getLanguageFromPath(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  const languageMap = {
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript',
    '.svelte': 'Svelte',
    '.go': 'Go',
    '.py': 'Python',
    '.md': 'Markdown',
    '.json': 'JSON',
    '.sql': 'SQL',
    '.css': 'CSS',
    '.scss': 'SCSS'
  }
  return languageMap[ext] || 'Unknown'
}

async function generateHash(content) {
  const crypto = await import('crypto')
  return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16)
}

function getLanguageDistribution(results) {
  const distribution = {}
  results.forEach(result => {
    distribution[result.language] = (distribution[result.language] || 0) + 1
  })
  return distribution
}

function getEmbeddingStats(results) {
  const withEmbeddings = results.filter(r => r.embedding).length
  const totalEmbeddingSize = results.reduce((sum, r) => sum + (r.embeddingSize || 0), 0)
  const avgEmbeddingSize = withEmbeddings > 0 ? totalEmbeddingSize / withEmbeddings : 0
  
  return {
    withEmbeddings,
    withoutEmbeddings: results.length - withEmbeddings,
    averageEmbeddingSize: Math.round(avgEmbeddingSize)
  }
}

function getSizeDistribution(results) {
  const ranges = {
    '0-1KB': 0,
    '1-10KB': 0,
    '10-100KB': 0,
    '100KB-1MB': 0,
    '1MB+': 0
  }
  
  results.forEach(result => {
    const size = result.size
    if (size < 1024) ranges['0-1KB']++
    else if (size < 10240) ranges['1-10KB']++
    else if (size < 102400) ranges['10-100KB']++
    else if (size < 1048576) ranges['100KB-1MB']++
    else ranges['1MB+']++
  })
  
  return ranges
}

function getTopLargestFiles(results, count) {
  return results
    .sort((a, b) => b.size - a.size)
    .slice(0, count)
    .map(r => ({ path: r.path, size: r.size }))
}

function createProgressBar(total) {
  // Simple progress bar implementation
  let current = 0
  const width = 50
  
  return {
    update: (value) => {
      current = value
      const percentage = Math.round((current / total) * 100)
      const progress = Math.round((current / total) * width)
      const bar = '█'.repeat(progress) + '░'.repeat(width - progress)
      
      process.stdout.write(`\r${chalk.cyan(bar)} ${percentage}% (${current}/${total})`)
    },
    stop: () => {
      console.log(chalk.green(`\n✅ Processing completed: ${current}/${total} files`))
    }
  }
}

function printConsoleSummary(summary) {
  console.log(chalk.cyan('\n📊 INDEXING SUMMARY'))
  console.log(chalk.cyan('='.repeat(50)))
  console.log(chalk.white(`📁 Total Files: ${summary.metadata.totalFiles.toLocaleString()}`))
  console.log(chalk.green(`✅ Processed: ${summary.metadata.processedFiles.toLocaleString()}`))
  console.log(chalk.red(`❌ Failed: ${summary.metadata.failedFiles.toLocaleString()}`))
  console.log(chalk.yellow(`⏱️  Processing Time: ${summary.metadata.processingTimeSeconds.toFixed(1)}s`))
  console.log(chalk.blue(`⚡ Speed: ${summary.metadata.filesPerSecond.toFixed(1)} files/sec`))
  console.log(chalk.magenta(`🧠 With Embeddings: ${summary.embeddingStats.withEmbeddings}`))
  
  console.log(chalk.cyan('\n🌐 LANGUAGE DISTRIBUTION'))
  Object.entries(summary.languageDistribution)
    .sort(([,a], [,b]) => b - a)
    .forEach(([lang, count]) => {
      console.log(chalk.gray(`  ${lang}: ${count} files`))
    })
}

/**
 * Semaphore class for limiting concurrent operations
 */
class Semaphore {
  constructor(max) {
    this.max = max
    this.current = 0
    this.queue = []
  }
  
  async acquire() {
    return new Promise((resolve) => {
      if (this.current < this.max) {
        this.current++
        resolve()
      } else {
        this.queue.push(resolve)
      }
    })
  }
  
  release() {
    this.current--
    if (this.queue.length > 0) {
      this.current++
      const resolve = this.queue.shift()
      resolve()
    }
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error)
}

export { main, CONFIG }