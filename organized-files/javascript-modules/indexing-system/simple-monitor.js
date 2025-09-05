#!/usr/bin/env node

/**
 * Simple Monitoring Script for Indexing Progress
 * Checks services and shows basic status
 */

import http from 'http'
import fs from 'fs/promises'
import path from 'path'

// Configuration
const SERVICES = [
  { name: 'Ollama', url: 'http://localhost:11434/api/tags' },
  { name: 'Go Indexer', url: 'http://localhost:8081/api/health' },
  { name: 'Monitor Dashboard', url: 'http://localhost:8084/api/health' },
  { name: 'GPU Clustering', url: 'http://localhost:8085/api/health' }
]

const OUTPUT_DIR = './indexing-output'

async function checkService(service) {
  try {
    const response = await fetch(service.url, { 
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    })
    return {
      name: service.name,
      status: response.ok ? 'Online' : `Error ${response.status}`,
      healthy: response.ok
    }
  } catch (error) {
    return {
      name: service.name,
      status: 'Offline',
      healthy: false,
      error: error.message
    }
  }
}

async function checkIndexingProgress() {
  try {
    const files = await fs.readdir(OUTPUT_DIR)
    const batches = files.filter(f => f.startsWith('batch-')).length
    
    // Try to read summary if available
    try {
      const summaryPath = path.join(OUTPUT_DIR, 'indexing-summary.json')
      const summary = JSON.parse(await fs.readFile(summaryPath, 'utf8'))
      return {
        hasOutput: true,
        batches,
        summary: summary.metadata || {}
      }
    } catch {
      return { hasOutput: true, batches, summary: {} }
    }
  } catch {
    return { hasOutput: false, batches: 0, summary: {} }
  }
}

async function showStatus() {
  console.log('\n🚀 Legal AI Indexing System Status')
  console.log('=' * 50)
  
  // Check services
  console.log('\n📡 Service Status:')
  const serviceChecks = await Promise.all(SERVICES.map(checkService))
  
  for (const service of serviceChecks) {
    const status = service.healthy ? '✅' : '❌'
    console.log(`  ${status} ${service.name}: ${service.status}`)
    if (service.error) {
      console.log(`     └── Error: ${service.error}`)
    }
  }
  
  // Check indexing progress
  console.log('\n📊 Indexing Progress:')
  const progress = await checkIndexingProgress()
  
  if (progress.hasOutput) {
    console.log(`  📁 Output batches: ${progress.batches}`)
    if (progress.summary.totalFiles) {
      console.log(`  📄 Total files: ${progress.summary.totalFiles.toLocaleString()}`)
      console.log(`  ✅ Processed: ${progress.summary.processedFiles?.toLocaleString() || 0}`)
      console.log(`  ⚡ Rate: ${progress.summary.filesPerSecond?.toFixed(1) || 0} files/sec`)
    }
  } else {
    console.log('  📋 No indexing output found yet')
  }
  
  // Show system info
  console.log('\n💻 System Info:')
  console.log(`  🖥️  Platform: ${process.platform} ${process.arch}`)
  console.log(`  📦 Node.js: ${process.version}`)
  console.log(`  🧠 Memory: ${Math.round(process.memoryUsage().rss / 1024 / 1024)}MB`)
  console.log(`  ⏱️  Uptime: ${Math.round(process.uptime())}s`)
  
  console.log('\n🔗 Access Points:')
  console.log('  • Monitor Dashboard: http://localhost:8084')
  console.log('  • Go Indexer API:    http://localhost:8081')  
  console.log('  • GPU Clustering:    http://localhost:8085')
  console.log('  • GRPC Server:       localhost:50052')
  console.log('  • Ollama API:        http://localhost:11434')
  
  console.log('\nℹ️  To check Python AutoGen progress, look for log output in the terminal.')
  console.log('   Output files will appear in ./indexing-output/ directory.')
}

// Run status check
showStatus().catch(console.error)