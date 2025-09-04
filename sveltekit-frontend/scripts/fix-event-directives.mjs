#!/usr/bin/env zx

/**
 * Fix Deprecated Event Directives - GPU Accelerated
 * Converts on:click -> onclick, on:submit -> onsubmit, etc.
 */

import 'zx/globals'
import { promises as fs } from 'fs'
import cluster from 'cluster'
import os from 'os'

const MAX_WORKERS = Math.min(os.cpus().length, 8)

// Event directive mappings (Svelte 4 -> Svelte 5)
const EVENT_MAPPINGS = {
  'on:click': 'onclick',
  'on:submit': 'onsubmit', 
  'on:change': 'onchange',
  'on:input': 'oninput',
  'on:focus': 'onfocus',
  'on:blur': 'onblur',
  'on:keydown': 'onkeydown',
  'on:keyup': 'onkeyup',
  'on:mouseenter': 'onmouseenter',
  'on:mouseleave': 'onmouseleave',
  'on:load': 'onload',
  'on:error': 'onerror'
}

async function main() {
  console.log(chalk.cyan('🔧 GPU-Accelerated Event Directive Fixes'))
  
  if (cluster.isPrimary) {
    await runPrimaryProcess()
  } else {
    await runWorkerProcess()
  }
}

async function runPrimaryProcess() {
  console.log(chalk.green('🎯 Starting concurrent event directive fixes'))
  
  // Get all Svelte files with deprecated event handlers
  const svelteFiles = await glob('src/**/*.svelte')
  
  const filesToFix = []
  for (const file of svelteFiles) {
    const content = await fs.readFile(file, 'utf8')
    const hasDeprecatedEvents = Object.keys(EVENT_MAPPINGS).some(pattern => 
      content.includes(pattern + '=')
    )
    
    if (hasDeprecatedEvents) {
      filesToFix.push(file)
    }
  }
  
  console.log(chalk.yellow(`📋 Found ${filesToFix.length} files with deprecated event handlers`))
  
  // Fork workers
  const workers = []
  for (let i = 0; i < MAX_WORKERS; i++) {
    const worker = cluster.fork({ WORKER_ID: i })
    workers.push(worker)
  }
  
  // Distribute files across workers
  const chunkSize = Math.ceil(filesToFix.length / workers.length)
  const promises = workers.map((worker, index) => {
    const start = index * chunkSize
    const chunk = filesToFix.slice(start, start + chunkSize)
    
    return processFilesWithWorker(worker, chunk)
  })
  
  try {
    const results = await Promise.allSettled(promises)
    
    let totalFixed = 0
    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        totalFixed += result.value.filesFixed
        console.log(chalk.green(`✅ Worker ${index}: Fixed ${result.value.filesFixed} files`))
      } else {
        console.log(chalk.red(`❌ Worker ${index}: ${result.reason}`))
      }
    })
    
    console.log(chalk.cyan(`\n🎉 Total files fixed: ${totalFixed}`))
    
    // Verify fixes work
    console.log(chalk.blue('🔍 Verifying fixes...'))
    await $`npm run check:ultra-fast`
    
  } finally {
    workers.forEach(worker => worker.kill())
  }
}

function processFilesWithWorker(worker, files) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Worker timeout'))
    }, 30000)

    worker.on('message', (message) => {
      if (message.type === 'result') {
        clearTimeout(timeout)
        resolve(message.data)
      } else if (message.type === 'error') {
        clearTimeout(timeout)
        reject(new Error(message.error))
      }
    })

    worker.send({ type: 'process-files', files })
  })
}

async function runWorkerProcess() {
  const workerId = process.env.WORKER_ID
  console.log(chalk.blue(`👷 Worker ${workerId}: Ready for event directive fixes`))
  
  process.on('message', async (message) => {
    if (message.type === 'process-files') {
      try {
        const result = await processFiles(message.files)
        process.send({ type: 'result', data: result })
      } catch (error) {
        process.send({ type: 'error', error: error.message })
      }
    }
  })
}

async function processFiles(files) {
  let filesFixed = 0
  let totalReplacements = 0
  
  for (const file of files) {
    try {
      let content = await fs.readFile(file, 'utf8')
      let fileChanged = false
      
      // Apply all event mapping replacements
      for (const [oldEvent, newEvent] of Object.entries(EVENT_MAPPINGS)) {
        const pattern = new RegExp(`${oldEvent.replace(':', '\\:')}=`, 'g')
        const matches = content.match(pattern)
        
        if (matches) {
          content = content.replace(pattern, `${newEvent}=`)
          totalReplacements += matches.length
          fileChanged = true
        }
      }
      
      // Write back if changed
      if (fileChanged) {
        await fs.writeFile(file, content, 'utf8')
        filesFixed++
        console.log(chalk.green(`✨ Fixed: ${file.replace(process.cwd() + '\\', '')}`))
      }
      
    } catch (error) {
      console.error(chalk.red(`💥 Error processing ${file}:`), error.message)
    }
  }
  
  return { filesFixed, totalReplacements }
}

main().catch(error => {
  console.error(chalk.red('💥 Fatal error:'), error)
  process.exit(1)
})