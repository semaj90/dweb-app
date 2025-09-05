#!/usr/bin/env node

import 'zx/globals'
import { createSpinner } from 'nanospinner'
import chalk from 'chalk'
import pLimit from 'p-limit'
import pRetry from 'p-retry'
import boxen from 'boxen'
import gradient from 'gradient-string'

// Configure zx
$.verbose = false

const limit = pLimit(5) // Concurrent service start limit

class SmartServiceManager {
  constructor() {
    this.services = {}
    this.processes = new Map()
    this.loadDetectionResults()
  }

  async loadDetectionResults() {
    try {
      const resultsFile = await fs.readFile('detection-results.json', 'utf8')
      const results = JSON.parse(resultsFile)
      this.services = results.services
    } catch (error) {
      console.log(chalk.yellow('⚠️ No detection results found, running detection...'))
      // Import and run detection
      const { default: SmartServiceDetector } = await import('./detect-services.mjs')
      const detector = new SmartServiceDetector()
      this.services = await detector.detectAll()
    }
  }

  async startAll() {
    console.log(gradient.pastel.multiline(`
    ╔══════════════════════════════════════════════════╗
    ║    🚀 STARTING EVIDENCE PROCESSING SERVICES     ║
    ║              Smart Concurrent Startup           ║
    ╚══════════════════════════════════════════════════╝
    `))

    const spinner = createSpinner('🔍 Analyzing service startup strategy...').start()
    
    try {
      // Analyze what needs to be started
      const strategy = this.generateStartupStrategy()
      spinner.success({ text: '✅ Startup strategy ready!' })
      
      // Show strategy
      this.displayStrategy(strategy)
      
      // Start services in optimal order with concurrency
      await this.executeStartupStrategy(strategy)
      
      // Verify all services are running
      await this.verifyServices()
      
      this.showSuccessMessage()
      
    } catch (error) {
      spinner.error({ text: `❌ Service startup failed: ${error.message}` })
      throw error
    }
  }

  generateStartupStrategy() {
    const strategy = {
      system: [],
      portable: [],
      running: [],
      missing: []
    }

    for (const [serviceName, info] of Object.entries(this.services)) {
      if (info.running) {
        strategy.running.push(serviceName)
      } else if (info.installed && !info.portable) {
        strategy.system.push(serviceName)
      } else if (info.installed && info.portable) {
        strategy.portable.push(serviceName)
      } else {
        strategy.missing.push(serviceName)
      }
    }

    return strategy
  }

  displayStrategy(strategy) {
    console.log('\n' + chalk.bold('📋 STARTUP STRATEGY'))
    console.log('═'.repeat(40))
    
    if (strategy.running.length > 0) {
      console.log(chalk.green('✅ Already running:'), chalk.dim(strategy.running.join(', ')))
    }
    
    if (strategy.system.length > 0) {
      console.log(chalk.blue('🔧 System services to start:'), chalk.cyan(strategy.system.join(', ')))
    }
    
    if (strategy.portable.length > 0) {
      console.log(chalk.magenta('📦 Portable services to start:'), chalk.yellow(strategy.portable.join(', ')))
    }
    
    if (strategy.missing.length > 0) {
      console.log(chalk.red('❌ Missing services:'), chalk.red(strategy.missing.join(', ')))
    }
    
    console.log('')
  }

  async executeStartupStrategy(strategy) {
    const allPromises = []

    // Start system services first (they may take longer)
    if (strategy.system.length > 0) {
      const systemPromises = strategy.system.map(service => 
        limit(() => this.startSystemService(service))
      )
      allPromises.push(...systemPromises)
    }

    // Start portable services concurrently
    if (strategy.portable.length > 0) {
      const portablePromises = strategy.portable.map(service => 
        limit(() => this.startPortableService(service))
      )
      allPromises.push(...portablePromises)
    }

    // Execute all startup operations concurrently
    const results = await Promise.allSettled(allPromises)
    
    // Report results
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.log(chalk.red(`❌ Failed to start service: ${result.reason.message}`))
      }
    })
  }

  async startSystemService(serviceName) {
    const spinner = createSpinner(`Starting ${serviceName} (system service)...`).start()
    
    try {
      switch (serviceName) {
        case 'postgresql':
          await this.startPostgreSQL()
          break
        case 'redis':
          await this.startRedisService()
          break
        case 'rabbitmq':
          await this.startRabbitMQService()
          break
        default:
          throw new Error(`Unknown system service: ${serviceName}`)
      }
      
      spinner.success({ text: `✅ ${serviceName} started (system)` })
      
    } catch (error) {
      spinner.error({ text: `❌ ${serviceName} failed to start` })
      throw error
    }
  }

  async startPortableService(serviceName) {
    const spinner = createSpinner(`Starting ${serviceName} (portable)...`).start()
    
    try {
      let process
      
      switch (serviceName) {
        case 'qdrant':
          process = await this.startQdrant()
          break
        case 'neo4j':
          process = await this.startNeo4j()
          break
        case 'minio':
          process = await this.startMinIO()
          break
        case 'ollama':
          process = await this.startOllama()
          break
        case 'redis':
          process = await this.startRedisPortable()
          break
        default:
          throw new Error(`Unknown portable service: ${serviceName}`)
      }
      
      if (process) {
        this.processes.set(serviceName, process)
      }
      
      spinner.success({ text: `✅ ${serviceName} started (portable)` })
      
    } catch (error) {
      spinner.error({ text: `❌ ${serviceName} failed to start` })
      throw error
    }
  }

  async startPostgreSQL() {
    try {
      // Try to start PostgreSQL service
      await $`net start postgresql*`.quiet()
    } catch (error) {
      // May already be running or different service name
      // Test if it's accessible
      process.env.PGPASSWORD = '123456'
      await $`psql -U postgres -c "SELECT 1"`.quiet()
    }
  }

  async startRedisService() {
    await pRetry(async () => {
      try {
        await $`sc start Redis`.quiet()
        await sleep(3000)
        
        // Verify Redis is responding
        const ping = await $`redis-cli ping`.quiet()
        if (ping.stdout.trim() !== 'PONG') {
          throw new Error('Redis not responding')
        }
      } catch (error) {
        if (error.message.includes('already')) {
          // Service already running, that's fine
          return
        }
        throw error
      }
    }, { retries: 3, minTimeout: 2000 })
  }

  async startRabbitMQService() {
    await pRetry(async () => {
      try {
        await $`sc start RabbitMQ`.quiet()
        await sleep(5000)
      } catch (error) {
        if (error.message.includes('already')) {
          return
        }
        throw error
      }
    }, { retries: 3, minTimeout: 3000 })
  }

  async startQdrant() {
    const execPath = path.join('services', 'qdrant.exe')
    const configPath = path.join('services', 'qdrant-config.yaml')
    
    const process = $`"${execPath}" --config-path "${configPath}"`
    
    // Give it time to start
    await sleep(3000)
    
    return process
  }

  async startNeo4j() {
    const batPath = path.join('services', 'neo4j', 'bin', 'neo4j.bat')
    
    const process = $`"${batPath}" console`
    
    // Give it time to start
    await sleep(8000)
    
    return process
  }

  async startMinIO() {
    const execPath = path.join('services', 'minio.exe')
    const dataPath = path.join('services', 'minio-data')
    
    // Set MinIO environment variables
    process.env.MINIO_ROOT_USER = 'evidence'
    process.env.MINIO_ROOT_PASSWORD = 'evidence123'
    
    const process = $`"${execPath}" server "${dataPath}" --console-address ":9001"`
    
    // Give it time to start
    await sleep(4000)
    
    return process
  }

  async startOllama() {
    const execPath = path.join('services', 'ollama.exe')
    
    if (!(await this.fileExists(execPath))) {
      console.log(chalk.yellow('⚠️ Ollama not found (optional service)'))
      return null
    }
    
    const process = $`"${execPath}" serve`
    
    // Give it time to start
    await sleep(3000)
    
    return process
  }

  async startRedisPortable() {
    const execPath = path.join('services', 'redis-server.exe')
    
    const process = $`"${execPath}"`
    
    // Give it time to start and verify
    await sleep(3000)
    
    const ping = await $`redis-cli ping`.quiet()
    if (ping.stdout.trim() !== 'PONG') {
      throw new Error('Portable Redis not responding')
    }
    
    return process
  }

  async verifyServices() {
    const spinner = createSpinner('🔍 Verifying service availability...').start()
    
    try {
      const verificationPromises = [
        this.verifyPostgreSQL(),
        this.verifyRedis(),
        this.verifyRabbitMQ(),
        this.verifyQdrant(),
        this.verifyNeo4j(),
        this.verifyMinIO()
      ].map(promise => limit(() => promise))
      
      const results = await Promise.allSettled(verificationPromises)
      
      const serviceStatus = {
        postgresql: results[0].status === 'fulfilled',
        redis: results[1].status === 'fulfilled',
        rabbitmq: results[2].status === 'fulfilled',
        qdrant: results[3].status === 'fulfilled',
        neo4j: results[4].status === 'fulfilled',
        minio: results[5].status === 'fulfilled'
      }
      
      const successCount = Object.values(serviceStatus).filter(Boolean).length
      const totalCount = Object.keys(serviceStatus).length
      
      if (successCount === totalCount) {
        spinner.success({ text: `✅ All ${successCount} services verified!` })
      } else {
        spinner.warn({ text: `⚠️ ${successCount}/${totalCount} services verified` })
      }
      
      // Display detailed status
      console.log('\n' + chalk.bold('📊 SERVICE STATUS'))
      console.log('═'.repeat(30))
      
      for (const [service, status] of Object.entries(serviceStatus)) {
        const icon = status ? '✅' : '❌'
        const statusText = status ? 'Running' : 'Failed'
        console.log(`${icon} ${service.padEnd(12)} ${statusText}`)
      }
      
    } catch (error) {
      spinner.error({ text: `❌ Service verification failed: ${error.message}` })
      throw error
    }
  }

  async verifyPostgreSQL() {
    process.env.PGPASSWORD = '123456'
    await $`psql -U postgres -c "SELECT 1" -q`.quiet()
  }

  async verifyRedis() {
    const result = await $`redis-cli ping`.quiet()
    if (result.stdout.trim() !== 'PONG') {
      throw new Error('Redis not responding')
    }
  }

  async verifyRabbitMQ() {
    // Check if RabbitMQ is listening on port 5672
    const netstat = await $`netstat -an`.quiet()
    if (!netstat.stdout.includes(':5672')) {
      throw new Error('RabbitMQ not listening')
    }
  }

  async verifyQdrant() {
    await fetch('http://localhost:6333/collections')
  }

  async verifyNeo4j() {
    await fetch('http://localhost:7474/')
  }

  async verifyMinIO() {
    await fetch('http://localhost:9000/minio/health/live')
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath)
      return true
    } catch {
      return false
    }
  }

  showSuccessMessage() {
    const message = boxen(
      chalk.bold.green('🎉 ALL SERVICES STARTED!\n\n') +
      chalk.white('Evidence Processing System is ready:\n\n') +
      chalk.cyan('📋 Web Interfaces:\n') +
      chalk.dim('  • RabbitMQ Management: http://localhost:15672 (guest/guest)\n') +
      chalk.dim('  • Neo4j Browser: http://localhost:7474 (neo4j/neo4j)\n') +
      chalk.dim('  • MinIO Console: http://localhost:9001 (evidence/evidence123)\n') +
      chalk.dim('  • Qdrant Dashboard: http://localhost:6333/dashboard\n\n') +
      chalk.bold('Next steps:\n') +
      chalk.dim('  npm run database  # Setup database\n') +
      chalk.dim('  npm run test      # Test system\n') +
      chalk.dim('  npm run worker    # Start processing\n') +
      chalk.dim('  npm run stop      # Stop all services'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'double',
        borderColor: 'green'
      }
    )
    
    console.log('\n' + message)
  }

  // Graceful shutdown handler
  setupShutdownHandler() {
    const shutdown = async () => {
      console.log(chalk.yellow('\n🛑 Shutting down services...'))
      
      for (const [serviceName, process] of this.processes) {
        try {
          process.kill('SIGTERM')
          console.log(chalk.dim(`  Stopped ${serviceName}`))
        } catch (error) {
          console.log(chalk.red(`  Failed to stop ${serviceName}`))
        }
      }
      
      process.exit(0)
    }

    process.on('SIGINT', shutdown)
    process.on('SIGTERM', shutdown)
  }
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const manager = new SmartServiceManager()
  manager.setupShutdownHandler()
  await manager.startAll()
  
  // Keep process alive to maintain services
  console.log(chalk.dim('\n💡 Press Ctrl+C to stop all services'))
  await new Promise(() => {}) // Keep alive
}

export default SmartServiceManager
