#!/usr/bin/env node

import 'zx/globals'
import { createSpinner } from 'nanospinner'
import chalk from 'chalk'
import boxen from 'boxen'
import gradient from 'gradient-string'

// Configure zx
$.verbose = false

class SmartWorkerManager {
  constructor() {
    this.workerProcess = null
    this.setupShutdownHandlers()
  }

  async start() {
    console.log(gradient.pastel.multiline(`
    ╔══════════════════════════════════════════════════╗
    ║      🏭 EVIDENCE PROCESSING WORKER              ║
    ║            Smart Startup & Monitoring           ║
    ╚══════════════════════════════════════════════════╝
    `))

    try {
      // Pre-flight checks
      await this.preflightChecks()
      
      // Environment setup
      await this.setupEnvironment()
      
      // Start worker
      await this.startWorker()
      
      // Keep alive and monitor
      await this.monitorWorker()
      
    } catch (error) {
      console.error(chalk.red('❌ Worker startup failed:'), error.message)
      process.exit(1)
    }
  }

  async preflightChecks() {
    const spinner = createSpinner('🔍 Running pre-flight checks...').start()
    
    try {
      // Check if workers directory exists
      const workersDir = path.join(process.cwd(), 'workers')
      await fs.access(workersDir)
      
      // Check if dependencies are installed
      const nodeModulesPath = path.join(workersDir, 'node_modules')
      try {
        await fs.access(nodeModulesPath)
      } catch (error) {
        spinner.warn({ text: '⚠️ Installing worker dependencies...' })
        cd(workersDir)
        await $`npm install`
        cd('..')
      }
      
      // Check if worker file exists
      const workerFile = path.join(workersDir, 'evidenceProcessor.js')
      await fs.access(workerFile)
      
      // Check environment file
      await fs.access('.env')
      
      spinner.success({ text: '✅ Pre-flight checks passed!' })
      
    } catch (error) {
      spinner.error({ text: '❌ Pre-flight checks failed!' })
      throw error
    }
  }

  async setupEnvironment() {
    const spinner = createSpinner('🔧 Setting up environment...').start()
    
    try {
      // Set PostgreSQL password
      process.env.PGPASSWORD = '123456'
      
      // Set Node environment
      process.env.NODE_ENV = 'development'
      
      // Test service connectivity
      await this.testConnectivity()
      
      spinner.success({ text: '✅ Environment ready!' })
      
    } catch (error) {
      spinner.error({ text: '❌ Environment setup failed!' })
      throw error
    }
  }

  async testConnectivity() {
    const tests = [
      { name: 'PostgreSQL', test: () => this.testPostgreSQL() },
      { name: 'Redis', test: () => this.testRedis() },
      { name: 'RabbitMQ', test: () => this.testRabbitMQ() }
    ]
    
    console.log('\n' + chalk.bold('🌐 Service Connectivity Check:'))
    
    for (const { name, test } of tests) {
      try {
        await test()
        console.log(`  ✅ ${name}: Connected`)
      } catch (error) {
        console.log(`  ❌ ${name}: ${error.message}`)
        
        // Provide specific guidance
        if (name === 'PostgreSQL') {
          console.log(chalk.dim('     • Ensure PostgreSQL is running'))
          console.log(chalk.dim('     • Verify password is 123456'))
          console.log(chalk.dim('     • Check database exists: npm run database'))
        } else if (name === 'Redis') {
          console.log(chalk.dim('     • Start Redis: sc start Redis'))
          console.log(chalk.dim('     • Or restart services: npm run start'))
        } else if (name === 'RabbitMQ') {
          console.log(chalk.dim('     • Start RabbitMQ: sc start RabbitMQ'))
          console.log(chalk.dim('     • Or restart services: npm run start'))
        }
        
        throw new Error(`${name} connectivity failed`)
      }
    }
  }

  async testPostgreSQL() {
    await $`psql -U postgres -d evidence_processing -c "SELECT 1;" -q`.quiet()
  }

  async testRedis() {
    const result = await $`redis-cli ping`.quiet()
    if (result.stdout.trim() !== 'PONG') {
      throw new Error('Redis not responding')
    }
  }

  async testRabbitMQ() {
    const netstat = await $`netstat -an`.quiet()
    if (!netstat.stdout.includes(':5672')) {
      throw new Error('RabbitMQ not listening')
    }
  }

  async startWorker() {
    const spinner = createSpinner('🚀 Starting evidence processing worker...').start()
    
    try {
      const workersDir = path.join(process.cwd(), 'workers')
      
      console.log('\n' + chalk.bold('📋 Worker Information:'))
      console.log(`  • Process: Evidence Processing Worker`)
      console.log(`  • Environment: ${process.env.NODE_ENV}`)
      console.log(`  • Directory: ${workersDir}`)
      console.log(`  • Queue: evidence.process.queue`)
      console.log(`  • Log Level: Info`)
      
      console.log('\n' + chalk.bold('🔧 Processing Capabilities:'))
      console.log(`  • OCR: Text extraction from documents`)
      console.log(`  • Embeddings: Vector generation for similarity search`)
      console.log(`  • RAG: AI-powered analysis and insights`)
      console.log(`  • WebSocket: Real-time progress updates`)
      
      spinner.success({ text: '✅ Worker configuration ready!' })
      
      // Start the worker process
      console.log('\n' + chalk.bold('🏭 Starting worker process...'))
      console.log(chalk.dim('Press Ctrl+C to stop the worker gracefully'))
      console.log('─'.repeat(60))
      
      cd(workersDir)
      
      // Start worker with proper handling
      this.workerProcess = $`node evidenceProcessor.js`
      
      // Handle worker output
      this.workerProcess.stdout.on('data', (data) => {
        process.stdout.write(chalk.cyan('[WORKER] ') + data.toString())
      })
      
      this.workerProcess.stderr.on('data', (data) => {
        process.stderr.write(chalk.red('[ERROR] ') + data.toString())
      })
      
      this.workerProcess.on('close', (code) => {
        if (code !== 0) {
          console.log(chalk.red(`\n❌ Worker exited with code ${code}`))
          this.showTroubleshooting()
        } else {
          console.log(chalk.green('\n✅ Worker stopped gracefully'))
        }
      })
      
      // Give it a moment to start
      await sleep(3000)
      
    } catch (error) {
      spinner.error({ text: '❌ Worker startup failed!' })
      throw error
    }
  }

  async monitorWorker() {
    // Keep the process alive and show status
    console.log('\n' + boxen(
      chalk.bold.green('🎉 WORKER STARTED SUCCESSFULLY!\n\n') +
      chalk.white('Evidence Processing Worker is now running and ready to:\n\n') +
      chalk.cyan('• Process uploaded evidence files\n') +
      chalk.cyan('• Extract text via OCR\n') +
      chalk.cyan('• Generate vector embeddings\n') +
      chalk.cyan('• Perform AI analysis\n') +
      chalk.cyan('• Send real-time progress updates\n\n') +
      chalk.bold('🌐 Web Interfaces:\n') +
      chalk.dim('  • RabbitMQ Management: http://localhost:15672\n') +
      chalk.dim('  • Neo4j Browser: http://localhost:7474\n') +
      chalk.dim('  • MinIO Console: http://localhost:9001\n') +
      chalk.dim('  • Qdrant Dashboard: http://localhost:6333/dashboard\n\n') +
      chalk.yellow('Press Ctrl+C to stop the worker'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'double',
        borderColor: 'green'
      }
    ))
    
    // Wait for worker process to complete or be interrupted
    try {
      await this.workerProcess
    } catch (error) {
      if (error.signal === 'SIGINT') {
        console.log(chalk.yellow('\n🛑 Graceful shutdown initiated...'))
      } else {
        console.log(chalk.red('\n❌ Worker process error:'), error.message)
        this.showTroubleshooting()
      }
    }
  }

  setupShutdownHandlers() {
    const gracefulShutdown = async (signal) => {
      console.log(chalk.yellow(`\n🛑 Received ${signal}. Shutting down gracefully...`))
      
      if (this.workerProcess) {
        console.log(chalk.dim('Stopping worker process...'))
        this.workerProcess.kill('SIGTERM')
        
        // Give it time to cleanup
        await sleep(2000)
        
        if (!this.workerProcess.killed) {
          console.log(chalk.dim('Force stopping worker...'))
          this.workerProcess.kill('SIGKILL')
        }
      }
      
      console.log(chalk.green('✅ Worker stopped successfully'))
      process.exit(0)
    }

    process.on('SIGINT', () => gracefulShutdown('SIGINT'))
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'))
    
    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      console.error(chalk.red('\n💥 Uncaught Exception:'), error.message)
      console.error(error.stack)
      process.exit(1)
    })
    
    process.on('unhandledRejection', (reason, promise) => {
      console.error(chalk.red('\n💥 Unhandled Rejection at:'), promise)
      console.error(chalk.red('Reason:'), reason)
      process.exit(1)
    })
  }

  showTroubleshooting() {
    console.log('\n' + boxen(
      chalk.red.bold('🛠️ TROUBLESHOOTING GUIDE\n\n') +
      chalk.white('Common issues and solutions:\n\n') +
      chalk.yellow('1. Service Connection Issues:\n') +
      chalk.dim('   • Check services: npm run test\n') +
      chalk.dim('   • Restart services: npm run start\n') +
      chalk.dim('   • Verify database: npm run database\n\n') +
      chalk.yellow('2. Database Issues:\n') +
      chalk.dim('   • Check PostgreSQL is running\n') +
      chalk.dim('   • Verify password is 123456\n') +
      chalk.dim('   • Run: psql -U postgres -d evidence_processing\n\n') +
      chalk.yellow('3. Queue Issues:\n') +
      chalk.dim('   • Check RabbitMQ: http://localhost:15672\n') +
      chalk.dim('   • Restart RabbitMQ: sc start RabbitMQ\n\n') +
      chalk.yellow('4. Permission Issues:\n') +
      chalk.dim('   • Run as Administrator\n') +
      chalk.dim('   • Check Windows Firewall settings\n\n') +
      chalk.yellow('5. Port Conflicts:\n') +
      chalk.dim('   • Check: netstat -ano | findstr ":5432"\n') +
      chalk.dim('   • Stop conflicting services'),
      {
        padding: 1,
        borderStyle: 'round',
        borderColor: 'red'
      }
    ))
  }
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const manager = new SmartWorkerManager()
  await manager.start()
}

export default SmartWorkerManager
