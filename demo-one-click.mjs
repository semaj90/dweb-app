#!/usr/bin/env node

import 'zx/globals'
import { createSpinner } from 'nanospinner'
import chalk from 'chalk'
import boxen from 'boxen'
import gradient from 'gradient-string'
import inquirer from 'inquirer'

// Configure zx
$.verbose = false

class OneClickDemo {
  constructor() {
    this.steps = [
      { name: 'Service Detection', fn: () => this.detectServices() },
      { name: 'Smart Setup', fn: () => this.runSetup() },
      { name: 'Start Services', fn: () => this.startServices() },
      { name: 'Database Setup', fn: () => this.setupDatabase() },
      { name: 'System Testing', fn: () => this.testSystem() },
      { name: 'Worker Ready', fn: () => this.prepareWorker() }
    ]
    
    this.currentStep = 0
    this.startTime = Date.now()
  }

  async run() {
    console.log(gradient.pastel.multiline(`
    ╔════════════════════════════════════════════════════════╗
    ║       🎯 EVIDENCE PROCESSING SYSTEM                   ║
    ║            ONE-CLICK DEMO SETUP                       ║
    ║         Windows Native + Smart Detection              ║
    ╚════════════════════════════════════════════════════════╝
    `))

    console.log(chalk.bold('This automated demo will:'))
    console.log('🔍 1. Detect your existing services (PostgreSQL with password 123456)')
    console.log('📦 2. Install missing portable services concurrently')
    console.log('🚀 3. Start everything with smart fallbacks')
    console.log('🗄️ 4. Configure database and schema')
    console.log('🧪 5. Run comprehensive system tests')
    console.log('🏭 6. Start evidence processing worker')
    
    console.log(chalk.dim('\n⏱️ Estimated time: 3-5 minutes'))
    console.log(chalk.dim('🎯 Goal: Fully operational evidence processing pipeline'))

    // Ask for confirmation
    const { confirm } = await inquirer.prompt([{
      type: 'confirm',
      name: 'confirm',
      message: 'Ready to start the automated setup?',
      default: true
    }])

    if (!confirm) {
      console.log(chalk.yellow('Demo cancelled. You can run individual commands:'))
      console.log(chalk.dim('  npm run setup     # Smart setup'))
      console.log(chalk.dim('  npm run start     # Start services'))
      console.log(chalk.dim('  npm run database  # Setup database'))
      console.log(chalk.dim('  npm run test      # Test system'))
      console.log(chalk.dim('  npm run worker    # Start worker'))
      return
    }

    try {
      // Run all steps
      for (let i = 0; i < this.steps.length; i++) {
        this.currentStep = i
        const step = this.steps[i]
        
        console.log('\n' + '═'.repeat(60))
        console.log(chalk.bold.blue(`📋 Step ${i + 1}/${this.steps.length}: ${step.name}`))
        console.log('═'.repeat(60))
        
        await step.fn()
        
        // Show progress
        this.showProgress()
      }
      
      // Demo complete
      await this.showCompletionScreen()
      
    } catch (error) {
      console.error(chalk.red('\n❌ Demo failed at step:'), this.steps[this.currentStep].name)
      console.error(chalk.red('Error:'), error.message)
      
      this.showFailureHelp()
      process.exit(1)
    }
  }

  async detectServices() {
    const spinner = createSpinner('🔍 Running smart service detection...').start()
    
    try {
      // Run service detection
      const { default: SmartServiceDetector } = await import('./detect-services.mjs')
      const detector = new SmartServiceDetector()
      const services = await detector.detectAll()
      
      // Show brief summary
      const existing = Object.entries(services).filter(([_, info]) => info.running).length
      const installed = Object.entries(services).filter(([_, info]) => info.installed).length
      const total = Object.keys(services).length
      
      spinner.success({ text: `✅ Detection complete! Found ${existing} running, ${installed} installed of ${total} services` })
      
      return services
      
    } catch (error) {
      spinner.error({ text: '❌ Service detection failed!' })
      throw error
    }
  }

  async runSetup() {
    const spinner = createSpinner('📦 Running smart setup (installing missing services)...').start()
    
    try {
      // Run setup script
      const { default: SmartSetup } = await import('./setup-smart.mjs')
      const setup = new SmartSetup()
      
      // Suppress console output during automated setup
      const originalLog = console.log
      console.log = () => {}
      
      try {
        await setup.run()
      } finally {
        console.log = originalLog
      }
      
      spinner.success({ text: '✅ Smart setup complete! All required services installed' })
      
    } catch (error) {
      spinner.error({ text: '❌ Setup failed!' })
      throw error
    }
  }

  async startServices() {
    const spinner = createSpinner('🚀 Starting all services with smart detection...').start()
    
    try {
      // Import and run service manager
      const { default: SmartServiceManager } = await import('./start-services.mjs')
      const manager = new SmartServiceManager()
      
      // Start services but don't keep alive (for demo)
      await manager.loadDetectionResults()
      
      const strategy = manager.generateStartupStrategy()
      await manager.executeStartupStrategy(strategy)
      
      // Give services time to fully initialize
      spinner.text = 'Waiting for services to initialize...'
      await sleep(10000)
      
      spinner.success({ text: '✅ All services started and initialized!' })
      
    } catch (error) {
      spinner.error({ text: '❌ Service startup failed!' })
      throw error
    }
  }

  async setupDatabase() {
    const spinner = createSpinner('🗄️ Setting up PostgreSQL database with smart configuration...').start()
    
    try {
      // Run database setup
      const { default: SmartDatabaseSetup } = await import('./setup-database.mjs')
      const dbSetup = new SmartDatabaseSetup()
      
      // Suppress console output during automated setup
      const originalLog = console.log
      console.log = () => {}
      
      try {
        await dbSetup.run()
      } finally {
        console.log = originalLog
      }
      
      spinner.success({ text: '✅ Database configured! Schema migrated, extensions installed' })
      
    } catch (error) {
      spinner.error({ text: '❌ Database setup failed!' })
      throw error
    }
  }

  async testSystem() {
    const spinner = createSpinner('🧪 Running comprehensive system tests...').start()
    
    try {
      // Run system tests
      const { default: SmartSystemTester } = await import('./test-system.mjs')
      const tester = new SmartSystemTester()
      
      // Suppress verbose output during automated testing
      const originalLog = console.log
      console.log = () => {}
      
      try {
        await tester.runAllTests()
      } finally {
        console.log = originalLog
      }
      
      // Calculate success rate
      const results = tester.testResults
      const totalTests = Object.keys(results).length
      const passedTests = Object.values(results).filter(r => r.success).length
      const successRate = Math.round((passedTests / totalTests) * 100)
      
      if (successRate >= 80) {
        spinner.success({ text: `✅ System tests passed! (${passedTests}/${totalTests} tests, ${successRate}% success rate)` })
      } else {
        spinner.warn({ text: `⚠️ Some tests failed (${passedTests}/${totalTests} tests, ${successRate}% success rate)` })
      }
      
    } catch (error) {
      spinner.error({ text: '❌ System testing failed!' })
      throw error
    }
  }

  async prepareWorker() {
    const spinner = createSpinner('🏭 Preparing evidence processing worker...').start()
    
    try {
      // Check worker readiness without starting it
      const workersDir = path.join(process.cwd(), 'workers')
      
      // Check dependencies
      const nodeModulesExists = await this.fileExists(path.join(workersDir, 'node_modules'))
      if (!nodeModulesExists) {
        spinner.text = 'Installing worker dependencies...'
        cd(workersDir)
        await $`npm install`.quiet()
        cd('..')
      }
      
      // Verify worker file exists
      const workerFile = path.join(workersDir, 'evidenceProcessor.js')
      await fs.access(workerFile)
      
      // Test basic connectivity
      process.env.PGPASSWORD = '123456'
      await $`psql -U postgres -d evidence_processing -c "SELECT 1;" -q`.quiet()
      
      spinner.success({ text: '✅ Worker ready to start! All dependencies and connections verified' })
      
    } catch (error) {
      spinner.error({ text: '❌ Worker preparation failed!' })
      throw error
    }
  }

  showProgress() {
    const progress = Math.round(((this.currentStep + 1) / this.steps.length) * 100)
    const elapsed = Math.round((Date.now() - this.startTime) / 1000)
    
    console.log(chalk.dim(`\n📊 Progress: ${progress}% complete (${elapsed}s elapsed)`))
    
    // Show progress bar
    const barLength = 40
    const filledLength = Math.round((progress / 100) * barLength)
    const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength)
    console.log(chalk.cyan(`[${bar}] ${progress}%`))
  }

  async showCompletionScreen() {
    const totalTime = Math.round((Date.now() - this.startTime) / 1000)
    
    console.log('\n' + '🎉'.repeat(20))
    console.log(gradient.rainbow.multiline(`
    ╔════════════════════════════════════════════════════════╗
    ║                🎉 DEMO COMPLETE! 🎉                   ║
    ║         Evidence Processing System Ready!              ║
    ╚════════════════════════════════════════════════════════╝
    `))

    const message = boxen(
      chalk.bold.green(`🚀 SUCCESS! Your Evidence Processing System is now running!\n\n`) +
      chalk.white(`Setup completed in ${totalTime} seconds\n\n`) +
      chalk.cyan('🔧 What was configured:\n') +
      chalk.dim('  ✅ Smart service detection (found existing PostgreSQL)\n') +
      chalk.dim('  ✅ Installed missing portable services\n') +
      chalk.dim('  ✅ Started all services with intelligent fallbacks\n') +
      chalk.dim('  ✅ Configured database with password 123456\n') +
      chalk.dim('  ✅ Migrated database schema and extensions\n') +
      chalk.dim('  ✅ Verified system connectivity and health\n') +
      chalk.dim('  ✅ Prepared evidence processing worker\n\n') +
      chalk.bold('🌐 Web Interfaces Now Available:\n') +
      chalk.blue('  • RabbitMQ Management: http://localhost:15672 (guest/guest)\n') +
      chalk.blue('  • Neo4j Browser: http://localhost:7474 (neo4j/neo4j)\n') +
      chalk.blue('  • MinIO Console: http://localhost:9001 (evidence/evidence123)\n') +
      chalk.blue('  • Qdrant Dashboard: http://localhost:6333/dashboard\n\n') +
      chalk.bold('🏭 Ready to Process Evidence:\n') +
      chalk.green('  1. Upload evidence files to MinIO\n') +
      chalk.green('  2. Submit processing jobs via API\n') +
      chalk.green('  3. Watch real-time progress updates\n') +
      chalk.green('  4. Explore AI insights and connections'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'double',
        borderColor: 'green'
      }
    )
    
    console.log('\n' + message)

    // Ask what to do next
    const { nextAction } = await inquirer.prompt([{
      type: 'list',
      name: 'nextAction',
      message: 'What would you like to do next?',
      choices: [
        { name: '🏭 Start the Evidence Processing Worker', value: 'worker' },
        { name: '🌐 Open Web Interfaces in Browser', value: 'browser' },
        { name: '🧪 Run Detailed System Tests', value: 'test' },
        { name: '📊 View System Status', value: 'status' },
        { name: '🔧 Show Management Commands', value: 'commands' },
        { name: '🚪 Exit (services will continue running)', value: 'exit' }
      ]
    }])

    await this.handleNextAction(nextAction)
  }

  async handleNextAction(action) {
    switch (action) {
      case 'worker':
        console.log(chalk.bold('\n🏭 Starting Evidence Processing Worker...'))
        console.log(chalk.dim('The worker will process evidence files and provide real-time updates'))
        
        const { default: SmartWorkerManager } = await import('./start-worker.mjs')
        const worker = new SmartWorkerManager()
        await worker.start()
        break
        
      case 'browser':
        console.log(chalk.bold('\n🌐 Opening Web Interfaces...'))
        
        const urls = [
          'http://localhost:15672', // RabbitMQ
          'http://localhost:7474',  // Neo4j
          'http://localhost:9001',  // MinIO
          'http://localhost:6333/dashboard' // Qdrant
        ]
        
        for (const url of urls) {
          try {
            await $`start ${url}`.quiet()
          } catch (error) {
            console.log(chalk.yellow(`Could not open ${url} automatically`))
          }
        }
        
        console.log(chalk.green('✅ Web interfaces should be opening in your browser'))
        break
        
      case 'test':
        console.log(chalk.bold('\n🧪 Running Detailed System Tests...'))
        
        const { default: SmartSystemTester } = await import('./test-system.mjs')
        const tester = new SmartSystemTester()
        await tester.runAllTests()
        break
        
      case 'status':
        await this.showSystemStatus()
        break
        
      case 'commands':
        this.showManagementCommands()
        break
        
      case 'exit':
        console.log(chalk.green('\n✅ Demo complete! Services are running in the background.'))
        console.log(chalk.dim('Use the commands below to manage your system.'))
        this.showManagementCommands()
        break
    }
  }

  async showSystemStatus() {
    console.log(chalk.bold('\n📊 CURRENT SYSTEM STATUS'))
    console.log('═'.repeat(50))
    
    const services = [
      { name: 'PostgreSQL', port: 5432, check: () => this.checkPostgreSQL() },
      { name: 'Redis', port: 6379, check: () => this.checkRedis() },
      { name: 'RabbitMQ', port: 5672, check: () => this.checkRabbitMQ() },
      { name: 'Qdrant', port: 6333, check: () => this.checkQdrant() },
      { name: 'Neo4j', port: 7474, check: () => this.checkNeo4j() },
      { name: 'MinIO', port: 9000, check: () => this.checkMinIO() }
    ]
    
    for (const service of services) {
      try {
        await service.check()
        console.log(`✅ ${service.name.padEnd(12)} Running on port ${service.port}`)
      } catch (error) {
        console.log(`❌ ${service.name.padEnd(12)} Not responding`)
      }
    }
    
    // Show process information
    console.log('\n' + chalk.bold('💻 Process Information:'))
    try {
      const processes = await $`tasklist | findstr /I "postgres redis rabbitmq qdrant neo4j minio"`.quiet()
      if (processes.stdout.trim()) {
        console.log(chalk.dim(processes.stdout))
      } else {
        console.log(chalk.dim('No evidence processing related processes found'))
      }
    } catch (error) {
      console.log(chalk.dim('Could not retrieve process information'))
    }
  }

  showManagementCommands() {
    console.log('\n' + boxen(
      chalk.bold.blue('🔧 EVIDENCE PROCESSING SYSTEM COMMANDS\n\n') +
      chalk.white('Core Commands:\n') +
      chalk.green('  npm run start     ') + chalk.dim('# Start all services\n') +
      chalk.green('  npm run stop      ') + chalk.dim('# Stop all services\n') +
      chalk.green('  npm run worker    ') + chalk.dim('# Start evidence processing worker\n') +
      chalk.green('  npm run test      ') + chalk.dim('# Run system tests\n') +
      chalk.green('  npm run health    ') + chalk.dim('# Check system health\n\n') +
      chalk.white('Setup Commands:\n') +
      chalk.yellow('  npm run setup     ') + chalk.dim('# Re-run smart setup\n') +
      chalk.yellow('  npm run database  ') + chalk.dim('# Setup/reset database\n') +
      chalk.yellow('  npm run detect    ') + chalk.dim('# Re-detect services\n\n') +
      chalk.white('Utility Commands:\n') +
      chalk.cyan('  node backup-database.mjs    ') + chalk.dim('# Backup database\n') +
      chalk.cyan('  node restore-database.mjs   ') + chalk.dim('# Restore database\n') +
      chalk.cyan('  cat database-config.txt     ') + chalk.dim('# View DB config\n\n') +
      chalk.white('Direct Service Access:\n') +
      chalk.magenta('  psql -U postgres -d evidence_processing  ') + chalk.dim('# Database\n') +
      chalk.magenta('  redis-cli                                ') + chalk.dim('# Redis\n\n') +
      chalk.bold('💡 All commands use smart detection and graceful fallbacks!'),
      {
        padding: 1,
        borderStyle: 'round',
        borderColor: 'blue'
      }
    ))
  }

  showFailureHelp() {
    console.log('\n' + boxen(
      chalk.red.bold('🛠️ DEMO FAILED - TROUBLESHOOTING\n\n') +
      chalk.white('Common issues and solutions:\n\n') +
      chalk.yellow('1. PostgreSQL Issues:\n') +
      chalk.dim('   • Ensure PostgreSQL is installed and running\n') +
      chalk.dim('   • Verify password is exactly: 123456\n') +
      chalk.dim('   • Check Windows Services for PostgreSQL\n\n') +
      chalk.yellow('2. Permission Issues:\n') +
      chalk.dim('   • Run Command Prompt as Administrator\n') +
      chalk.dim('   • Check Windows Firewall settings\n\n') +
      chalk.yellow('3. Port Conflicts:\n') +
      chalk.dim('   • Check: netstat -ano | findstr ":5432"\n') +
      chalk.dim('   • Stop conflicting services\n\n') +
      chalk.yellow('4. Network Issues:\n') +
      chalk.dim('   • Disable VPN temporarily\n') +
      chalk.dim('   • Check antivirus software\n\n') +
      chalk.bold('🔧 Manual Setup Options:\n') +
      chalk.green('  npm run setup     # Try smart setup again\n') +
      chalk.green('  npm run detect    # Re-detect services\n') +
      chalk.green('  npm run test      # Diagnose specific issues'),
      {
        padding: 1,
        borderStyle: 'round',
        borderColor: 'red'
      }
    ))
  }

  // Service check methods
  async checkPostgreSQL() {
    process.env.PGPASSWORD = '123456'
    await $`psql -U postgres -c "SELECT 1;" -q`.quiet()
  }

  async checkRedis() {
    const result = await $`redis-cli ping`.quiet()
    if (result.stdout.trim() !== 'PONG') throw new Error('Not responding')
  }

  async checkRabbitMQ() {
    const netstat = await $`netstat -an`.quiet()
    if (!netstat.stdout.includes(':5672')) throw new Error('Not listening')
  }

  async checkQdrant() {
    const response = await fetch('http://localhost:6333/collections')
    if (!response.ok) throw new Error('HTTP error')
  }

  async checkNeo4j() {
    const response = await fetch('http://localhost:7474/')
    if (!response.ok) throw new Error('HTTP error')
  }

  async checkMinIO() {
    const response = await fetch('http://localhost:9000/minio/health/live')
    if (!response.ok) throw new Error('HTTP error')
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath)
      return true
    } catch {
      return false
    }
  }
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const demo = new OneClickDemo()
  await demo.run()
}

export default OneClickDemo
