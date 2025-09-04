#!/usr/bin/env node

import 'zx/globals'
import { createSpinner } from 'nanospinner'
import chalk from 'chalk'
import pLimit from 'p-limit'
import boxen from 'boxen'
import gradient from 'gradient-string'

// Configure zx
$.verbose = false

const limit = pLimit(5) // Concurrent test limit

class SmartSystemTester {
  constructor() {
    this.testResults = {}
    this.services = {}
    this.loadServiceInfo()
  }

  async loadServiceInfo() {
    try {
      const resultsFile = await fs.readFile('detection-results.json', 'utf8')
      const results = JSON.parse(resultsFile)
      this.services = results.services
    } catch (error) {
      console.log(chalk.yellow('⚠️ No detection results found, will test all services'))
    }
  }

  async runAllTests() {
    console.log(gradient.pastel.multiline(`
    ╔══════════════════════════════════════════════════╗
    ║       🧪 SMART SYSTEM TESTING                   ║
    ║     Evidence Processing System Verification     ║
    ╚══════════════════════════════════════════════════╝
    `))

    try {
      // Run detection if not already done
      await this.runServiceDetection()
      
      // Test service connectivity
      await this.testServiceConnectivity()
      
      // Test application components
      await this.testApplicationComponents()
      
      // Run worker health check
      await this.testWorkerHealth()
      
      // Generate comprehensive report
      this.generateTestReport()
      
    } catch (error) {
      console.error(chalk.red('❌ System testing failed:'), error.message)
      process.exit(1)
    }
  }

  async runServiceDetection() {
    const spinner = createSpinner('🔍 Running smart service detection...').start()
    
    try {
      const { default: SmartServiceDetector } = await import('./detect-services.mjs')
      const detector = new SmartServiceDetector()
      this.services = await detector.detectAll()
      
      spinner.success({ text: '✅ Service detection complete!' })
      
    } catch (error) {
      spinner.error({ text: '❌ Service detection failed!' })
      throw error
    }
  }

  async testServiceConnectivity() {
    console.log('\n' + chalk.bold('🌐 TESTING SERVICE CONNECTIVITY'))
    console.log('═'.repeat(50))
    
    const tests = [
      { name: 'postgresql', fn: () => this.testPostgreSQL() },
      { name: 'redis', fn: () => this.testRedis() },
      { name: 'rabbitmq', fn: () => this.testRabbitMQ() },
      { name: 'qdrant', fn: () => this.testQdrant() },
      { name: 'neo4j', fn: () => this.testNeo4j() },
      { name: 'minio', fn: () => this.testMinIO() },
      { name: 'ollama', fn: () => this.testOllama(), optional: true }
    ]
    
    const testPromises = tests.map(test => 
      limit(async () => {
        const spinner = createSpinner(`Testing ${test.name}...`).start()
        
        try {
          const result = await test.fn()
          this.testResults[test.name] = { success: true, ...result }
          spinner.success({ text: `✅ ${test.name}: ${result.status}` })
        } catch (error) {
          this.testResults[test.name] = { success: false, error: error.message }
          const icon = test.optional ? '⚠️' : '❌'
          spinner.error({ text: `${icon} ${test.name}: ${error.message}` })
        }
      })
    )
    
    await Promise.allSettled(testPromises)
  }

  async testPostgreSQL() {
    // Set password
    process.env.PGPASSWORD = '123456'
    
    // Test connection
    const version = await $`psql -U postgres -t -c "SELECT version();"`.quiet()
    
    // Test database exists
    const dbExists = await $`psql -U postgres -d evidence_processing -c "SELECT 1;" -q`.quiet()
    
    // Test tables exist
    const tables = await $`psql -U postgres -d evidence_processing -c "\\dt" -q`.quiet()
    
    if (!tables.stdout.includes('evidence_process')) {
      throw new Error('Evidence processing tables not found')
    }
    
    // Test basic operations
    await $`psql -U postgres -d evidence_processing -c "SELECT count(*) FROM queue_stats;" -q`.quiet()
    
    return {
      status: 'Connected with evidence_processing database',
      version: version.stdout.trim(),
      hasSchema: true
    }
  }

  async testRedis() {
    const ping = await $`redis-cli ping`.quiet()
    
    if (ping.stdout.trim() !== 'PONG') {
      throw new Error('Redis not responding to ping')
    }
    
    // Test basic operations
    await $`redis-cli set test_key test_value`.quiet()
    const getValue = await $`redis-cli get test_key`.quiet()
    await $`redis-cli del test_key`.quiet()
    
    if (getValue.stdout.trim() !== 'test_value') {
      throw new Error('Redis operations not working')
    }
    
    const info = await $`redis-cli info server`.quiet()
    const version = info.stdout.match(/redis_version:([^\r\n]+)/)?.[1] || 'unknown'
    
    return {
      status: 'Connected and operational',
      version: version.trim()
    }
  }

  async testRabbitMQ() {
    // Check if RabbitMQ is listening
    const netstat = await $`netstat -an`.quiet()
    
    if (!netstat.stdout.includes(':5672')) {
      throw new Error('RabbitMQ not listening on port 5672')
    }
    
    // Check management interface
    const managementAvailable = netstat.stdout.includes(':15672')
    
    try {
      // Try to get version (if rabbitmqctl is available)
      const version = await $`rabbitmqctl version`.quiet()
      return {
        status: 'Service running',
        version: version.stdout.trim(),
        managementUI: managementAvailable
      }
    } catch (error) {
      return {
        status: 'Service running (version unknown)',
        managementUI: managementAvailable
      }
    }
  }

  async testQdrant() {
    const response = await fetch('http://localhost:6333/collections')
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    const collections = await response.json()
    
    // Test creating a collection
    const testCollection = {
      vectors: {
        size: 384,
        distance: 'Cosine'
      }
    }
    
    await fetch('http://localhost:6333/collections/test_collection', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testCollection)
    })
    
    // Clean up test collection
    await fetch('http://localhost:6333/collections/test_collection', {
      method: 'DELETE'
    })
    
    return {
      status: 'API responding',
      collections: collections.result?.collections?.length || 0
    }
  }

  async testNeo4j() {
    const response = await fetch('http://localhost:7474/')
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    // Check if bolt port is available
    const netstat = await $`netstat -an`.quiet()
    const boltAvailable = netstat.stdout.includes(':7687')
    
    return {
      status: 'Browser interface responding',
      boltPort: boltAvailable ? 'Available' : 'Not available'
    }
  }

  async testMinIO() {
    const healthResponse = await fetch('http://localhost:9000/minio/health/live')
    
    if (!healthResponse.ok) {
      throw new Error(`Health check failed: ${healthResponse.status}`)
    }
    
    // Check console
    const consoleResponse = await fetch('http://localhost:9001/')
    const consoleAvailable = consoleResponse.ok
    
    return {
      status: 'Health endpoint responding',
      console: consoleAvailable ? 'Available' : 'Not available'
    }
  }

  async testOllama() {
    const versionResponse = await fetch('http://localhost:11434/api/version')
    
    if (!versionResponse.ok) {
      throw new Error(`API not responding: ${versionResponse.status}`)
    }
    
    const version = await versionResponse.json()
    
    // Check for models
    const modelsResponse = await fetch('http://localhost:11434/api/tags')
    const models = await modelsResponse.json()
    
    const embeddingModel = models.models?.find(m => 
      m.name.includes('nomic-embed') || m.name.includes('embed')
    )
    
    return {
      status: 'API responding',
      version: version.version || 'unknown',
      embeddingModel: embeddingModel ? 'Available' : 'Not installed'
    }
  }

  async testApplicationComponents() {
    console.log('\n' + chalk.bold('📦 TESTING APPLICATION COMPONENTS'))
    console.log('═'.repeat(50))
    
    const tests = [
      { name: 'Worker Dependencies', fn: () => this.testWorkerDependencies() },
      { name: 'Frontend Dependencies', fn: () => this.testFrontendDependencies() },
      { name: 'Environment Configuration', fn: () => this.testEnvironmentConfig() },
      { name: 'Migration Files', fn: () => this.testMigrationFiles() }
    ]
    
    for (const test of tests) {
      const spinner = createSpinner(`Testing ${test.name}...`).start()
      
      try {
        const result = await test.fn()
        this.testResults[test.name.toLowerCase().replace(/\s+/g, '_')] = { success: true, ...result }
        spinner.success({ text: `✅ ${test.name}: ${result.status}` })
      } catch (error) {
        this.testResults[test.name.toLowerCase().replace(/\s+/g, '_')] = { success: false, error: error.message }
        spinner.error({ text: `❌ ${test.name}: ${error.message}` })
      }
    }
  }

  async testWorkerDependencies() {
    const workersDir = path.join(process.cwd(), 'workers')
    const nodeModulesExists = await this.fileExists(path.join(workersDir, 'node_modules'))
    
    if (!nodeModulesExists) {
      throw new Error('Worker dependencies not installed')
    }
    
    // Check package.json
    const packageJsonPath = path.join(workersDir, 'package.json')
    const packageJsonExists = await this.fileExists(packageJsonPath)
    
    if (packageJsonExists) {
      const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf8'))
      const depCount = Object.keys(packageJson.dependencies || {}).length
      
      return {
        status: `Dependencies installed (${depCount} packages)`,
        packageJson: true
      }
    }
    
    return {
      status: 'Dependencies installed',
      packageJson: false
    }
  }

  async testFrontendDependencies() {
    const frontendDir = path.join(process.cwd(), 'sveltekit-frontend')
    const nodeModulesExists = await this.fileExists(path.join(frontendDir, 'node_modules'))
    
    if (!nodeModulesExists) {
      throw new Error('Frontend dependencies not installed')
    }
    
    // Check if required packages exist
    const requiredPackages = ['uuid', 'amqplib', 'ioredis', 'ws']
    const missingPackages = []
    
    for (const pkg of requiredPackages) {
      const pkgPath = path.join(frontendDir, 'node_modules', pkg)
      if (!(await this.fileExists(pkgPath))) {
        missingPackages.push(pkg)
      }
    }
    
    if (missingPackages.length > 0) {
      throw new Error(`Missing packages: ${missingPackages.join(', ')}`)
    }
    
    return {
      status: 'All required packages installed',
      packages: requiredPackages.length
    }
  }

  async testEnvironmentConfig() {
    const envExists = await this.fileExists('.env')
    
    if (!envExists) {
      throw new Error('.env file not found')
    }
    
    const envContent = await fs.readFile('.env', 'utf8')
    const requiredVars = [
      'DATABASE_URL',
      'REDIS_URL',
      'RABBITMQ_URL',
      'QDRANT_URL',
      'NEO4J_URL',
      'MINIO_ENDPOINT'
    ]
    
    const missingVars = requiredVars.filter(varName => !envContent.includes(varName))
    
    if (missingVars.length > 0) {
      throw new Error(`Missing environment variables: ${missingVars.join(', ')}`)
    }
    
    return {
      status: 'All required variables present',
      variables: requiredVars.length
    }
  }

  async testMigrationFiles() {
    const migrationDir = path.join(process.cwd(), 'migrations')
    const migrationExists = await this.fileExists(migrationDir)
    
    if (!migrationExists) {
      throw new Error('Migration directory not found')
    }
    
    const schemaFile = path.join(migrationDir, 'create_evidence_processing_schema.sql')
    const schemaExists = await this.fileExists(schemaFile)
    
    if (!schemaExists) {
      throw new Error('Schema migration file not found')
    }
    
    // Check file size (should be substantial)
    const stats = await fs.stat(schemaFile)
    
    if (stats.size < 1000) {
      throw new Error('Migration file appears to be empty or incomplete')
    }
    
    return {
      status: 'Migration files present and valid',
      size: `${Math.round(stats.size / 1024)}KB`
    }
  }

  async testWorkerHealth() {
    console.log('\n' + chalk.bold('🏥 TESTING WORKER HEALTH'))
    console.log('═'.repeat(30))
    
    const spinner = createSpinner('Running worker health check...').start()
    
    try {
      const workersDir = path.join(process.cwd(), 'workers')
      const healthCheckFile = path.join(workersDir, 'health-check.js')
      
      if (!(await this.fileExists(healthCheckFile))) {
        throw new Error('Health check script not found')
      }
      
      // Run health check
      cd(workersDir)
      const result = await $`node health-check.js`.quiet()
      cd('..')
      
      this.testResults.worker_health = { success: true, status: 'Health check passed' }
      spinner.success({ text: '✅ Worker health check passed!' })
      
    } catch (error) {
      this.testResults.worker_health = { success: false, error: error.message }
      spinner.error({ text: `❌ Worker health check failed: ${error.message}` })
    }
  }

  generateTestReport() {
    console.log('\n' + chalk.bold('📊 COMPREHENSIVE TEST REPORT'))
    console.log('═'.repeat(60))
    
    const categories = {
      'Core Services': ['postgresql', 'redis', 'rabbitmq'],
      'AI & Storage': ['qdrant', 'neo4j', 'minio'],
      'Optional Services': ['ollama'],
      'Application': ['worker_dependencies', 'frontend_dependencies', 'environment_configuration', 'migration_files', 'worker_health']
    }
    
    let totalTests = 0
    let passedTests = 0
    
    for (const [category, tests] of Object.entries(categories)) {
      console.log(`\n${chalk.bold.blue(category)}:`)
      
      for (const test of tests) {
        totalTests++
        const result = this.testResults[test]
        
        if (result) {
          const icon = result.success ? '✅' : '❌'
          const status = result.success ? 'PASS' : 'FAIL'
          const testName = test.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
          
          console.log(`  ${icon} ${testName.padEnd(25)} ${status}`)
          
          if (result.success) {
            passedTests++
            if (result.status) {
              console.log(chalk.dim(`      ${result.status}`))
            }
          } else {
            console.log(chalk.red(`      Error: ${result.error}`))
          }
        } else {
          console.log(`  ⚪ ${test.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()).padEnd(25)} NOT TESTED`)
        }
      }
    }
    
    // Generate summary
    const successRate = Math.round((passedTests / totalTests) * 100)
    
    console.log('\n' + chalk.bold('🎯 TEST SUMMARY'))
    console.log('═'.repeat(30))
    console.log(`Total Tests: ${totalTests}`)
    console.log(`Passed: ${chalk.green(passedTests)}`)
    console.log(`Failed: ${chalk.red(totalTests - passedTests)}`)
    console.log(`Success Rate: ${successRate >= 80 ? chalk.green(successRate + '%') : chalk.red(successRate + '%')}`)
    
    // Show next steps
    this.showNextSteps(successRate)
  }

  showNextSteps(successRate) {
    if (successRate >= 90) {
      const message = boxen(
        chalk.bold.green('🎉 SYSTEM READY!\n\n') +
        chalk.white('Your Evidence Processing System is fully operational.\n\n') +
        chalk.bold('🚀 Next Steps:\n') +
        chalk.dim('  • npm run worker    # Start evidence processing\n') +
        chalk.dim('  • Upload evidence files to test the pipeline\n') +
        chalk.dim('  • Access web interfaces for monitoring\n\n') +
        chalk.cyan('🌐 Web Interfaces:\n') +
        chalk.dim('  • RabbitMQ: http://localhost:15672 (guest/guest)\n') +
        chalk.dim('  • Neo4j: http://localhost:7474 (neo4j/neo4j)\n') +
        chalk.dim('  • MinIO: http://localhost:9001 (evidence/evidence123)\n') +
        chalk.dim('  • Qdrant: http://localhost:6333/dashboard'),
        {
          padding: 1,
          margin: 1,
          borderStyle: 'double',
          borderColor: 'green'
        }
      )
      
      console.log('\n' + message)
      
    } else if (successRate >= 70) {
      console.log('\n' + boxen(
        chalk.yellow.bold('⚠️ PARTIAL SUCCESS\n\n') +
        chalk.white('Most components are working, but some issues detected.\n\n') +
        chalk.bold('🔧 Recommended Actions:\n') +
        chalk.dim('  • Review failed tests above\n') +
        chalk.dim('  • Check service logs for errors\n') +
        chalk.dim('  • Restart services: npm run start\n') +
        chalk.dim('  • Verify configurations in .env file'),
        {
          padding: 1,
          borderStyle: 'round',
          borderColor: 'yellow'
        }
      ))
      
    } else {
      console.log('\n' + boxen(
        chalk.red.bold('❌ SYSTEM ISSUES DETECTED\n\n') +
        chalk.white('Multiple components have failed tests.\n\n') +
        chalk.bold('🛠️ Required Actions:\n') +
        chalk.dim('  • Run npm run setup to reconfigure\n') +
        chalk.dim('  • Check PostgreSQL password is 123456\n') +
        chalk.dim('  • Verify all services are installed\n') +
        chalk.dim('  • Review error messages above\n') +
        chalk.dim('  • Contact support if issues persist'),
        {
          padding: 1,
          borderStyle: 'round',
          borderColor: 'red'
        }
      ))
    }
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
  const tester = new SmartSystemTester()
  await tester.runAllTests()
}

export default SmartSystemTester
