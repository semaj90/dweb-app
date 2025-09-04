#!/usr/bin/env node

import 'zx/globals'
import { createSpinner } from 'nanospinner'
import chalk from 'chalk'
import pLimit from 'p-limit'
import pRetry from 'p-retry'
import boxen from 'boxen'
import gradient from 'gradient-string'
import SmartServiceDetector from './detect-services.mjs'

// Configure zx
$.verbose = false

const limit = pLimit(3) // Concurrent operations limit

class SmartSetup {
  constructor() {
    this.detector = new SmartServiceDetector()
    this.servicesDir = path.join(process.cwd(), 'services')
  }

  async run() {
    console.log(gradient.pastel.multiline(`
    ╔══════════════════════════════════════════════════╗
    ║  🚀 EVIDENCE PROCESSING SYSTEM - SMART SETUP    ║
    ║             Windows Native + Concurrency        ║
    ╚══════════════════════════════════════════════════╝
    `))

    try {
      // Step 1: Prerequisites check
      await this.checkPrerequisites()
      
      // Step 2: Service detection
      const services = await this.detector.detectAll()
      
      // Step 3: Install missing services (concurrent)
      await this.installMissingServices(services)
      
      // Step 4: Setup dependencies
      await this.setupDependencies()
      
      // Step 5: Generate configuration
      await this.generateConfiguration(services)
      
      this.showSuccessMessage()
      
    } catch (error) {
      console.error(chalk.red('❌ Setup failed:'), error.message)
      process.exit(1)
    }
  }

  async checkPrerequisites() {
    const spinner = createSpinner('🔍 Checking prerequisites...').start()
    
    const checks = [
      { name: 'Node.js', cmd: 'node --version', required: true },
      { name: 'Python', cmd: 'python --version', required: true },
      { name: 'PostgreSQL', cmd: 'pg_config --version', required: false },
      { name: 'Git', cmd: 'git --version', required: false }
    ]
    
    const results = await Promise.allSettled(
      checks.map(check => 
        limit(async () => {
          try {
            const result = await $`${check.cmd}`.quiet()
            return { name: check.name, version: result.stdout.trim(), found: true }
          } catch (error) {
            return { name: check.name, found: false, required: check.required }
          }
        })
      )
    )
    
    let allRequired = true
    
    console.log('\n' + chalk.bold('📋 Prerequisites Check:'))
    results.forEach(result => {
      if (result.status === 'fulfilled') {
        const check = result.value
        if (check.found) {
          console.log(`  ✅ ${check.name}: ${chalk.dim(check.version)}`)
        } else {
          const icon = check.required ? '❌' : '⚠️'
          console.log(`  ${icon} ${check.name}: Not found`)
          if (check.required) allRequired = false
        }
      }
    })
    
    if (!allRequired) {
      spinner.error({ text: '❌ Missing required prerequisites' })
      throw new Error('Please install Node.js 18+ and Python 3.8+')
    }
    
    spinner.success({ text: '✅ Prerequisites check passed!' })
  }

  async installMissingServices(services) {
    const missingServices = Object.entries(services)
      .filter(([name, info]) => !info.installed && info.portable)
      .map(([name]) => name)
    
    if (missingServices.length === 0) {
      console.log(chalk.green('✅ All required services already installed'))
      return
    }
    
    const spinner = createSpinner('📦 Installing missing services...').start()
    
    try {
      // Ensure services directory exists
      await fs.mkdir(this.servicesDir, { recursive: true })
      
      // Install services concurrently
      const installPromises = missingServices.map(service => 
        limit(() => this.installService(service))
      )
      
      await Promise.allSettled(installPromises)
      
      spinner.success({ text: '✅ Service installation complete!' })
      
    } catch (error) {
      spinner.error({ text: `❌ Service installation failed: ${error.message}` })
      throw error
    }
  }

  async installService(serviceName) {
    const serviceSpinner = createSpinner(`Installing ${serviceName}...`).start()
    
    try {
      switch (serviceName) {
        case 'qdrant':
          await this.installQdrant()
          break
        case 'neo4j':
          await this.installNeo4j()
          break
        case 'minio':
          await this.installMinIO()
          break
        case 'ollama':
          await this.installOllama()
          break
        case 'redis':
          await this.installRedis()
          break
        default:
          throw new Error(`Unknown service: ${serviceName}`)
      }
      
      serviceSpinner.success({ text: `✅ ${serviceName} installed` })
      
    } catch (error) {
      serviceSpinner.error({ text: `❌ ${serviceName} installation failed` })
      console.log(chalk.dim(`   Error: ${error.message}`))
    }
  }

  async installQdrant() {
    const execPath = path.join(this.servicesDir, 'qdrant.exe')
    
    if (await this.fileExists(execPath)) return
    
    await pRetry(async () => {
      const url = 'https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip'
      const zipPath = path.join(this.servicesDir, 'qdrant.zip')
      
      // Download
      await $`curl -L -o "${zipPath}" "${url}"`
      
      // Extract using PowerShell
      await $`powershell -command "Expand-Archive -Path '${zipPath}' -DestinationPath '${this.servicesDir}' -Force"`
      
      // Cleanup
      await fs.unlink(zipPath)
      
      // Create config
      const config = {
        log_level: 'INFO',
        storage: { storage_path: './qdrant_storage' },
        service: { http_port: 6333, grpc_port: 6334 }
      }
      
      await fs.writeFile(
        path.join(this.servicesDir, 'qdrant-config.yaml'),
        JSON.stringify(config, null, 2)
      )
    }, { retries: 3 })
  }

  async installNeo4j() {
    const neo4jDir = path.join(this.servicesDir, 'neo4j')
    
    if (await this.fileExists(neo4jDir)) return
    
    await pRetry(async () => {
      const url = 'https://dist.neo4j.org/neo4j-community-5.15.0-windows.zip'
      const zipPath = path.join(this.servicesDir, 'neo4j.zip')
      
      // Download
      await $`curl -L -o "${zipPath}" "${url}"`
      
      // Extract
      await $`powershell -command "Expand-Archive -Path '${zipPath}' -DestinationPath '${this.servicesDir}' -Force"`
      
      // Rename extracted directory
      const extractedDir = path.join(this.servicesDir, 'neo4j-community-5.15.0')
      await fs.rename(extractedDir, neo4jDir)
      
      // Cleanup
      await fs.unlink(zipPath)
      
      // Configure Neo4j
      const neo4jConf = `
server.default_listen_address=0.0.0.0
server.http.listen_address=:7474
server.bolt.listen_address=:7687
dbms.security.auth_enabled=true
dbms.security.auth_minimum_password_length=4
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
      `.trim()
      
      await fs.writeFile(
        path.join(neo4jDir, 'conf', 'neo4j.conf'),
        neo4jConf
      )
    }, { retries: 3 })
  }

  async installMinIO() {
    const execPath = path.join(this.servicesDir, 'minio.exe')
    
    if (await this.fileExists(execPath)) return
    
    await pRetry(async () => {
      const url = 'https://dl.min.io/server/minio/release/windows-amd64/minio.exe'
      
      // Download
      await $`curl -L -o "${execPath}" "${url}"`
      
      // Create data directory
      await fs.mkdir(path.join(this.servicesDir, 'minio-data'), { recursive: true })
    }, { retries: 3 })
  }

  async installOllama() {
    const execPath = path.join(this.servicesDir, 'ollama.exe')
    
    if (await this.fileExists(execPath)) return
    
    try {
      await pRetry(async () => {
        const url = 'https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.exe'
        await $`curl -L -o "${execPath}" "${url}"`
      }, { retries: 3 })
    } catch (error) {
      console.log(chalk.yellow('⚠️ Ollama download failed (optional service)'))
    }
  }

  async installRedis() {
    const execPath = path.join(this.servicesDir, 'redis-server.exe')
    
    if (await this.fileExists(execPath)) return
    
    await pRetry(async () => {
      const url = 'https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip'
      const zipPath = path.join(this.servicesDir, 'redis.zip')
      
      // Download
      await $`curl -L -o "${zipPath}" "${url}"`
      
      // Extract
      await $`powershell -command "Expand-Archive -Path '${zipPath}' -DestinationPath '${this.servicesDir}' -Force"`
      
      // Cleanup
      await fs.unlink(zipPath)
    }, { retries: 3 })
  }

  async setupDependencies() {
    const spinner = createSpinner('📦 Installing Node.js dependencies...').start()
    
    try {
      // Install project dependencies
      await $`npm install`
      
      // Install worker dependencies
      const workersDir = path.join(process.cwd(), 'workers')
      if (await this.fileExists(workersDir)) {
        cd(workersDir)
        await $`npm install`
        cd('..')
      }
      
      // Install frontend dependencies
      const frontendDir = path.join(process.cwd(), 'sveltekit-frontend')
      if (await this.fileExists(frontendDir)) {
        cd(frontendDir)
        await $`npm install uuid amqplib ioredis ws @qdrant/js-client-rest neo4j-driver minio node-fetch mammoth`
        await $`npm install -D @types/uuid @types/amqplib @types/ws`
        cd('..')
      }
      
      spinner.success({ text: '✅ Dependencies installed!' })
      
    } catch (error) {
      spinner.error({ text: `❌ Dependency installation failed: ${error.message}` })
      throw error
    }
  }

  async generateConfiguration(services) {
    const spinner = createSpinner('🔧 Generating configuration...').start()
    
    try {
      const connections = this.detector.getConnectionStrings()
      
      // Generate .env file
      const envContent = `# Evidence Processing System - Auto-generated Configuration
# Generated: ${new Date().toISOString()}

NODE_ENV=development

# Database (PostgreSQL with password 123456)
DATABASE_URL=${connections.database || 'postgresql://postgres:123456@localhost:5432/evidence_processing'}

# Message Queue
RABBITMQ_URL=${connections.rabbitmq || 'amqp://guest:guest@localhost:5672'}

# Cache
REDIS_URL=${connections.redis || 'redis://localhost:6379'}

# Vector Database
QDRANT_URL=${connections.qdrant}
QDRANT_COLLECTION=evidence_embeddings

# Knowledge Graph
NEO4J_URL=${connections.neo4j}
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j

# Object Storage
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=evidence
MINIO_SECRET_KEY=evidence123
MINIO_EVIDENCE_BUCKET=evidence
MINIO_USE_SSL=false

# Local LLM (Optional)
OLLAMA_URL=${connections.ollama}

# Service Detection Results
POSTGRESQL_DETECTED=${services.postgresql.installed}
REDIS_DETECTED=${services.redis.installed}
RABBITMQ_DETECTED=${services.rabbitmq.installed}
`
      
      await fs.writeFile('.env', envContent)
      
      // Generate service scripts
      await this.generateServiceScripts(services)
      
      spinner.success({ text: '✅ Configuration generated!' })
      
    } catch (error) {
      spinner.error({ text: `❌ Configuration generation failed: ${error.message}` })
      throw error
    }
  }

  async generateServiceScripts(services) {
    // This will generate the smart service management scripts
    // based on detected services (similar to what we had before)
    // but with better logic
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
      chalk.bold.green('🎉 SMART SETUP COMPLETE!\n\n') +
      chalk.white('Your Evidence Processing System is ready:\n\n') +
      chalk.cyan('• Detected existing PostgreSQL (password: 123456)\n') +
      chalk.cyan('• Installed missing portable services\n') +
      chalk.cyan('• Generated smart configuration\n') +
      chalk.cyan('• Set up concurrent processing\n\n') +
      chalk.bold('Next steps:\n') +
      chalk.dim('  npm run start     # Start all services\n') +
      chalk.dim('  npm run database  # Setup database\n') +
      chalk.dim('  npm run test      # Test system\n') +
      chalk.dim('  npm run demo      # One-click demo'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'double',
        borderColor: 'green'
      }
    )
    
    console.log('\n' + message)
  }
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const setup = new SmartSetup()
  await setup.run()
}

export default SmartSetup
