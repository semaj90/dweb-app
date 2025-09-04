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

const CONFIG = {
  postgresql: { password: '123456', port: 5432 },
  redis: { port: 6379 },
  rabbitmq: { port: 5672, mgmtPort: 15672 },
  qdrant: { port: 6333 },
  neo4j: { port: 7474, boltPort: 7687 },
  minio: { port: 9000, consolePort: 9001 },
  ollama: { port: 11434 }
}

class SmartServiceDetector {
  constructor() {
    this.services = {
      postgresql: { installed: false, running: false, version: null },
      redis: { installed: false, running: false, version: null },
      rabbitmq: { installed: false, running: false, version: null },
      qdrant: { installed: false, running: false, version: null, portable: true },
      neo4j: { installed: false, running: false, version: null, portable: true },
      minio: { installed: false, running: false, version: null, portable: true },
      ollama: { installed: false, running: false, version: null, portable: true }
    }
    
    this.limit = pLimit(3) // Concurrent detection limit
  }

  async detectAll() {
    console.log(gradient.pastel.multiline(`
    ╔══════════════════════════════════════╗
    ║     🔍 SMART SERVICE DETECTION       ║
    ║     Evidence Processing System       ║
    ╚══════════════════════════════════════╝
    `))

    const spinner = createSpinner('🔍 Detecting services...').start()

    try {
      // Run detections concurrently with limit
      const detectionPromises = [
        this.limit(() => this.detectPostgreSQL()),
        this.limit(() => this.detectRedis()),
        this.limit(() => this.detectRabbitMQ()),
        this.limit(() => this.detectPortableServices())
      ]

      await Promise.allSettled(detectionPromises)
      await this.checkRunningPorts()
      
      spinner.success({ text: '✅ Service detection complete!' })
      this.generateReport()
      
      return this.services
    } catch (error) {
      spinner.error({ text: `❌ Detection failed: ${error.message}` })
      throw error
    }
  }

  async detectPostgreSQL() {
    const service = 'postgresql'
    
    try {
      // Check if PostgreSQL is installed
      const version = await $`pg_config --version`.quiet()
      this.services[service].installed = true
      this.services[service].version = version.stdout.trim()
      
      // Test connection with password 123456
      process.env.PGPASSWORD = CONFIG.postgresql.password
      await $`psql -U postgres -c "SELECT 1" -q`.quiet()
      this.services[service].running = true
      
    } catch (error) {
      // PostgreSQL not found or connection failed
    }
  }

  async detectRedis() {
    const service = 'redis'
    
    try {
      // Check Redis CLI
      const version = await $`redis-cli --version`.quiet()
      this.services[service].installed = true
      this.services[service].version = version.stdout.trim()
      
      // Test Redis connection
      const ping = await $`redis-cli ping`.quiet()
      if (ping.stdout.trim() === 'PONG') {
        this.services[service].running = true
      } else {
        // Try to start Redis service
        try {
          await $`sc start Redis`.quiet()
          await sleep(3000)
          const retryPing = await $`redis-cli ping`.quiet()
          if (retryPing.stdout.trim() === 'PONG') {
            this.services[service].running = true
          }
        } catch (startError) {
          // Service start failed
        }
      }
    } catch (error) {
      // Redis not found
    }
  }

  async detectRabbitMQ() {
    const service = 'rabbitmq'
    
    try {
      // Check RabbitMQ
      const version = await $`rabbitmqctl version`.quiet()
      this.services[service].installed = true
      this.services[service].version = version.stdout.trim()
      
      // Check service status
      const serviceStatus = await $`sc query RabbitMQ`.quiet()
      if (serviceStatus.stdout.includes('RUNNING')) {
        this.services[service].running = true
      } else {
        // Try to start service
        try {
          await $`sc start RabbitMQ`.quiet()
          await sleep(5000)
          this.services[service].running = true
        } catch (startError) {
          // Service start failed
        }
      }
    } catch (error) {
      // RabbitMQ not found
    }
  }

  async detectPortableServices() {
    const portableServices = ['qdrant', 'neo4j', 'minio', 'ollama']
    
    const checkPromises = portableServices.map(service => 
      this.limit(async () => {
        let executablePath
        
        switch (service) {
          case 'qdrant':
            executablePath = path.join('services', 'qdrant.exe')
            break
          case 'neo4j':
            executablePath = path.join('services', 'neo4j', 'bin', 'neo4j.bat')
            break
          case 'minio':
            executablePath = path.join('services', 'minio.exe')
            break
          case 'ollama':
            executablePath = path.join('services', 'ollama.exe')
            break
        }
        
        try {
          await fs.access(executablePath)
          this.services[service].installed = true
          
          // Try to get version for some services
          if (service === 'minio') {
            try {
              const version = await $`"${executablePath}" --version`.quiet()
              this.services[service].version = version.stdout.trim()
            } catch (versionError) {
              this.services[service].version = 'Unknown'
            }
          }
        } catch (error) {
          // File doesn't exist
        }
      })
    )
    
    await Promise.allSettled(checkPromises)
  }

  async checkRunningPorts() {
    try {
      const netstat = await $`netstat -an`.quiet()
      const lines = netstat.stdout.split('\n')
      
      const portMapping = {
        [CONFIG.postgresql.port]: 'postgresql',
        [CONFIG.redis.port]: 'redis',
        [CONFIG.rabbitmq.port]: 'rabbitmq',
        [CONFIG.qdrant.port]: 'qdrant',
        [CONFIG.neo4j.port]: 'neo4j',
        [CONFIG.minio.port]: 'minio',
        [CONFIG.ollama.port]: 'ollama'
      }
      
      for (const line of lines) {
        if (line.includes('LISTENING')) {
          for (const [port, serviceName] of Object.entries(portMapping)) {
            if (line.includes(`:${port} `)) {
              this.services[serviceName].running = true
            }
          }
        }
      }
    } catch (error) {
      console.log(chalk.yellow('⚠️ Could not check ports via netstat'))
    }
  }

  generateReport() {
    console.log('\n' + chalk.bold('📋 SERVICE DETECTION REPORT'))
    console.log('═'.repeat(50))
    
    for (const [serviceName, info] of Object.entries(this.services)) {
      const status = info.running ? '🟢 RUNNING' : 
                    info.installed ? '🟡 INSTALLED' : '🔴 NOT FOUND'
      
      const serviceTitle = serviceName.toUpperCase().padEnd(12)
      console.log(`${serviceTitle} ${status}`)
      
      if (info.version) {
        console.log(`             Version: ${chalk.dim(info.version)}`)
      }
      
      const type = info.portable ? 'Portable' : 'System'
      console.log(`             Type: ${chalk.dim(type)}`)
    }
    
    this.generateStrategy()
  }

  generateStrategy() {
    console.log('\n' + chalk.bold('🚀 STARTUP STRATEGY'))
    console.log('═'.repeat(30))
    
    const systemServices = []
    const portableServices = []
    const missingServices = []
    
    for (const [serviceName, info] of Object.entries(this.services)) {
      if (info.running) {
        console.log(`✅ ${serviceName}: Already running`)
      } else if (info.installed && !info.portable) {
        systemServices.push(serviceName)
      } else if (info.installed && info.portable) {
        portableServices.push(serviceName)
      } else {
        missingServices.push(serviceName)
      }
    }
    
    if (systemServices.length > 0) {
      console.log(`\n🔧 System services to start: ${chalk.cyan(systemServices.join(', '))}`)
    }
    
    if (portableServices.length > 0) {
      console.log(`📦 Portable services to start: ${chalk.green(portableServices.join(', '))}`)
    }
    
    if (missingServices.length > 0) {
      console.log(`❌ Missing services: ${chalk.red(missingServices.join(', '))}`)
    }
  }

  getConnectionStrings() {
    const connections = {}
    
    if (this.services.postgresql.running) {
      connections.database = `postgresql://postgres:${CONFIG.postgresql.password}@localhost:${CONFIG.postgresql.port}/evidence_processing`
    }
    
    if (this.services.redis.running) {
      connections.redis = `redis://localhost:${CONFIG.redis.port}`
    }
    
    if (this.services.rabbitmq.running) {
      connections.rabbitmq = `amqp://guest:guest@localhost:${CONFIG.rabbitmq.port}`
    }
    
    connections.qdrant = `http://localhost:${CONFIG.qdrant.port}`
    connections.neo4j = `bolt://localhost:${CONFIG.neo4j.boltPort}`
    connections.minio = `http://localhost:${CONFIG.minio.port}`
    connections.ollama = `http://localhost:${CONFIG.ollama.port}`
    
    return connections
  }
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const detector = new SmartServiceDetector()
    const services = await detector.detectAll()
    
    const connections = detector.getConnectionStrings()
    
    console.log('\n' + boxen(
      chalk.bold('🔗 CONNECTION STRINGS\n\n') +
      Object.entries(connections)
        .map(([service, url]) => `${chalk.cyan(service)}: ${chalk.dim(url)}`)
        .join('\n'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'round',
        borderColor: 'green'
      }
    ))
    
    // Export results for other scripts
    await fs.writeFile('detection-results.json', JSON.stringify({
      services,
      connections,
      timestamp: new Date().toISOString()
    }, null, 2))
    
    console.log(chalk.green('✅ Detection results saved to detection-results.json'))
    
  } catch (error) {
    console.error(chalk.red('❌ Detection failed:'), error.message)
    process.exit(1)
  }
}

export default SmartServiceDetector
