#!/usr/bin/env node

import 'zx/globals'
import { createSpinner } from 'nanospinner'
import chalk from 'chalk'
import pLimit from 'p-limit'

// Configure zx
$.verbose = false

const limit = pLimit(3)

class SmartServiceStopper {
  constructor() {
    this.processes = new Map()
  }

  async stopAll() {
    console.log(chalk.bold('🛑 Stopping All Evidence Processing Services'))
    console.log('═'.repeat(50))

    try {
      // Stop processes concurrently
      const stopPromises = [
        limit(() => this.stopPortableServices()),
        limit(() => this.stopSystemServices()),
        limit(() => this.stopByPort())
      ]

      await Promise.allSettled(stopPromises)
      
      // Final verification
      await this.verifyShutdown()
      
      console.log(chalk.green('\n✅ All Evidence Processing services stopped!'))
      
    } catch (error) {
      console.error(chalk.red('❌ Error during shutdown:'), error.message)
    }
  }

  async stopPortableServices() {
    const spinner = createSpinner('📦 Stopping portable services...').start()
    
    const portableProcesses = [
      'qdrant.exe',
      'neo4j.exe', 
      'minio.exe',
      'ollama.exe',
      'redis-server.exe'
    ]

    for (const processName of portableProcesses) {
      try {
        await $`taskkill /F /IM ${processName} /T`.quiet()
        console.log(chalk.dim(`  Stopped ${processName}`))
      } catch (error) {
        // Process might not be running
      }
    }

    // Also stop Java processes that might be Neo4j
    try {
      await $`taskkill /F /IM java.exe /FI "WINDOWTITLE eq Neo4j*" /T`.quiet()
    } catch (error) {
      // Neo4j might not be running
    }

    spinner.success({ text: '✅ Portable services stopped' })
  }

  async stopSystemServices() {
    const spinner = createSpinner('🔧 Stopping system services...').start()
    
    const services = [
      { name: 'Redis', optional: true },
      { name: 'RabbitMQ', optional: true },
      // Don't stop PostgreSQL as it might be used by other applications
    ]

    for (const service of services) {
      try {
        const result = await $`sc stop ${service.name}`.quiet()
        if (result.exitCode === 0) {
          console.log(chalk.dim(`  Stopped ${service.name} service`))
        }
      } catch (error) {
        if (!service.optional) {
          console.log(chalk.yellow(`  Could not stop ${service.name} service`))
        }
      }
    }

    spinner.success({ text: '✅ System services processed' })
  }

  async stopByPort() {
    const spinner = createSpinner('🌐 Stopping services by port...').start()
    
    const ports = [6333, 7474, 9000, 11434] // Don't kill PostgreSQL, Redis, RabbitMQ ports
    
    for (const port of ports) {
      try {
        const netstatResult = await $`netstat -ano | findstr ":${port}"`.quiet()
        const lines = netstatResult.stdout.split('\n')
        
        for (const line of lines) {
          const parts = line.trim().split(/\s+/)
          if (parts.length >= 5 && parts[0] === 'TCP') {
            const pid = parts[4]
            if (pid && pid !== '0') {
              try {
                await $`taskkill /F /PID ${pid}`.quiet()
                console.log(chalk.dim(`  Stopped process on port ${port} (PID: ${pid})`))
              } catch (killError) {
                // Process might already be dead
              }
            }
          }
        }
      } catch (error) {
        // Port might not be in use
      }
    }

    spinner.success({ text: '✅ Port-based cleanup complete' })
  }

  async verifyShutdown() {
    const spinner = createSpinner('🔍 Verifying shutdown...').start()
    
    try {
      const serviceStatus = {
        postgresql: await this.checkPort(5432),
        redis: await this.checkPort(6379),
        rabbitmq: await this.checkPort(5672),
        qdrant: await this.checkPort(6333),
        neo4j: await this.checkPort(7474),
        minio: await this.checkPort(9000),
        ollama: await this.checkPort(11434)
      }

      console.log('\n' + chalk.bold('📊 Final Port Status:'))
      for (const [service, isRunning] of Object.entries(serviceStatus)) {
        const status = isRunning ? 
          (service === 'postgresql' ? chalk.yellow('Still running (shared service)') : chalk.red('Still running')) :
          chalk.green('Stopped')
        
        console.log(`  ${service.padEnd(12)} ${status}`)
      }

      const evidenceServicesRunning = Object.entries(serviceStatus)
        .filter(([service, running]) => running && service !== 'postgresql')
        .length

      if (evidenceServicesRunning === 0) {
        spinner.success({ text: '✅ All evidence processing services stopped' })
      } else {
        spinner.warn({ text: `⚠️ ${evidenceServicesRunning} services still running` })
      }

    } catch (error) {
      spinner.error({ text: '❌ Could not verify shutdown' })
    }
  }

  async checkPort(port) {
    try {
      const result = await $`netstat -an | findstr ":${port}"`.quiet()
      return result.stdout.includes('LISTENING')
    } catch (error) {
      return false
    }
  }
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const stopper = new SmartServiceStopper()
  await stopper.stopAll()
  
  console.log(chalk.bold('\n🔧 To restart everything:'))
  console.log(chalk.dim('  npm run start     # Start all services'))
  console.log(chalk.dim('  npm run demo      # Run complete demo'))
  console.log(chalk.dim('  npm run test      # Test system status'))
}

export default SmartServiceStopper
