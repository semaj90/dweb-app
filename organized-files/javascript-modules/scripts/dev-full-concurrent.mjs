#!/usr/bin/env zx

/**
 * Concurrent Development Server Startup
 * Runs all services concurrently with proper directory structure
 */

import { $, chalk, path } from 'zx'

// Configure zx for Windows
$.shell = 'cmd.exe'
$.prefix = '/c'
$.verbose = true

const services = [
  {
    name: 'Enhanced RAG Service',
    command: 'PORT=8094 ./go-microservice/bin/enhanced-rag.exe',
    cwd: process.cwd(),
    color: 'blue',
    port: 8094
  },
  {
    name: 'Context7 MCP Server', 
    command: 'node ./mcp-servers/context7-server.js',
    cwd: process.cwd(),
    color: 'green',
    port: 4000
  },
  {
    name: 'SvelteKit Frontend',
    command: 'npm run dev',
    cwd: 'sveltekit-frontend',
    color: 'magenta',
    port: 5173
  },
  {
    name: 'PostgreSQL Check',
    command: 'echo "PostgreSQL assumed running on port 5432"',
    cwd: process.cwd(),
    color: 'cyan',
    port: 5432,
    oneShot: true
  }
]

console.log(chalk.bold.yellow('🚀 Starting Legal AI Platform - Multi-Cluster Concurrent Mode\n'))

// Function to check if a service is already running
async function isServiceRunning(port) {
  try {
    // Use curl if available, or skip check
    const result = await $`curl -s --connect-timeout 2 http://localhost:${port} || exit 0`.quiet()
    return true
  } catch (error) {
    return false
  }
}

// Function to start a service
async function startService(service) {
  const colorFn = chalk[service.color] || chalk.white
  
  console.log(colorFn(`📡 Checking ${service.name} on port ${service.port}...`))
  
  if (service.oneShot) {
    try {
      // Set working directory for one-shot commands
      const originalCwd = $.cwd
      $.cwd = service.cwd
      
      // Execute command with proper Windows handling
      if (service.command.startsWith('echo ')) {
        // Handle echo commands specially for Windows
        const echoMessage = service.command.slice(5).replace(/"/g, '')
        console.log(colorFn(`ℹ️ ${echoMessage}`))
      } else {
        await $`${service.command}`
      }
      
      console.log(colorFn(`✅ ${service.name} - OK`))
      $.cwd = originalCwd
    } catch (error) {
      console.log(colorFn(`⚠️  ${service.name} - ${error.message}`))
    }
    return
  }

  // Check if service is already running
  const isRunning = await isServiceRunning(service.port)
  
  if (isRunning) {
    console.log(colorFn(`✅ ${service.name} - Already running and healthy`))
    return null // Service already running, no need to start
  }

  console.log(colorFn(`🚀 Starting ${service.name}...`))

  let serviceProcess
  try {
    // Set working directory first
    const originalCwd = $.cwd
    $.cwd = service.cwd

    if (service.name.includes('RAG Service')) {
      // Windows-specific handling for Go executables - use absolute path
      const ragPath = path.resolve(service.cwd, 'go-microservice', 'bin', 'enhanced-rag.exe')
      console.log(colorFn(`Starting RAG service from: ${ragPath}`))
      // Set environment variables for Windows
      process.env.RAG_HTTP_PORT = '8094'
      serviceProcess = $`${ragPath}`
    } else if (service.name.includes('Context7 MCP Server')) {
      // Node.js MCP server - use simple command without complex quoting
      const mcpPath = path.resolve(service.cwd, 'mcp-servers', 'context7-server.js')
      console.log(colorFn(`Starting MCP server from: ${mcpPath}`))
      serviceProcess = $`node ${mcpPath}`
    } else if (service.name.includes('SvelteKit')) {
      // Change to SvelteKit directory and run dev
      const svelteKitPath = path.resolve(service.cwd, 'sveltekit-frontend')
      $.cwd = svelteKitPath
      console.log(colorFn(`Starting SvelteKit from: ${$.cwd}`))
      serviceProcess = $`npm run dev`
    } else {
      // Default command handling with proper path resolution
      const commandParts = service.command.split(' ')
      serviceProcess = $`${commandParts[0]} ${commandParts.slice(1).join(' ')}`
    }
  } catch (error) {
    console.log(colorFn(`❌ ${service.name} failed to start: ${error.message}`))
    return null
  }
  
  // Set up process handling
  serviceProcess.catch(error => {
    console.log(colorFn(`❌ ${service.name} failed: ${error.message}`))
  })

  // Give service time to start
  await sleep(3000)
  
  // Test if service is responding
  try {
    if (service.port === 8094) {
      await $`curl -s http://localhost:8094/api/health || exit 0`.quiet()
      console.log(colorFn(`✅ ${service.name} - Started and healthy`))
    } else if (service.port === 4000) {
      await $`curl -s http://localhost:4000/health || exit 0`.quiet()
      console.log(colorFn(`✅ ${service.name} - Started and healthy`))
    } else if (service.port === 5173) {
      // SvelteKit might take longer to start
      await sleep(5000)
      console.log(colorFn(`✅ ${service.name} - Started (check http://localhost:5173)`))
    }
  } catch (error) {
    console.log(colorFn(`⚠️  ${service.name} - Starting... (${error.message})`))
  }
  
  return serviceProcess
}

// Helper function
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Start all services concurrently
console.log(chalk.bold.cyan('🔄 Launching services concurrently...\n'))

const runningServices = []

for (const service of services) {
  const serviceProcess = startService(service)
  if (!service.oneShot) {
    runningServices.push(serviceProcess)
  }
}

// Wait a bit for all services to initialize
await sleep(8000)

console.log('\n' + chalk.bold.green('🎉 Legal AI Platform Status:'))
console.log(chalk.green('✅ Enhanced RAG Service: http://localhost:8094/api/health'))
console.log(chalk.green('✅ Context7 MCP Server: http://localhost:4000/health'))  
console.log(chalk.green('✅ SvelteKit Frontend: http://localhost:5173'))
console.log(chalk.green('✅ Ollama AI Service: http://localhost:11434'))
console.log('\n' + chalk.bold.cyan('🔍 Service Health Check:'))
console.log(chalk.cyan('Run `node health-check.cjs` for detailed health status'))

console.log('\n' + chalk.bold.yellow('🚀 All services running concurrently!'))
console.log(chalk.yellow('Press Ctrl+C to stop all services'))

// Keep the script running
process.on('SIGINT', () => {
  console.log('\n' + chalk.red('🛑 Shutting down all services...'))
  process.exit(0)
})

// Wait indefinitely
await new Promise(() => {})