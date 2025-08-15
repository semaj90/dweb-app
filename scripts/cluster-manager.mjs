#!/usr/bin/env node

/**
 * Node.js Cluster Manager with NATS Integration
 * Handles WebGPU frontend requests and distributes compute to Go microservices
 */

import cluster from 'cluster'
import express from 'express'
import { createServer } from 'http'
import { WebSocketServer } from 'ws'
import { connect } from 'nats'
import Redis from 'redis'
import { cpus } from 'os'

const PORT = process.env.PORT || 3000
const NATS_URL = process.env.NATS_URL || 'nats://localhost:4222'
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379'

if (cluster.isPrimary) {
  console.log(`🔧 Master ${process.pid} starting cluster manager`)
  
  const numWorkers = cpus().length - 1 // Leave one core for system
  console.log(`🚀 Starting ${numWorkers} worker processes`)

  // Fork workers
  for (let i = 0; i < numWorkers; i++) {
    const worker = cluster.fork()
    console.log(`👷 Worker ${worker.process.pid} started`)
  }

  // Handle worker deaths
  cluster.on('exit', (worker, code, signal) => {
    console.log(`💀 Worker ${worker.process.pid} died (${signal || code})`)
    console.log('🔄 Starting new worker...')
    cluster.fork()
  })

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('🛑 Master received SIGTERM, shutting down workers...')
    for (const id in cluster.workers) {
      cluster.workers[id].kill()
    }
  })

} else {
  // Worker process
  console.log(`👷 Worker ${process.pid} starting...`)
  
  const app = express()
  const server = createServer(app)
  const wss = new WebSocketServer({ server })

  // Middleware
  app.use(express.json({ limit: '50mb' }))
  app.use(express.raw({ type: 'application/octet-stream', limit: '100mb' }))

  // CORS for WebGPU frontend
  app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*')
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization')
    if (req.method === 'OPTIONS') {
      res.sendStatus(200)
    } else {
      next()
    }
  })

  // Initialize NATS and Redis connections
  let natsConnection, redisClient

  async function initializeConnections() {
    try {
      // NATS connection
      natsConnection = await connect({ servers: NATS_URL })
      console.log(`📡 Worker ${process.pid} connected to NATS`)

      // Redis connection
      redisClient = Redis.createClient({ url: REDIS_URL })
      await redisClient.connect()
      console.log(`🔴 Worker ${process.pid} connected to Redis`)

    } catch (error) {
      console.error(`❌ Worker ${process.pid} connection failed:`, error)
      process.exit(1)
    }
  }

  // WebGPU Vertex Buffer API
  app.post('/api/webgpu/vertex-cache', async (req, res) => {
    try {
      const { bufferData, bufferKey, operation } = req.body

      if (operation === 'store') {
        // Cache vertex buffer metadata in Redis
        const metadata = {
          size: bufferData.length,
          type: 'vertex_buffer',
          timestamp: Date.now(),
          worker: process.pid
        }
        
        await redisClient.setEx(`vertex:${bufferKey}`, 3600, JSON.stringify(metadata))
        
        // Publish to NATS for GPU services
        natsConnection.publish('gpu.vertex.cache', JSON.stringify({
          operation: 'store',
          key: bufferKey,
          data: bufferData,
          metadata
        }))

        res.json({ success: true, cached: true, key: bufferKey })
      
      } else if (operation === 'retrieve') {
        const metadata = await redisClient.get(`vertex:${bufferKey}`)
        
        if (metadata) {
          res.json({ 
            success: true, 
            found: true, 
            metadata: JSON.parse(metadata) 
          })
        } else {
          res.json({ success: true, found: false })
        }
      }

    } catch (error) {
      console.error('WebGPU vertex cache error:', error)
      res.status(500).json({ error: error.message })
    }
  })

  // Protobuf API Gateway
  app.post('/api/protobuf/:service', async (req, res) => {
    try {
      const { service } = req.params
      const protobufData = req.body

      // Route to appropriate Go microservice via NATS
      const subject = `legal.ai.${service}`
      const response = await natsConnection.request(subject, JSON.stringify(protobufData), { timeout: 30000 })
      
      // Parse response
      const result = JSON.parse(response.data.toString())
      
      // Cache in Redis if it's a successful query
      if (result.success && service === 'search') {
        const cacheKey = `search:${Buffer.from(JSON.stringify(protobufData)).toString('base64')}`
        await redisClient.setEx(cacheKey, 300, JSON.stringify(result)) // 5 min cache
      }

      res.json(result)

    } catch (error) {
      console.error(`Protobuf ${service} error:`, error)
      res.status(500).json({ error: error.message })
    }
  })

  // Semantic RAG with Caching
  app.post('/api/rag/semantic-search', async (req, res) => {
    try {
      const { query, filters, options = {} } = req.body
      
      // Check cache first
      const cacheKey = `rag:${Buffer.from(query + JSON.stringify(filters)).toString('base64')}`
      const cachedResult = await redisClient.get(cacheKey)
      
      if (cachedResult) {
        console.log(`🎯 Cache hit for query: ${query.substring(0, 50)}...`)
        return res.json({ 
          ...JSON.parse(cachedResult), 
          cached: true,
          worker: process.pid 
        })
      }

      // Forward to Enhanced RAG microservice via NATS
      const ragRequest = {
        query,
        filters,
        options: {
          ...options,
          worker_id: process.pid,
          timestamp: Date.now()
        }
      }

      console.log(`🔍 Processing RAG query via NATS: ${query.substring(0, 50)}...`)
      
      const response = await natsConnection.request('legal.ai.rag.search', JSON.stringify(ragRequest), { timeout: 45000 })
      const result = JSON.parse(response.data.toString())

      // Cache successful results
      if (result.success) {
        await redisClient.setEx(cacheKey, 300, JSON.stringify(result))
      }

      res.json({ ...result, worker: process.pid })

    } catch (error) {
      console.error('Semantic RAG error:', error)
      res.status(500).json({ error: error.message, worker: process.pid })
    }
  })

  // GPU Acceleration Status
  app.get('/api/gpu/status', async (req, res) => {
    try {
      // Request GPU status from Go services
      const gpuStatusResponses = await Promise.allSettled([
        natsConnection.request('gpu.status.rag', '{}', { timeout: 5000 }),
        natsConnection.request('gpu.status.simd', '{}', { timeout: 5000 })
      ])

      const gpuStatus = {
        worker: process.pid,
        timestamp: new Date().toISOString(),
        services: {
          rag: gpuStatusResponses[0].status === 'fulfilled' ? 
            JSON.parse(gpuStatusResponses[0].value.data.toString()) : { error: 'timeout' },
          simd: gpuStatusResponses[1].status === 'fulfilled' ? 
            JSON.parse(gpuStatusResponses[1].value.data.toString()) : { error: 'timeout' }
        }
      }

      // Cache GPU status for 30 seconds
      await redisClient.setEx('gpu:status', 30, JSON.stringify(gpuStatus))

      res.json(gpuStatus)

    } catch (error) {
      console.error('GPU status error:', error)
      res.status(500).json({ error: error.message })
    }
  })

  // Health Check
  app.get('/health', async (req, res) => {
    try {
      const health = {
        status: 'healthy',
        worker: process.pid,
        timestamp: new Date().toISOString(),
        connections: {
          nats: natsConnection?.isClosed() === false,
          redis: redisClient?.isReady === true
        },
        memory: process.memoryUsage(),
        uptime: process.uptime()
      }

      res.json(health)
    } catch (error) {
      res.status(500).json({ status: 'unhealthy', error: error.message })
    }
  })

  // WebSocket for real-time updates
  wss.on('connection', (ws, req) => {
    console.log(`🔌 WebSocket connected from ${req.socket.remoteAddress}`)

    // Subscribe to real-time updates via NATS
    const subscription = natsConnection.subscribe('legal.ai.realtime.*')
    
    ;(async () => {
      for await (const msg of subscription) {
        try {
          const data = JSON.parse(msg.data.toString())
          ws.send(JSON.stringify({
            type: 'realtime_update',
            subject: msg.subject,
            data,
            worker: process.pid
          }))
        } catch (error) {
          console.error('WebSocket broadcast error:', error)
        }
      }
    })()

    ws.on('message', async (message) => {
      try {
        const data = JSON.parse(message.toString())
        
        // Forward client messages to appropriate NATS subjects
        if (data.type === 'subscribe') {
          // Handle subscription requests
        } else if (data.type === 'gpu_command') {
          // Forward GPU commands to microservices
          natsConnection.publish(`gpu.command.${data.service}`, JSON.stringify(data.payload))
        }
      } catch (error) {
        console.error('WebSocket message error:', error)
      }
    })

    ws.on('close', () => {
      console.log(`🔌 WebSocket disconnected`)
      subscription.unsubscribe()
    })
  })

  // Error handling
  app.use((err, req, res, next) => {
    console.error(`❌ Worker ${process.pid} error:`, err)
    res.status(500).json({ 
      error: 'Internal server error', 
      worker: process.pid 
    })
  })

  // Start server
  server.listen(PORT, async () => {
    await initializeConnections()
    console.log(`✅ Worker ${process.pid} listening on port ${PORT}`)
    console.log(`🎮 WebGPU acceleration: ${process.env.WEBGPU_ENABLED === 'true' ? 'ENABLED' : 'DISABLED'}`)
    console.log(`🔧 GPU device: ${process.env.VITE_GPU_DEVICE || 'Unknown'}`)
  })

  // Graceful shutdown
  process.on('SIGTERM', async () => {
    console.log(`🛑 Worker ${process.pid} received SIGTERM, closing connections...`)
    
    try {
      await natsConnection?.close()
      await redisClient?.quit()
      server.close(() => {
        console.log(`💀 Worker ${process.pid} shut down cleanly`)
        process.exit(0)
      })
    } catch (error) {
      console.error(`❌ Worker ${process.pid} shutdown error:`, error)
      process.exit(1)
    }
  })
}