#!/usr/bin/env node

/**
 * Real-time Monitoring Dashboard for Legal AI Indexing System
 * WebSocket-based real-time metrics and health monitoring
 */

import express from 'express'
import http from 'http'
import WebSocket, { WebSocketServer } from 'ws'
import os from 'os'
import path from 'path'
import fs from 'fs/promises'
import { performance } from 'perf_hooks'
import { exec } from 'child_process'
import { promisify } from 'util'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const app = express()
const server = http.createServer(app)
const wss = new WebSocketServer({ server })

// Configuration
const PORT = process.env.PORT || 8084
const UPDATE_INTERVAL = 2000 // 2 seconds
const METRICS_HISTORY_SIZE = 100

// Global metrics storage
let systemMetrics = {
  timestamp: Date.now(),
  cpu: [],
  memory: {},
  processes: [],
  indexing: {
    totalFiles: 0,
    processedFiles: 0,
    failedFiles: 0,
    rate: 0,
    estimatedCompletion: null
  },
  services: {
    ollama: false,
    goService: false,
    autoGen: false
  }
}

let metricsHistory = []
let connectedClients = new Set()

// Middleware
app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')))

/**
 * WebSocket connection handler
 */
wss.on('connection', (ws) => {
  console.log('📱 New dashboard client connected')
  connectedClients.add(ws)
  
  // Send current metrics immediately
  ws.send(JSON.stringify({
    type: 'metrics',
    data: systemMetrics,
    history: metricsHistory.slice(-20) // Last 20 data points
  }))
  
  ws.on('close', () => {
    console.log('📱 Dashboard client disconnected')
    connectedClients.delete(ws)
  })
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message)
      handleClientMessage(ws, data)
    } catch (error) {
      console.error('Invalid message from client:', error)
    }
  })
})

/**
 * Handle messages from dashboard clients
 */
function handleClientMessage(ws, data) {
  switch (data.type) {
    case 'get_logs':
      sendLogs(ws, data.service)
      break
    case 'restart_service':
      restartService(data.service)
      break
    case 'get_detailed_metrics':
      sendDetailedMetrics(ws)
      break
    default:
      console.log('Unknown message type:', data.type)
  }
}

/**
 * REST API endpoints
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    metrics: systemMetrics
  })
})

app.get('/api/metrics', (req, res) => {
  res.json({
    current: systemMetrics,
    history: metricsHistory
  })
})

app.get('/api/services', (req, res) => {
  res.json(systemMetrics.services)
})

app.post('/api/services/:service/restart', async (req, res) => {
  try {
    await restartService(req.params.service)
    res.json({ success: true, message: `Service ${req.params.service} restart initiated` })
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

app.get('/api/logs/:service', async (req, res) => {
  try {
    const logs = await getLogs(req.params.service)
    res.json({ service: req.params.service, logs })
  } catch (error) {
    res.status(500).json({ error: error.message })
  }
})

/**
 * System metrics collection
 */
async function collectSystemMetrics() {
  const startTime = performance.now()
  
  try {
    // CPU usage
    const cpuUsage = os.loadavg()
    
    // Memory usage
    const memoryUsage = {
      total: os.totalmem(),
      free: os.freemem(),
      used: os.totalmem() - os.freemem(),
      percentage: ((os.totalmem() - os.freemem()) / os.totalmem()) * 100
    }
    
    // Process information
    const processes = await getProcessInfo()
    
    // Service health checks
    const services = await checkAllServices()
    
    // Indexing progress (from various sources)
    const indexing = await getIndexingProgress()
    
    systemMetrics = {
      timestamp: Date.now(),
      cpu: cpuUsage,
      memory: memoryUsage,
      processes,
      indexing,
      services,
      collectionTime: performance.now() - startTime
    }
    
    // Store in history
    metricsHistory.push({ ...systemMetrics })
    if (metricsHistory.length > METRICS_HISTORY_SIZE) {
      metricsHistory.shift()
    }
    
    // Broadcast to all connected clients
    broadcastMetrics()
    
  } catch (error) {
    console.error('Error collecting metrics:', error)
  }
}

/**
 * Get process information for all services
 */
async function getProcessInfo() {
  const processes = []
  
  try {
    // Check PM2 processes
    // exec and promisify imported at top
    const execAsync = promisify(exec)
    
    const { stdout } = await execAsync('pm2 jlist')
    const pm2Processes = JSON.parse(stdout)
    
    for (const proc of pm2Processes) {
      if (proc.name && proc.name.startsWith('legal-ai-')) {
        processes.push({
          name: proc.name,
          pid: proc.pid,
          status: proc.pm2_env.status,
          cpu: proc.monit.cpu,
          memory: proc.monit.memory,
          uptime: Date.now() - proc.pm2_env.pm_uptime,
          restarts: proc.pm2_env.restart_time
        })
      }
    }
  } catch (error) {
    console.error('Error getting PM2 process info:', error)
  }
  
  return processes
}

/**
 * Check health of all services
 */
async function checkAllServices() {
  const services = {
    ollama: false,
    goService: false,
    autoGen: false,
    clusterService: false,
    monitoring: true // This service is always healthy if running
  }
  
  // Check Ollama
  try {
    const response = await fetch('http://localhost:11434/api/tags', { 
      timeout: 5000 
    })
    services.ollama = response.ok
  } catch (error) {
    services.ollama = false
  }
  
  // Check Go Service
  try {
    const response = await fetch('http://localhost:8081/api/health', { 
      timeout: 5000 
    })
    services.goService = response.ok
  } catch (error) {
    services.goService = false
  }
  
  // Check AutoGen (WebSocket)
  try {
    const response = await fetch('http://localhost:8083/health', { 
      timeout: 5000 
    })
    services.autoGen = response.ok
  } catch (error) {
    services.autoGen = false
  }
  
  // Check GPU Clustering Service
  try {
    const response = await fetch('http://localhost:8085/api/health', { 
      timeout: 5000 
    })
    services.clusterService = response.ok
  } catch (error) {
    services.clusterService = false
  }
  
  return services
}

/**
 * Get indexing progress from various sources
 */
async function getIndexingProgress() {
  let progress = {
    totalFiles: 0,
    processedFiles: 0,
    failedFiles: 0,
    rate: 0,
    estimatedCompletion: null,
    activeWorkers: 0
  }
  
  // Try to get progress from Go service
  try {
    const response = await fetch('http://localhost:8081/api/progress', { 
      timeout: 3000 
    })
    if (response.ok) {
      const data = await response.json()
      progress = { ...progress, ...data }
    }
  } catch (error) {
    // Try other sources or use file-based progress
  }
  
  // Try to read progress from file system
  try {
    const progressFile = path.join(__dirname, 'indexing-output', 'progress.json')
    const data = await fs.readFile(progressFile, 'utf8')
    const fileProgress = JSON.parse(data)
    progress = { ...progress, ...fileProgress }
  } catch (error) {
    // No file-based progress available
  }
  
  return progress
}

/**
 * Get logs for a specific service
 */
async function getLogs(serviceName, lines = 100) {
  try {
    const logFile = path.join(__dirname, 'logs', `${serviceName}.log`)
    const content = await fs.readFile(logFile, 'utf8')
    const logLines = content.split('\n').slice(-lines)
    return logLines.join('\n')
  } catch (error) {
    return `Error reading logs for ${serviceName}: ${error.message}`
  }
}

/**
 * Send logs to WebSocket client
 */
async function sendLogs(ws, service) {
  try {
    const logs = await getLogs(service)
    ws.send(JSON.stringify({
      type: 'logs',
      service: service,
      data: logs
    }))
  } catch (error) {
    ws.send(JSON.stringify({
      type: 'error',
      message: `Failed to get logs for ${service}: ${error.message}`
    }))
  }
}

/**
 * Restart a service using PM2
 */
async function restartService(serviceName) {
  // exec and promisify imported at top
  const execAsync = promisify(exec)
  
  try {
    await execAsync(`pm2 restart legal-ai-${serviceName}`)
    console.log(`✅ Restarted service: ${serviceName}`)
    
    // Broadcast service restart notification
    broadcastNotification({
      type: 'service_restart',
      service: serviceName,
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error(`❌ Failed to restart ${serviceName}:`, error)
    throw error
  }
}

/**
 * Send detailed metrics to client
 */
async function sendDetailedMetrics(ws) {
  const detailedMetrics = {
    system: {
      platform: os.platform(),
      arch: os.arch(),
      hostname: os.hostname(),
      uptime: os.uptime(),
      cpuCount: os.cpus().length,
      cpuInfo: os.cpus()[0]
    },
    node: {
      version: process.version,
      memoryUsage: process.memoryUsage(),
      uptime: process.uptime()
    },
    metrics: systemMetrics,
    history: metricsHistory
  }
  
  ws.send(JSON.stringify({
    type: 'detailed_metrics',
    data: detailedMetrics
  }))
}

/**
 * Broadcast metrics to all connected clients
 */
function broadcastMetrics() {
  const message = JSON.stringify({
    type: 'metrics_update',
    data: systemMetrics
  })
  
  connectedClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message)
    }
  })
}

/**
 * Broadcast notification to all clients
 */
function broadcastNotification(notification) {
  const message = JSON.stringify({
    type: 'notification',
    data: notification
  })
  
  connectedClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message)
    }
  })
}

/**
 * Generate HTML dashboard
 */
app.get('/', (req, res) => {
  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal AI Indexing Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #fff; overflow-x: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
        .panel { background: #1a1a1a; border-radius: 12px; padding: 20px; border: 1px solid #333; }
        .panel h3 { margin-bottom: 15px; color: #667eea; }
        .metric { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding: 8px 0; border-bottom: 1px solid #333; }
        .metric:last-child { border-bottom: none; }
        .metric-value { font-weight: bold; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-healthy { background: #4CAF50; }
        .status-unhealthy { background: #f44336; }
        .progress-bar { width: 100%; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); transition: width 0.3s; }
        .chart-container { height: 200px; margin: 20px 0; }
        .log-container { background: #0d1117; border-radius: 8px; padding: 15px; font-family: 'Monaco', 'Menlo', monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
        .log-line { margin-bottom: 5px; }
        .btn { background: #667eea; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #5a6fd8; }
        .notification { position: fixed; top: 20px; right: 20px; background: #4CAF50; color: white; padding: 15px; border-radius: 8px; z-index: 1000; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Legal AI Indexing Dashboard</h1>
        <p>Real-time monitoring and control</p>
    </div>
    
    <div class="dashboard">
        <div class="panel">
            <h3>System Metrics</h3>
            <div class="metric">
                <span>CPU Usage</span>
                <span class="metric-value" id="cpu-usage">0%</span>
            </div>
            <div class="metric">
                <span>Memory Usage</span>
                <span class="metric-value" id="memory-usage">0%</span>
            </div>
            <div class="metric">
                <span>Uptime</span>
                <span class="metric-value" id="uptime">0s</span>
            </div>
            <div class="chart-container">
                <canvas id="systemChart"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h3>Service Status</h3>
            <div class="metric">
                <span><span class="status-indicator" id="ollama-status"></span>Ollama</span>
                <button class="btn" onclick="restartService('ollama')">Restart</button>
            </div>
            <div class="metric">
                <span><span class="status-indicator" id="go-status"></span>Go Service</span>
                <button class="btn" onclick="restartService('go')">Restart</button>
            </div>
            <div class="metric">
                <span><span class="status-indicator" id="autogen-status"></span>AutoGen</span>
                <button class="btn" onclick="restartService('autogen')">Restart</button>
            </div>
            <div class="metric">
                <span><span class="status-indicator" id="cluster-status"></span>GPU Clustering</span>
                <button class="btn" onclick="restartService('cluster-gpu')">Restart</button>
            </div>
        </div>
        
        <div class="panel">
            <h3>Indexing Progress</h3>
            <div class="metric">
                <span>Total Files</span>
                <span class="metric-value" id="total-files">0</span>
            </div>
            <div class="metric">
                <span>Processed</span>
                <span class="metric-value" id="processed-files">0</span>
            </div>
            <div class="metric">
                <span>Processing Rate</span>
                <span class="metric-value" id="processing-rate">0 files/sec</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="panel">
            <h3>Process Monitor</h3>
            <div id="process-list"></div>
        </div>
        
        <div class="panel" style="grid-column: 1 / -1;">
            <h3>System Logs</h3>
            <select id="log-selector" onchange="switchLogs()">
                <option value="indexer-go">Go Indexer</option>
                <option value="autogen">AutoGen</option>
                <option value="concurrent">Concurrent Processor</option>
                <option value="monitor">Monitor</option>
            </select>
            <div class="log-container" id="log-display">
                <div class="log-line">Connecting to log stream...</div>
            </div>
        </div>
    </div>
    
    <script>
        let socket = null;
        let systemChart = null;
        
        function initWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            socket = new WebSocket(protocol + '//' + location.host);
            
            socket.onopen = () => {
                console.log('Connected to dashboard');
                showNotification('Connected to monitoring system', 'success');
            };
            
            socket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            socket.onclose = () => {
                console.log('Disconnected from dashboard');
                showNotification('Connection lost, attempting reconnect...', 'warning');
                setTimeout(initWebSocket, 5000);
            };
            
            socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleMessage(message) {
            switch (message.type) {
                case 'metrics':
                case 'metrics_update':
                    updateMetrics(message.data);
                    break;
                case 'logs':
                    updateLogs(message.service, message.data);
                    break;
                case 'notification':
                    showNotification(message.data.message || 'System notification', 'info');
                    break;
            }
        }
        
        function updateMetrics(metrics) {
            // System metrics
            document.getElementById('cpu-usage').textContent = metrics.cpu[0]?.toFixed(1) + '%' || '0%';
            document.getElementById('memory-usage').textContent = metrics.memory.percentage?.toFixed(1) + '%' || '0%';
            document.getElementById('uptime').textContent = formatUptime(process.uptime);
            
            // Service status
            updateServiceStatus('ollama-status', metrics.services.ollama);
            updateServiceStatus('go-status', metrics.services.goService);
            updateServiceStatus('autogen-status', metrics.services.autoGen);
            updateServiceStatus('cluster-status', metrics.services.clusterService);
            
            // Indexing progress
            document.getElementById('total-files').textContent = metrics.indexing.totalFiles.toLocaleString();
            document.getElementById('processed-files').textContent = metrics.indexing.processedFiles.toLocaleString();
            document.getElementById('processing-rate').textContent = metrics.indexing.rate.toFixed(1) + ' files/sec';
            
            const progressPercent = metrics.indexing.totalFiles > 0 
                ? (metrics.indexing.processedFiles / metrics.indexing.totalFiles) * 100 
                : 0;
            document.getElementById('progress-fill').style.width = progressPercent + '%';
            
            // Update process list
            updateProcessList(metrics.processes);
            
            // Update chart
            updateSystemChart(metrics);
        }
        
        function updateServiceStatus(elementId, isHealthy) {
            const element = document.getElementById(elementId);
            element.className = 'status-indicator ' + (isHealthy ? 'status-healthy' : 'status-unhealthy');
        }
        
        function updateProcessList(processes) {
            const container = document.getElementById('process-list');
            container.innerHTML = processes.map(proc => 
                '<div class="metric">' +
                '<span>' + proc.name + '</span>' +
                '<span class="metric-value">' + proc.status + '</span>' +
                '</div>'
            ).join('');
        }
        
        function updateSystemChart(metrics) {
            // Chart implementation would go here
        }
        
        function restartService(serviceName) {
            fetch('/api/services/' + serviceName + '/restart', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showNotification('Service restart initiated: ' + serviceName, 'success');
                    } else {
                        showNotification('Failed to restart service: ' + serviceName, 'error');
                    }
                })
                .catch(error => {
                    showNotification('Error restarting service: ' + error.message, 'error');
                });
        }
        
        function switchLogs() {
            const selector = document.getElementById('log-selector');
            const service = selector.value;
            
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'get_logs',
                    service: service
                }));
            }
        }
        
        function updateLogs(service, logs) {
            const display = document.getElementById('log-display');
            display.innerHTML = logs.split('\\n').map(line => 
                '<div class="log-line">' + escapeHtml(line) + '</div>'
            ).join('');
            display.scrollTop = display.scrollHeight;
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.className = 'notification';
            notification.style.background = type === 'success' ? '#4CAF50' : 
                                          type === 'warning' ? '#FF9800' : 
                                          type === 'error' ? '#f44336' : '#2196F3';
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }
        
        function formatUptime(seconds) {
            const days = Math.floor(seconds / 86400);
            const hours = Math.floor((seconds % 86400) / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return days + 'd ' + hours + 'h ' + minutes + 'm';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Initialize
        initWebSocket();
        
        // Auto-refresh every 5 seconds
        setInterval(() => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'get_detailed_metrics' }));
            }
        }, 5000);
    </script>
</body>
</html>`
  
  res.send(html)
})

/**
 * Start monitoring system
 */
async function startMonitoring() {
  console.log('🚀 Starting Legal AI Indexing Monitor Dashboard')
  
  // Ensure logs directory exists
  try {
    await fs.mkdir(path.join(__dirname, 'logs'), { recursive: true })
  } catch (error) {
    // Directory might already exist
  }
  
  // Start metrics collection
  setInterval(collectSystemMetrics, UPDATE_INTERVAL)
  
  // Initial metrics collection
  await collectSystemMetrics()
  
  // Start server
  server.listen(PORT, () => {
    console.log(`📊 Monitor Dashboard running on http://localhost:${PORT}`)
    console.log(`📈 Metrics collection interval: ${UPDATE_INTERVAL}ms`)
    console.log(`🔌 WebSocket server ready for real-time updates`)
  })
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('📊 Monitor Dashboard shutting down...')
  server.close(() => {
    process.exit(0)
  })
})

// Start the monitoring system
startMonitoring().catch(console.error)