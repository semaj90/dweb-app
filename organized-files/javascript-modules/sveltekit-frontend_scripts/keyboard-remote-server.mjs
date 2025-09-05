#!/usr/bin/env node
/**
 * Keyboard Remote Control Server
 * WebSocket server for remote keyboard shortcuts control
 * Supports multiple client connections and command broadcasting
 */

import WebSocket, { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';

const __dirname = dirname(fileURLToPath(import.meta.url));

class KeyboardRemoteServer {
  constructor(port = 8085) {
    this.port = port;
    this.clients = new Map(); // WebSocket connections
    this.commandHistory = [];
    this.stats = {
      totalConnections: 0,
      activeConnections: 0,
      commandsExecuted: 0,
      startTime: Date.now()
    };
    
    this.server = null;
    this.wss = null;
  }

  async start() {
    try {
      // Create HTTP server for WebSocket upgrade and web interface
      this.server = createServer((req, res) => {
        this.handleHttpRequest(req, res);
      });

      // Create WebSocket server
      this.wss = new WebSocketServer({ 
        server: this.server,
        path: '/keyboard-remote'
      });

      this.wss.on('connection', (ws, req) => {
        this.handleConnection(ws, req);
      });

      this.server.listen(this.port, () => {
        console.log(chalk.green(`🎮 Keyboard Remote Control Server started`));
        console.log(chalk.cyan(`📡 WebSocket: ws://localhost:${this.port}/keyboard-remote`));
        console.log(chalk.cyan(`🌐 Web Interface: http://localhost:${this.port}`));
        console.log(chalk.yellow(`⌨️  Ready to receive remote commands...\n`));
      });

      // Set up periodic stats logging
      setInterval(() => {
        this.logStats();
      }, 30000);

    } catch (error) {
      console.error(chalk.red(`❌ Failed to start server: ${error.message}`));
      process.exit(1);
    }
  }

  handleConnection(ws, req) {
    const clientId = this.generateClientId();
    const clientInfo = {
      id: clientId,
      ws,
      connectedAt: Date.now(),
      lastActivity: Date.now(),
      commandsSent: 0,
      userAgent: req.headers['user-agent'] || 'Unknown',
      ip: req.connection.remoteAddress || 'Unknown'
    };

    this.clients.set(clientId, clientInfo);
    this.stats.totalConnections++;
    this.stats.activeConnections++;

    console.log(chalk.blue(`🔗 Client connected: ${clientId} (${clientInfo.ip})`));
    console.log(chalk.gray(`   User Agent: ${clientInfo.userAgent}`));
    console.log(chalk.gray(`   Active connections: ${this.stats.activeConnections}`));

    // Send welcome message
    this.sendToClient(clientId, {
      type: 'welcome',
      clientId,
      serverVersion: '1.0.0',
      availableCommands: this.getAvailableCommands(),
      stats: this.getPublicStats()
    });

    // Handle messages from client
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        this.handleClientMessage(clientId, message);
      } catch (error) {
        console.error(chalk.red(`❌ Invalid message from ${clientId}: ${error.message}`));
        this.sendError(clientId, 'Invalid JSON message');
      }
    });

    // Handle client disconnect
    ws.on('close', () => {
      console.log(chalk.yellow(`🔌 Client disconnected: ${clientId}`));
      this.clients.delete(clientId);
      this.stats.activeConnections--;
    });

    // Handle connection errors
    ws.on('error', (error) => {
      console.error(chalk.red(`❌ WebSocket error for ${clientId}: ${error.message}`));
    });

    // Update last activity
    clientInfo.lastActivity = Date.now();
  }

  handleClientMessage(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.lastActivity = Date.now();

    console.log(chalk.cyan(`📨 Message from ${clientId}: ${message.type || 'unknown'}`));

    switch (message.type) {
      case 'execute-shortcut':
        this.executeShortcut(clientId, message.payload);
        break;
      
      case 'get-shortcuts':
        this.sendShortcutList(clientId);
        break;
      
      case 'get-stats':
        this.sendStats(clientId);
        break;
      
      case 'ping':
        this.sendToClient(clientId, { type: 'pong', timestamp: Date.now() });
        break;
      
      case 'subscribe':
        this.handleSubscription(clientId, message.payload);
        break;
      
      case 'broadcast':
        this.handleBroadcast(clientId, message.payload);
        break;
      
      default:
        console.log(chalk.yellow(`⚠️  Unknown message type: ${message.type}`));
        this.sendError(clientId, `Unknown message type: ${message.type}`);
    }
  }

  executeShortcut(clientId, payload) {
    if (!payload.command) {
      this.sendError(clientId, 'Missing command in shortcut execution');
      return;
    }

    const command = {
      id: this.generateCommandId(),
      command: payload.command,
      args: payload.args || {},
      source: 'websocket',
      timestamp: Date.now(),
      clientId
    };

    // Add to command history
    this.commandHistory.push(command);
    if (this.commandHistory.length > 100) {
      this.commandHistory = this.commandHistory.slice(-100);
    }

    this.stats.commandsExecuted++;
    
    const client = this.clients.get(clientId);
    if (client) {
      client.commandsSent++;
    }

    console.log(chalk.green(`🎹 Executing shortcut: ${command.command} from ${clientId}`));

    // Broadcast to all connected frontend clients (excluding the sender)
    this.broadcastToFrontends(command, clientId);

    // Send confirmation back to sender
    this.sendToClient(clientId, {
      type: 'command-executed',
      commandId: command.id,
      success: true,
      timestamp: Date.now()
    });
  }

  broadcastToFrontends(command, excludeClientId) {
    let broadcastCount = 0;
    
    for (const [clientId, client] of this.clients.entries()) {
      if (clientId !== excludeClientId && client.ws.readyState === WebSocket.OPEN) {
        try {
          client.ws.send(JSON.stringify(command));
          broadcastCount++;
        } catch (error) {
          console.error(chalk.red(`❌ Broadcast failed to ${clientId}: ${error.message}`));
        }
      }
    }

    console.log(chalk.blue(`📡 Broadcasted command to ${broadcastCount} clients`));
  }

  sendToClient(clientId, message) {
    const client = this.clients.get(clientId);
    if (client && client.ws.readyState === WebSocket.OPEN) {
      try {
        client.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error(chalk.red(`❌ Send failed to ${clientId}: ${error.message}`));
      }
    }
  }

  sendError(clientId, errorMessage) {
    this.sendToClient(clientId, {
      type: 'error',
      message: errorMessage,
      timestamp: Date.now()
    });
  }

  sendShortcutList(clientId) {
    const shortcuts = this.getAvailableCommands();
    this.sendToClient(clientId, {
      type: 'shortcuts',
      shortcuts,
      timestamp: Date.now()
    });
  }

  sendStats(clientId) {
    const stats = {
      ...this.stats,
      clients: Array.from(this.clients.values()).map(client => ({
        id: client.id,
        connectedAt: client.connectedAt,
        lastActivity: client.lastActivity,
        commandsSent: client.commandsSent,
        ip: client.ip.replace(/^.*:/, '') // Hide full IP for privacy
      })),
      recentCommands: this.commandHistory.slice(-10)
    };

    this.sendToClient(clientId, {
      type: 'stats',
      stats,
      timestamp: Date.now()
    });
  }

  getAvailableCommands() {
    // Define available keyboard shortcuts that can be executed remotely
    return [
      { id: 'ai-chat-toggle', description: 'Toggle AI chat', category: 'ai', remote: true },
      { id: 'ai-analyze', description: 'Analyze current document', category: 'ai', remote: true },
      { id: 'ai-summarize', description: 'Summarize content', category: 'ai', remote: true },
      { id: 'nav-home', description: 'Navigate to home', category: 'navigation', remote: false },
      { id: 'nav-cases', description: 'Navigate to cases', category: 'navigation', remote: false },
      { id: 'nav-evidence', description: 'Navigate to evidence', category: 'navigation', remote: false },
      { id: 'nav-ai', description: 'Navigate to AI assistant', category: 'navigation', remote: false },
      { id: 'case-new', description: 'Create new case', category: 'cases', remote: false },
      { id: 'case-search', description: 'Search cases', category: 'cases', remote: false },
      { id: 'evidence-upload', description: 'Upload evidence', category: 'evidence', remote: false },
      { id: 'evidence-tag', description: 'Tag evidence', category: 'evidence', remote: false },
      { id: 'system-command-palette', description: 'Open command palette', category: 'system', remote: false },
      { id: 'system-help', description: 'Show help', category: 'system', remote: false },
      { id: 'remote-connect', description: 'Connect remote', category: 'remote', remote: true },
      { id: 'remote-disconnect', description: 'Disconnect remote', category: 'remote', remote: true },
      { id: 'remote-status', description: 'Show remote status', category: 'remote', remote: true }
    ];
  }

  getPublicStats() {
    return {
      activeConnections: this.stats.activeConnections,
      totalConnections: this.stats.totalConnections,
      commandsExecuted: this.stats.commandsExecuted,
      uptime: Date.now() - this.stats.startTime
    };
  }

  handleHttpRequest(req, res) {
    // Simple web interface for remote control
    if (req.url === '/') {
      const html = this.generateWebInterface();
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } else if (req.url === '/api/stats') {
      const stats = {
        ...this.stats,
        clients: this.clients.size,
        recentCommands: this.commandHistory.slice(-10)
      };
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(stats, null, 2));
    } else if (req.url === '/api/commands') {
      const commands = this.getAvailableCommands();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(commands, null, 2));
    } else {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
    }
  }

  generateWebInterface() {
    const uptime = Math.round((Date.now() - this.stats.startTime) / 1000);
    
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keyboard Remote Control Server</title>
    <style>
        body { font-family: 'Monaco', 'Consolas', monospace; background: #1a1a1a; color: #00ff00; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 1px solid #333; padding-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #2a2a2a; padding: 15px; border-radius: 8px; border-left: 4px solid #00ff00; }
        .stat-value { font-size: 2em; font-weight: bold; color: #00ff00; }
        .stat-label { color: #888; margin-top: 5px; }
        .section { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .section h3 { color: #00ff00; margin-top: 0; }
        .command-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
        .command-card { background: #333; padding: 15px; border-radius: 6px; border: 1px solid #444; }
        .command-card.remote { border-color: #00ff00; }
        .command-title { font-weight: bold; color: #00ff00; }
        .command-desc { color: #ccc; margin: 5px 0; }
        .command-category { color: #888; font-size: 0.9em; }
        .remote-badge { background: #00ff00; color: #000; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
        .log { background: #1a1a1a; color: #00ff00; font-family: 'Monaco', monospace; font-size: 0.9em; height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #333; }
        .connection-test { margin-top: 20px; text-align: center; }
        button { background: #333; color: #00ff00; border: 1px solid #00ff00; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #00ff00; color: #000; }
        .status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
        .status.online { background: #00ff00; color: #000; }
        .status.offline { background: #ff0000; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Keyboard Remote Control Server</h1>
            <p>Legal AI Platform - Remote Command Interface</p>
            <div class="status online">Server Online</div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">${this.stats.activeConnections}</div>
                <div class="stat-label">Active Connections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${this.stats.totalConnections}</div>
                <div class="stat-label">Total Connections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${this.stats.commandsExecuted}</div>
                <div class="stat-label">Commands Executed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${uptime}s</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>

        <div class="section">
            <h3>📡 Connection Information</h3>
            <p><strong>WebSocket Endpoint:</strong> ws://localhost:${this.port}/keyboard-remote</p>
            <p><strong>HTTP API:</strong> http://localhost:${this.port}/api/</p>
            <div class="connection-test">
                <button onclick="testConnection()">Test Connection</button>
                <button onclick="sendTestCommand()">Send Test Command</button>
                <button onclick="location.reload()">Refresh Status</button>
            </div>
            <div id="connectionStatus" class="log" style="margin-top: 10px; height: 100px;"></div>
        </div>

        <div class="section">
            <h3>⌨️ Available Commands</h3>
            <div class="command-grid">
                ${this.getAvailableCommands().map(cmd => `
                    <div class="command-card ${cmd.remote ? 'remote' : ''}">
                        <div class="command-title">${cmd.id}</div>
                        <div class="command-desc">${cmd.description}</div>
                        <div class="command-category">
                            Category: ${cmd.category} 
                            ${cmd.remote ? '<span class="remote-badge">Remote</span>' : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="section">
            <h3>📜 Recent Commands</h3>
            <div class="log">
                ${this.commandHistory.slice(-10).map(cmd => 
                    `[${new Date(cmd.timestamp).toLocaleTimeString()}] ${cmd.command} (from ${cmd.clientId})`
                ).join('<br>') || 'No commands executed yet'}
            </div>
        </div>
    </div>

    <script>
        let ws = null;

        function log(message) {
            const statusDiv = document.getElementById('connectionStatus');
            const timestamp = new Date().toLocaleTimeString();
            statusDiv.innerHTML += \`[\${timestamp}] \${message}<br>\`;
            statusDiv.scrollTop = statusDiv.scrollHeight;
        }

        function testConnection() {
            log('Testing WebSocket connection...');
            
            ws = new WebSocket('ws://localhost:${this.port}/keyboard-remote');
            
            ws.onopen = function() {
                log('✅ WebSocket connected successfully');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                log(\`📨 Received: \${data.type}\`);
            };
            
            ws.onclose = function() {
                log('🔌 WebSocket connection closed');
            };
            
            ws.onerror = function(error) {
                log('❌ WebSocket error: ' + error);
            };
        }

        function sendTestCommand() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                log('⚠️ No active WebSocket connection. Test connection first.');
                return;
            }
            
            const testCommand = {
                type: 'execute-shortcut',
                payload: {
                    command: 'system-help',
                    args: { test: true }
                }
            };
            
            ws.send(JSON.stringify(testCommand));
            log('🎹 Sent test command: system-help');
        }

        // Auto-refresh stats every 10 seconds
        setInterval(() => {
            fetch('/api/stats')
                .then(response => response.json())
                .then(stats => {
                    // Update stats display would go here
                })
                .catch(console.error);
        }, 10000);
    </script>
</body>
</html>`;
  }

  handleSubscription(clientId, payload) {
    // Handle client subscriptions to specific events
    const client = this.clients.get(clientId);
    if (client) {
      client.subscriptions = payload.events || [];
      console.log(chalk.blue(`📝 Client ${clientId} subscribed to: ${client.subscriptions.join(', ')}`));
    }
  }

  handleBroadcast(clientId, payload) {
    // Handle broadcast messages from clients
    if (!payload.message) return;

    console.log(chalk.magenta(`📢 Broadcast from ${clientId}: ${payload.message}`));
    
    // Broadcast to all other connected clients
    const broadcastMessage = {
      type: 'broadcast',
      from: clientId,
      message: payload.message,
      timestamp: Date.now()
    };

    this.broadcastToFrontends(broadcastMessage, clientId);
  }

  generateClientId() {
    return 'client_' + Math.random().toString(36).substr(2, 9);
  }

  generateCommandId() {
    return 'cmd_' + Math.random().toString(36).substr(2, 9);
  }

  logStats() {
    if (this.stats.activeConnections > 0) {
      console.log(chalk.cyan('\n📊 Server Statistics:'));
      console.log(chalk.gray(`   Active connections: ${this.stats.activeConnections}`));
      console.log(chalk.gray(`   Total connections: ${this.stats.totalConnections}`));
      console.log(chalk.gray(`   Commands executed: ${this.stats.commandsExecuted}`));
      console.log(chalk.gray(`   Uptime: ${Math.round((Date.now() - this.stats.startTime) / 1000)}s\n`));
    }
  }

  stop() {
    console.log(chalk.yellow('\n🛑 Shutting down server...'));
    
    // Close all client connections
    for (const [clientId, client] of this.clients.entries()) {
      client.ws.close();
      console.log(chalk.gray(`   Closed connection: ${clientId}`));
    }

    // Close servers
    if (this.wss) {
      this.wss.close();
    }
    
    if (this.server) {
      this.server.close();
    }

    console.log(chalk.green('✅ Server stopped successfully\n'));
  }
}

// Main execution
async function main() {
  const port = process.argv[2] ? parseInt(process.argv[2]) : 8085;
  const server = new KeyboardRemoteServer(port);

  // Graceful shutdown handling
  process.on('SIGINT', () => {
    server.stop();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    server.stop();
    process.exit(0);
  });

  await server.start();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { KeyboardRemoteServer };