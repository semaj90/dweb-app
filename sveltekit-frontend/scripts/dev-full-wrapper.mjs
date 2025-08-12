// dev-full-wrapper.mjs
// Enhanced development environment with all services

import { spawn } from 'child_process';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import readline from 'readline';
import path from 'path';
import fs from 'fs/promises';

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

class DevEnvironment {
  constructor() {
    this.services = new Map();
    this.logs = [];
    this.isWindows = process.platform === 'win32';
    this.wsServer = null;
    this.clients = new Set();
  }

  log(service, message, level = 'info') {
    const timestamp = new Date().toISOString();
    const entry = { timestamp, service, message, level };
    this.logs.push(entry);
    
    // Keep only last 1000 logs
    if (this.logs.length > 1000) {
      this.logs.shift();
    }
    
    // Format for console
    const colors = {
      error: COLORS.red,
      warn: COLORS.yellow,
      info: COLORS.cyan,
      success: COLORS.green
    };
    
    const color = colors[level] || COLORS.reset;
    const serviceTag = `[${service}]`.padEnd(12);
    
    console.log(`${color}${serviceTag}${COLORS.reset} ${message}`);
    
    // Broadcast to WebSocket clients
    this.broadcast({ type: 'log', data: entry });
  }

  broadcast(message) {
    const data = JSON.stringify(message);
    this.clients.forEach(client => {
      if (client.readyState === 1) { // WebSocket.OPEN
        client.send(data);
      }
    });
  }

  async startService(name, command, args = [], options = {}) {
    return new Promise((resolve) => {
      this.log('System', `Starting ${name}...`, 'info');
      
      const proc = spawn(command, args, {
        shell: this.isWindows,
        stdio: ['inherit', 'pipe', 'pipe'],
        env: { ...process.env, ...options.env },
        cwd: options.cwd || process.cwd()
      });
      
      proc.stdout?.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        lines.forEach(line => {
          if (options.filter && !options.filter(line)) return;
          this.log(name, line.trim());
        });
      });
      
      proc.stderr?.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        lines.forEach(line => {
          if (options.errorFilter && !options.errorFilter(line)) return;
          
          // Determine if it's actually an error or just info on stderr
          const level = line.toLowerCase().includes('error') ? 'error' : 
                       line.toLowerCase().includes('warn') ? 'warn' : 'info';
          this.log(name, line.trim(), level);
        });
      });
      
      proc.on('close', (code) => {
        this.log(name, `Process exited with code ${code}`, code === 0 ? 'info' : 'error');
        this.services.delete(name);
      });
      
      proc.on('error', (err) => {
        this.log(name, `Failed to start: ${err.message}`, 'error');
        resolve(false);
      });
      
      this.services.set(name, proc);
      
      // Consider service started after a delay
      setTimeout(() => resolve(true), options.startDelay || 2000);
    });
  }

  async checkPort(port, retries = 5) {
    const net = await import('net');
    
    for (let i = 0; i < retries; i++) {
      const isAvailable = await new Promise((resolve) => {
        const server = net.createServer();
        server.once('error', () => resolve(false));
        server.once('listening', () => {
          server.close();
          resolve(true);
        });
        server.listen(port);
      });
      
      if (isAvailable) return true;
      
      // Port is in use, try to find and kill the process
      if (i === 0) {
        this.log('System', `Port ${port} is in use, attempting to free it...`, 'warn');
        await this.killProcessOnPort(port);
        await new Promise(r => setTimeout(r, 1000));
      } else {
        await new Promise(r => setTimeout(r, 1000));
      }
    }
    
    return false;
  }

  async killProcessOnPort(port) {
    try {
      if (this.isWindows) {
        const { exec } = await import('child_process');
        const { promisify } = await import('util');
        const execAsync = promisify(exec);
        
        const { stdout } = await execAsync(`netstat -ano | findstr :${port}`);
        const lines = stdout.split('\n').filter(l => l.includes('LISTENING'));
        
        for (const line of lines) {
          const pid = line.trim().split(/\s+/).pop();
          if (pid && pid !== '0') {
            await execAsync(`taskkill /F /PID ${pid}`);
            this.log('System', `Killed process ${pid} on port ${port}`, 'info');
          }
        }
      } else {
        const { exec } = await import('child_process');
        const { promisify } = await import('util');
        const execAsync = promisify(exec);
        
        await execAsync(`lsof -ti:${port} | xargs kill -9`);
      }
    } catch {
      // Ignore errors if no process found
    }
  }

  async startRedis() {
    if (await this.checkPort(6379)) {
      const started = await this.startService('Redis', 'redis-server', [], {
        filter: (line) => !line.includes('WARNING'),
        startDelay: 1000
      });
      
      if (started) {
        this.log('Redis', 'Cache service ready', 'success');
      } else {
        this.log('Redis', 'Not available - using memory cache', 'warn');
        process.env.USE_MEMORY_CACHE = 'true';
      }
    }
  }

  async startOllama() {
    if (await this.checkPort(11434)) {
      const started = await this.startService('Ollama', 'ollama', ['serve'], {
        filter: (line) => !line.includes('routes registered'),
        startDelay: 3000
      });
      
      if (started) {
        this.log('Ollama', 'AI service ready', 'success');
        
        // Check for gemma model
        setTimeout(async () => {
          try {
            const response = await fetch('http://localhost:11434/api/tags');
            const data = await response.json();
            const hasGemma = data.models?.some(m => m.name?.includes('gemma'));
            
            if (!hasGemma) {
              this.log('Ollama', 'Gemma3-legal model not found - run: ollama pull gemma3-legal:latest', 'warn');
            }
          } catch {}
        }, 2000);
      }
    }
  }

  async startGoService() {
    // Check if main.go exists in parent directory
    const goPath = path.join(process.cwd(), '..', 'main.go');
    try {
      await fs.access(goPath);
    } catch {
      this.log('Go', 'main.go not found - API features disabled', 'warn');
      return;
    }
    
    if (await this.checkPort(8084)) {
      const env = {
        PORT: '8084',
        REDIS_ADDR: 'localhost:6379',
        OLLAMA_URL: 'http://localhost:11434',
        MAX_CONCURRENCY: '3',
        ENABLE_GPU: process.env.ENABLE_GPU || 'true',
        GPU_MEMORY_LIMIT_MB: '6000'
      };
      
      const started = await this.startService('Go API', 'go', ['run', 'main.go'], {
        cwd: path.join(process.cwd(), '..'),
        env,
        filter: (line) => {
          // Filter out verbose logs
          return !line.includes('cors') && !line.includes('[GIN-debug]');
        },
        startDelay: 3000
      });
      
      if (started) {
        this.log('Go API', 'Legal AI service ready on port 8084', 'success');
      }
    }
  }

  async startVite() {
    if (await this.checkPort(5173)) {
      const env = {
        NODE_ENV: 'development',
        NODE_OPTIONS: '--max-old-space-size=4096',
        VITE_LEGAL_AI_API: 'http://localhost:8084',
        VITE_OLLAMA_URL: 'http://localhost:11434',
        VITE_REDIS_URL: 'redis://localhost:6379'
      };
      
      await this.startService('Vite', 'npm', ['run', 'dev'], {
        env,
        filter: (line) => {
          // Filter out routine Vite logs
          return !line.includes('hmr update') && 
                 !line.includes('page reload') &&
                 !line.includes('vite:transform');
        }
      });
      
      this.log('Vite', 'Frontend ready on http://localhost:5173', 'success');
    }
  }

  async startWebSocketServer() {
    // Create WebSocket server for real-time monitoring
    const server = createServer();
    this.wsServer = new WebSocketServer({ server });
    
    this.wsServer.on('connection', (ws) => {
      this.clients.add(ws);
      
      // Send initial state
      ws.send(JSON.stringify({
        type: 'init',
        data: {
          services: Array.from(this.services.keys()),
          logs: this.logs.slice(-100)
        }
      }));
      
      ws.on('close', () => {
        this.clients.delete(ws);
      });
    });
    
    server.listen(8085, () => {
      this.log('Monitor', 'WebSocket server on ws://localhost:8085', 'info');
    });
  }

  setupShutdown() {
    const shutdown = async () => {
      console.log('\n');
      this.log('System', 'Shutting down services...', 'warn');
      
      // Kill all services
      for (const [name, proc] of this.services) {
        this.log('System', `Stopping ${name}...`, 'info');
        
        if (this.isWindows) {
          // On Windows, use taskkill to ensure child processes are killed
          try {
            const { exec } = await import('child_process');
            const { promisify } = await import('util');
            const execAsync = promisify(exec);
            await execAsync(`taskkill /F /T /PID ${proc.pid}`);
          } catch {
            proc.kill('SIGTERM');
          }
        } else {
          proc.kill('SIGTERM');
        }
      }
      
      // Close WebSocket server
      if (this.wsServer) {
        this.wsServer.close();
      }
      
      this.log('System', 'All services stopped', 'success');
      process.exit(0);
    };
    
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    // Windows-specific
    if (this.isWindows) {
      readline.createInterface({
        input: process.stdin,
        output: process.stdout
      }).on('SIGINT', shutdown);
    }
  }

  async start() {
    console.clear();
    console.log(`${COLORS.cyan}${COLORS.bright}╔════════════════════════════════════════════════╗${COLORS.reset}`);
    console.log(`${COLORS.cyan}${COLORS.bright}║     LEGAL AI FULL DEVELOPMENT ENVIRONMENT     ║${COLORS.reset}`);
    console.log(`${COLORS.cyan}${COLORS.bright}╚════════════════════════════════════════════════╝${COLORS.reset}`);
    console.log();
    
    this.setupShutdown();
    
    // Start services in order
    await this.startWebSocketServer();
    await this.startRedis();
    await this.startOllama();
    await this.startGoService();
    
    // Small delay before starting Vite
    await new Promise(r => setTimeout(r, 2000));
    await this.startVite();
    
    console.log();
    console.log(`${COLORS.green}${COLORS.bright}════════════════════════════════════════════════${COLORS.reset}`);
    console.log(`${COLORS.green}${COLORS.bright}     All services started successfully!         ${COLORS.reset}`);
    console.log(`${COLORS.green}${COLORS.bright}════════════════════════════════════════════════${COLORS.reset}`);
    console.log();
    console.log(`${COLORS.cyan}📌 Access URLs:${COLORS.reset}`);
    console.log(`   Frontend:    ${COLORS.bright}http://localhost:5173${COLORS.reset}`);
    console.log(`   API:         ${COLORS.bright}http://localhost:8084${COLORS.reset}`);
    console.log(`   API Health:  ${COLORS.bright}http://localhost:8084/api/health${COLORS.reset}`);
    console.log(`   UnoCSS:      ${COLORS.bright}http://localhost:5173/__unocss/${COLORS.reset}`);
    console.log(`   Monitor WS:  ${COLORS.bright}ws://localhost:8085${COLORS.reset}`);
    console.log();
    console.log(`${COLORS.yellow}Press Ctrl+C to stop all services${COLORS.reset}`);
    console.log();
  }
}

// Check for required dependencies
async function checkDependencies() {
  const required = ['concurrently', 'ws'];
  const missing = [];
  
  for (const dep of required) {
    try {
      await import(dep);
    } catch {
      missing.push(dep);
    }
  }
  
  if (missing.length > 0) {
    console.log('Installing required dependencies...');
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);
    
    await execAsync(`npm install --save-dev ${missing.join(' ')}`);
    console.log('Dependencies installed. Please run the command again.');
    process.exit(0);
  }
}

// Main execution
await checkDependencies();
const env = new DevEnvironment();
env.start();
