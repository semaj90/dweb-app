// monitor-lite.mjs
// Lightweight monitoring for development

import { exec } from 'child_process';
import { promisify } from 'util';
import http from 'http';
import net from 'net';

const execAsync = promisify(exec);

const SERVICES = {
  'Frontend': { port: 5173, url: 'http://localhost:5173' },
  'Go API': { port: 8084, url: 'http://localhost:8084/api/health' },
  'Redis': { port: 6379 },
  'Ollama': { port: 11434, url: 'http://localhost:11434/api/tags' },
  'PostgreSQL': { port: 5432 }
};

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m'
};

class Monitor {
  constructor() {
    this.startTime = Date.now();
    this.checkInterval = 5000; // 5 seconds
    this.stats = {
      checks: 0,
      services: {}
    };
  }

  async checkPort(port) {
    return new Promise((resolve) => {
      const client = new net.Socket();
      client.setTimeout(1000);
      
      client.on('connect', () => {
        client.destroy();
        resolve(true);
      });
      
      client.on('timeout', () => {
        client.destroy();
        resolve(false);
      });
      
      client.on('error', () => {
        resolve(false);
      });
      
      client.connect(port, '127.0.0.1');
    });
  }

  async checkHttp(url) {
    return new Promise((resolve) => {
      const timeout = setTimeout(() => resolve(false), 2000);
      
      http.get(url, (res) => {
        clearTimeout(timeout);
        resolve(res.statusCode >= 200 && res.statusCode < 400);
      }).on('error', () => {
        clearTimeout(timeout);
        resolve(false);
      });
    });
  }

  async checkService(name, config) {
    const portOpen = await this.checkPort(config.port);
    let healthy = portOpen;
    
    if (portOpen && config.url) {
      healthy = await this.checkHttp(config.url);
    }
    
    // Update stats
    if (!this.stats.services[name]) {
      this.stats.services[name] = { up: 0, down: 0 };
    }
    
    if (healthy) {
      this.stats.services[name].up++;
    } else {
      this.stats.services[name].down++;
    }
    
    return healthy;
  }

  async getMemoryUsage() {
    try {
      if (process.platform === 'win32') {
        const { stdout } = await execAsync('wmic process where name="node.exe" get WorkingSetSize');
        const lines = stdout.split('\n').filter(l => l.trim() && !l.includes('WorkingSetSize'));
        const totalBytes = lines.reduce((sum, line) => {
          const bytes = parseInt(line.trim());
          return isNaN(bytes) ? sum : sum + bytes;
        }, 0);
        return Math.round(totalBytes / 1024 / 1024);
      } else {
        const { stdout } = await execAsync("ps aux | grep node | awk '{sum+=$6} END {print sum/1024}'");
        return Math.round(parseFloat(stdout));
      }
    } catch {
      return 0;
    }
  }

  async getGPUUsage() {
    try {
      const { stdout } = await execAsync('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits');
      const [util, used, total] = stdout.trim().split(',').map(v => parseInt(v.trim()));
      return { utilization: util, memoryUsed: used, memoryTotal: total };
    } catch {
      return null;
    }
  }

  formatUptime() {
    const seconds = Math.floor((Date.now() - this.startTime) / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  formatStatus(healthy) {
    return healthy 
      ? `${COLORS.green}● UP${COLORS.reset}`
      : `${COLORS.red}○ DOWN${COLORS.reset}`;
  }

  async displayStatus() {
    console.clear();
    this.stats.checks++;
    
    // Header
    console.log(`${COLORS.cyan}${COLORS.bright}╔════════════════════════════════════════════════╗${COLORS.reset}`);
    console.log(`${COLORS.cyan}${COLORS.bright}║        LEGAL AI DEVELOPMENT MONITOR           ║${COLORS.reset}`);
    console.log(`${COLORS.cyan}${COLORS.bright}╚════════════════════════════════════════════════╝${COLORS.reset}`);
    console.log();
    
    // Uptime and checks
    console.log(`${COLORS.gray}Uptime: ${this.formatUptime()} | Checks: ${this.stats.checks}${COLORS.reset}`);
    console.log();
    
    // Services status
    console.log(`${COLORS.yellow}Services:${COLORS.reset}`);
    console.log(`${COLORS.gray}─────────────────────────────────────${COLORS.reset}`);
    
    for (const [name, config] of Object.entries(SERVICES)) {
      const healthy = await this.checkService(name, config);
      const status = this.formatStatus(healthy);
      const stats = this.stats.services[name];
      const uptime = stats ? Math.round((stats.up / (stats.up + stats.down)) * 100) : 0;
      
      console.log(`  ${status}  ${name.padEnd(12)} Port ${config.port.toString().padEnd(5)} ${COLORS.gray}(${uptime}% uptime)${COLORS.reset}`);
    }
    
    // System resources
    console.log();
    console.log(`${COLORS.yellow}Resources:${COLORS.reset}`);
    console.log(`${COLORS.gray}─────────────────────────────────────${COLORS.reset}`);
    
    const memoryMB = await this.getMemoryUsage();
    const memoryColor = memoryMB > 2000 ? COLORS.red : memoryMB > 1000 ? COLORS.yellow : COLORS.green;
    console.log(`  ${COLORS.blue}Memory:${COLORS.reset} ${memoryColor}${memoryMB}MB${COLORS.reset}`);
    
    const gpu = await this.getGPUUsage();
    if (gpu) {
      const gpuColor = gpu.utilization > 80 ? COLORS.red : gpu.utilization > 50 ? COLORS.yellow : COLORS.green;
      console.log(`  ${COLORS.magenta}GPU:${COLORS.reset} ${gpuColor}${gpu.utilization}%${COLORS.reset} | VRAM: ${gpu.memoryUsed}/${gpu.memoryTotal}MB`);
    }
    
    // URLs
    console.log();
    console.log(`${COLORS.yellow}Access URLs:${COLORS.reset}`);
    console.log(`${COLORS.gray}─────────────────────────────────────${COLORS.reset}`);
    console.log(`  Frontend:  ${COLORS.cyan}http://localhost:5173${COLORS.reset}`);
    console.log(`  API:       ${COLORS.cyan}http://localhost:8084${COLORS.reset}`);
    console.log(`  UnoCSS:    ${COLORS.cyan}http://localhost:5173/__unocss/${COLORS.reset}`);
    
    // Footer
    console.log();
    console.log(`${COLORS.gray}Press Ctrl+C to stop monitoring${COLORS.reset}`);
  }

  async start() {
    // Initial display
    await this.displayStatus();
    
    // Set up interval
    this.interval = setInterval(async () => {
      await this.displayStatus();
    }, this.checkInterval);
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      clearInterval(this.interval);
      console.log(`\n${COLORS.yellow}Monitor stopped${COLORS.reset}`);
      process.exit(0);
    });
  }
}

// Start monitoring
const monitor = new Monitor();
monitor.start();
