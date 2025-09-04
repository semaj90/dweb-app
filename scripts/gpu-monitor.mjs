#!/usr/bin/env node
// GPU Monitor & FlashAttention Integration for Legal AI Platform
import { spawn, exec } from 'node:child_process';
import { promisify } from 'node:util';
import { writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

const execAsync = promisify(exec);
const log = (msg, ...rest) => console.log(`[gpu] ${msg}`, ...rest);
const warn = (msg, ...rest) => console.warn(`[gpu] WARN ${msg}`, ...rest);
const err = (msg, ...rest) => console.error(`[gpu] ERR ${msg}`, ...rest);

class GPUMonitorService {
  constructor() {
    this.monitoring = false;
    this.gpuInfo = null;
    this.processes = new Map();
    this.metrics = {
      gpu_utilization: 0,
      memory_used: 0,
      memory_total: 0,
      temperature: 0,
      power_draw: 0,
      processes: []
    };
    this.metricsHistory = [];
    this.maxHistorySize = 100;
  }

  async detectGPU() {
    try {
      log('🔍 Detecting NVIDIA GPU...');
      
      const { stdout } = await execAsync('nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader,nounits');
      
      const [name, memoryTotal, driverVersion, cudaVersion] = stdout.trim().split(', ');
      
      this.gpuInfo = {
        name: name.trim(),
        memoryTotal: parseInt(memoryTotal),
        driverVersion: driverVersion.trim(),
        cudaVersion: cudaVersion.trim(),
        detected: true
      };

      log(`✅ GPU Detected: ${this.gpuInfo.name}`);
      log(`📊 VRAM: ${this.gpuInfo.memoryTotal}MB`);
      log(`🔧 Driver: ${this.gpuInfo.driverVersion}`);
      log(`🔧 CUDA: ${this.gpuInfo.cudaVersion}`);

      // Check for FlashAttention compatibility
      await this.checkFlashAttentionSupport();

      return true;
    } catch (error) {
      warn('No NVIDIA GPU detected or nvidia-smi not available');
      this.gpuInfo = { detected: false };
      return false;
    }
  }

  async checkFlashAttentionSupport() {
    try {
      log('🔬 Checking FlashAttention support...');
      
      const computeCapability = await this.getComputeCapability();
      const flashAttentionSupported = computeCapability >= 7.0; // Requires SM 7.0+
      
      this.gpuInfo.flashAttentionSupported = flashAttentionSupported;
      this.gpuInfo.computeCapability = computeCapability;

      if (flashAttentionSupported) {
        log(`✅ FlashAttention supported (Compute ${computeCapability})`);
        log('⚡ Ultra-fast attention mechanism available');
        log('🚀 GPU-optimized transformer inference enabled');
      } else {
        warn(`❌ FlashAttention not supported (Compute ${computeCapability} < 7.0)`);
        warn('💡 Consider upgrading to RTX 2070+ or newer for FlashAttention');
      }
    } catch (error) {
      warn('Could not determine FlashAttention support:', error.message);
    }
  }

  async getComputeCapability() {
    try {
      const { stdout } = await execAsync('nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits');
      return parseFloat(stdout.trim());
    } catch (error) {
      // Fallback: estimate based on GPU name
      const gpuName = this.gpuInfo?.name?.toLowerCase() || '';
      
      if (gpuName.includes('rtx 4090') || gpuName.includes('rtx 4080')) return 8.9;
      if (gpuName.includes('rtx 4070') || gpuName.includes('rtx 4060')) return 8.9;
      if (gpuName.includes('rtx 3090') || gpuName.includes('rtx 3080')) return 8.6;
      if (gpuName.includes('rtx 3070') || gpuName.includes('rtx 3060')) return 8.6;
      if (gpuName.includes('rtx 2080') || gpuName.includes('rtx 2070')) return 7.5;
      if (gpuName.includes('gtx 1080') || gpuName.includes('gtx 1070')) return 6.1;
      
      return 6.0; // Conservative fallback
    }
  }

  async collectMetrics() {
    if (!this.gpuInfo?.detected) {
      return this.metrics;
    }

    try {
      const { stdout } = await execAsync(`nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits`);
      
      const [utilization, memoryUsed, memoryTotal, temperature, powerDraw] = stdout.trim().split(', ').map(v => parseFloat(v.trim()));

      this.metrics = {
        gpu_utilization: utilization || 0,
        memory_used: memoryUsed || 0,
        memory_total: memoryTotal || 0,
        memory_usage_percent: memoryTotal ? Math.round((memoryUsed / memoryTotal) * 100) : 0,
        temperature: temperature || 0,
        power_draw: powerDraw || 0,
        timestamp: new Date().toISOString(),
        processes: await this.getGPUProcesses()
      };

      // Add FlashAttention specific metrics
      this.metrics.flashattention_enabled = this.gpuInfo?.flashAttentionSupported || false;
      this.metrics.compute_capability = this.gpuInfo?.computeCapability || 0;

      // Store in history
      this.metricsHistory.push({ ...this.metrics });
      if (this.metricsHistory.length > this.maxHistorySize) {
        this.metricsHistory.shift();
      }

      return this.metrics;
    } catch (error) {
      warn('Failed to collect GPU metrics:', error.message);
      return this.metrics;
    }
  }

  async getGPUProcesses() {
    try {
      const { stdout } = await execAsync(`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`);
      
      if (!stdout.trim()) return [];

      return stdout.trim().split('\n').map(line => {
        const [pid, name, memory] = line.split(', ');
        return {
          pid: parseInt(pid.trim()),
          name: name.trim(),
          memory: parseInt(memory.trim()) || 0
        };
      });
    } catch (error) {
      return [];
    }
  }

  async startMonitoring() {
    if (this.monitoring) {
      warn('GPU monitoring already active');
      return;
    }

    log('🚀 Starting GPU monitoring service...');

    const gpuDetected = await this.detectGPU();
    if (!gpuDetected) {
      warn('GPU monitoring disabled - no compatible GPU found');
      return;
    }

    this.monitoring = true;

    // Monitor GPU metrics every 3 seconds
    const monitorInterval = setInterval(async () => {
      if (!this.monitoring) {
        clearInterval(monitorInterval);
        return;
      }

      await this.collectMetrics();
      this.logStatus();
      this.saveMetricsFile();
    }, 3000);

    // Start GPU-accelerated services if configured
    await this.startGPUServices();

    log('✅ GPU monitoring service active');
    log('📊 Metrics collection interval: 3 seconds');
    log('💾 Metrics saved to: .vscode/gpu-metrics.json');
  }

  logStatus() {
    const m = this.metrics;
    
    if (!this.gpuInfo?.detected) {
      log('⚠️ No GPU detected');
      return;
    }

    log(`📊 GPU: ${m.gpu_utilization}% util | ${m.memory_used}/${m.memory_total}MB (${m.memory_usage_percent}%) | ${m.temperature}°C | ${m.power_draw}W`);

    if (m.processes.length > 0) {
      log(`🔧 Processes: ${m.processes.map(p => `${p.name}(${p.memory}MB)`).join(', ')}`);
    }

    // FlashAttention status
    if (this.gpuInfo?.flashAttentionSupported) {
      log('⚡ FlashAttention: READY');
    }
  }

  saveMetricsFile() {
    try {
      const metricsFile = resolve('.vscode/gpu-metrics.json');
      const data = {
        gpu_info: this.gpuInfo,
        current_metrics: this.metrics,
        history: this.metricsHistory.slice(-20), // Last 20 readings
        flashattention: {
          supported: this.gpuInfo?.flashAttentionSupported || false,
          compute_capability: this.gpuInfo?.computeCapability || 0,
          enabled_services: Array.from(this.processes.keys())
        },
        timestamp: new Date().toISOString()
      };
      
      writeFileSync(metricsFile, JSON.stringify(data, null, 2));
    } catch (error) {
      warn('Failed to save GPU metrics:', error.message);
    }
  }

  async startGPUServices() {
    log('🔥 Starting GPU-accelerated services...');

    const services = [
      {
        name: 'vllm-server',
        cmd: 'python',
        args: ['-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-7b-chat-hf', '--port', '8224'],
        condition: () => this.gpuInfo?.flashAttentionSupported
      },
      {
        name: 'gpu-indexer',
        cmd: resolve('../go-microservice/bin/gpu-indexer-service.exe'),
        args: ['--port=8225', '--gpu-layers=35'],
        condition: () => existsSync(resolve('../go-microservice/bin/gpu-indexer-service.exe'))
      },
      {
        name: 'tensor-server',
        cmd: resolve('../quic-services/quic-tensor-server.exe'),
        args: ['--port=8226', '--use-flash-attention'],
        condition: () => this.gpuInfo?.flashAttentionSupported && existsSync(resolve('../quic-services/quic-tensor-server.exe'))
      }
    ];

    let started = 0;

    for (const service of services) {
      if (service.condition && !service.condition()) {
        log(`⏭️ Skipping ${service.name} - requirements not met`);
        continue;
      }

      try {
        log(`🚀 Starting ${service.name}...`);
        
        const child = spawn(service.cmd, service.args, {
          stdio: 'inherit',
          shell: true,
          env: {
            ...process.env,
            CUDA_VISIBLE_DEVICES: '0',
            FLASH_ATTENTION_ENABLED: '1'
          }
        });

        this.processes.set(service.name, child);
        started++;

        child.on('exit', (code) => {
          warn(`${service.name} exited with code ${code}`);
          this.processes.delete(service.name);
        });

      } catch (error) {
        err(`Failed to start ${service.name}:`, error.message);
      }
    }

    log(`✅ GPU services started: ${started}`);
  }

  async stop() {
    log('🛑 Stopping GPU monitoring...');
    this.monitoring = false;

    // Stop GPU services
    for (const [name, process] of this.processes) {
      try {
        log(`Stopping ${name}...`);
        process.kill('SIGTERM');
      } catch (error) {
        warn(`Error stopping ${name}:`, error.message);
      }
    }

    this.processes.clear();
    log('✅ GPU monitoring stopped');
  }

  getStatus() {
    return {
      monitoring: this.monitoring,
      gpu_detected: this.gpuInfo?.detected || false,
      flashattention_supported: this.gpuInfo?.flashAttentionSupported || false,
      active_services: Array.from(this.processes.keys()),
      current_metrics: this.metrics,
      gpu_info: this.gpuInfo
    };
  }
}

// Global GPU monitor instance
const gpuMonitor = new GPUMonitorService();

// Graceful shutdown
function shutdown() {
  log('Received shutdown signal');
  gpuMonitor.stop().then(() => {
    process.exit(0);
  }).catch(() => {
    process.exit(1);
  });
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Unhandled errors
process.on('uncaughtException', (error) => {
  err('Uncaught exception:', error);
  shutdown();
});

process.on('unhandledRejection', (reason) => {
  err('Unhandled rejection:', reason);
  shutdown();
});

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  log('🔥 GPU Monitor & FlashAttention Service Starting...');
  
  gpuMonitor.startMonitoring().then(() => {
    log('🎉 GPU monitoring active with FlashAttention support');
    log('💡 Use Ctrl+C to stop monitoring');
    log('📊 View metrics: .vscode/gpu-metrics.json');
  }).catch(error => {
    err('Failed to start GPU monitoring:', error.message);
    process.exit(1);
  });
}

// Export for use by other modules
export { gpuMonitor as default, GPUMonitorService };