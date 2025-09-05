const { Service } = require('node-windows');
const path = require('path');
const net = require('net');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const { EventEmitter } = require('events');

/**
 * Legal AI Windows Service
 * Manages the entire Legal AI system as a Windows service with IPC/gRPC coordination
 */
class LegalAIWindowsService extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      serviceName: 'LegalAISystem',
      serviceDescription: 'Legal AI Document Processing and Analysis System',
      scriptPath: path.join(__dirname, 'service-manager.js'),
      
      // IPC Configuration
      ipcSocketPath: '\\\\.\\pipe\\legal-ai-ipc',
      ipcPort: 9876,
      
      // gRPC Configuration
      grpcPort: 50051,
      grpcHost: '127.0.0.1',
      
      // Service Dependencies
      dependencies: [
        'postgresql-x64-17',
        'Winmgmt', // WMI
        'EventLog'
      ],
      
      // Process Management
      processes: {
        'sveltekit-frontend': {
          script: 'npm run dev',
          cwd: path.resolve(__dirname, '../sveltekit-frontend'),
          env: { PORT: '5173' },
          restart: true,
          maxRestarts: 5
        },
        'node-cluster': {
          script: 'node cluster-manager.js',
          cwd: path.resolve(__dirname, '../node-cluster'),
          env: { CLUSTER_WORKERS: '4' },
          restart: true,
          maxRestarts: 3
        },
        'go-kratos-service': {
          script: './kratos-server.exe',
          cwd: path.resolve(__dirname, '../go-services/cmd/kratos-server'),
          env: { PORT: '8080' },
          restart: true,
          maxRestarts: 3
        },
        'nats-coordinator': {
          script: './nats-coordinator.exe',
          cwd: path.resolve(__dirname, '../message-queue'),
          env: { NATS_PORT: '4222' },
          restart: true,
          maxRestarts: 3
        }
      },
      
      // Monitoring
      healthCheckInterval: 30000,
      logRotationSize: 100 * 1024 * 1024, // 100MB
      maxLogFiles: 10
    };
    
    this.service = null;
    this.ipcServer = null;
    this.grpcServer = null;
    this.processes = new Map();
    this.isRunning = false;
    this.startTime = null;
    
    this.init();
  }
  
  init() {
    console.log('[LEGAL-AI-SERVICE] Initializing Windows Service Manager');
    
    // Create the Windows service definition
    this.createService();
    
    // Setup IPC communication
    this.setupIPC();
    
    // Setup gRPC server
    this.setupGRPC();
    
    // Setup process monitoring
    this.setupProcessMonitoring();
    
    // Setup service event handlers
    this.setupServiceHandlers();
  }
  
  createService() {
    this.service = new Service({
      name: this.config.serviceName,
      description: this.config.serviceDescription,
      script: this.config.scriptPath,
      nodeOptions: [
        '--max-old-space-size=4096',
        '--expose-gc'
      ],
      env: {
        NODE_ENV: 'production',
        SERVICE_MODE: 'true',
        LOG_LEVEL: 'info'
      },
      dependencies: this.config.dependencies,
      wait: 2,
      grow: 0.5,
      maxRestarts: 10
    });
    
    console.log(`[LEGAL-AI-SERVICE] Service definition created: ${this.config.serviceName}`);
  }
  
  setupIPC() {
    // Named pipe IPC server for Windows
    this.ipcServer = net.createServer((socket) => {
      console.log('[LEGAL-AI-SERVICE] IPC client connected');
      
      socket.on('data', (data) => {
        this.handleIPCMessage(socket, data);
      });
      
      socket.on('error', (error) => {
        console.error('[LEGAL-AI-SERVICE] IPC socket error:', error);
      });
      
      socket.on('close', () => {
        console.log('[LEGAL-AI-SERVICE] IPC client disconnected');
      });
    });
    
    this.ipcServer.on('error', (error) => {
      console.error('[LEGAL-AI-SERVICE] IPC server error:', error);
    });
  }
  
  setupGRPC() {
    // Load protobuf definitions
    const packageDefinition = protoLoader.loadSync('./protos/legal-ai-service.proto', {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true
    });
    
    const serviceProto = grpc.loadPackageDefinition(packageDefinition).legalai.service;
    
    this.grpcServer = new grpc.Server();
    
    // Implement service methods
    this.grpcServer.addService(serviceProto.LegalAIService.service, {
      StartProcess: this.startProcess.bind(this),
      StopProcess: this.stopProcess.bind(this),
      RestartProcess: this.restartProcess.bind(this),
      GetProcessStatus: this.getProcessStatus.bind(this),
      GetSystemHealth: this.getSystemHealth.bind(this),
      ExecuteCommand: this.executeCommand.bind(this),
      GetMetrics: this.getMetrics.bind(this),
      SetConfiguration: this.setConfiguration.bind(this)
    });
    
    console.log('[LEGAL-AI-SERVICE] gRPC server configured');
  }
  
  setupProcessMonitoring() {
    // Health check interval
    setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);
    
    // Resource monitoring
    setInterval(() => {
      this.monitorResources();
    }, 60000); // Every minute
  }
  
  setupServiceHandlers() {
    // Service installation handlers
    this.service.on('install', () => {
      console.log('[LEGAL-AI-SERVICE] Service installed successfully');
      this.service.start();
    });
    
    this.service.on('alreadyinstalled', () => {
      console.log('[LEGAL-AI-SERVICE] Service already installed');
    });
    
    this.service.on('invalidinstallation', () => {
      console.error('[LEGAL-AI-SERVICE] Invalid installation detected');
    });
    
    // Service runtime handlers
    this.service.on('start', () => {
      console.log('[LEGAL-AI-SERVICE] Service started');
      this.startTime = Date.now();
      this.isRunning = true;
      this.startAllProcesses();
    });
    
    this.service.on('stop', () => {
      console.log('[LEGAL-AI-SERVICE] Service stopped');
      this.isRunning = false;
      this.stopAllProcesses();
    });
    
    this.service.on('error', (error) => {
      console.error('[LEGAL-AI-SERVICE] Service error:', error);
      this.emit('error', error);
    });
  }
  
  async startAllProcesses() {
    console.log('[LEGAL-AI-SERVICE] Starting all managed processes');
    
    try {
      // Start IPC server
      await this.startIPCServer();
      
      // Start gRPC server
      await this.startGRPCServer();
      
      // Start managed processes in dependency order
      const startOrder = [
        'nats-coordinator',    // Message queue first
        'go-kratos-service',   // Core services
        'node-cluster',        // Node.js cluster
        'sveltekit-frontend'   // Frontend last
      ];
      
      for (const processName of startOrder) {
        await this.startManagedProcess(processName);
        // Wait between starts to allow proper initialization
        await this.sleep(2000);
      }
      
      console.log('[LEGAL-AI-SERVICE] All processes started successfully');
      this.emit('allProcessesStarted');
      
    } catch (error) {
      console.error('[LEGAL-AI-SERVICE] Failed to start processes:', error);
      this.emit('error', error);
    }
  }
  
  async stopAllProcesses() {
    console.log('[LEGAL-AI-SERVICE] Stopping all managed processes');
    
    try {
      // Stop in reverse order
      const stopOrder = [
        'sveltekit-frontend',
        'node-cluster',
        'go-kratos-service',
        'nats-coordinator'
      ];
      
      for (const processName of stopOrder) {
        await this.stopManagedProcess(processName);
      }
      
      // Stop servers
      if (this.grpcServer) {
        await new Promise((resolve) => {
          this.grpcServer.tryShutdown(resolve);
        });
      }
      
      if (this.ipcServer) {
        this.ipcServer.close();
      }
      
      console.log('[LEGAL-AI-SERVICE] All processes stopped');
      this.emit('allProcessesStopped');
      
    } catch (error) {
      console.error('[LEGAL-AI-SERVICE] Error stopping processes:', error);
    }
  }
  
  async startIPCServer() {
    return new Promise((resolve, reject) => {
      this.ipcServer.listen(this.config.ipcSocketPath, (error) => {
        if (error) {
          reject(error);
        } else {
          console.log(`[LEGAL-AI-SERVICE] IPC server listening on ${this.config.ipcSocketPath}`);
          resolve();
        }
      });
    });
  }
  
  async startGRPCServer() {
    return new Promise((resolve, reject) => {
      const address = `${this.config.grpcHost}:${this.config.grpcPort}`;
      
      this.grpcServer.bindAsync(
        address,
        grpc.ServerCredentials.createInsecure(),
        (error, port) => {
          if (error) {
            reject(error);
          } else {
            this.grpcServer.start();
            console.log(`[LEGAL-AI-SERVICE] gRPC server listening on ${address}`);
            resolve();
          }
        }
      );
    });
  }
  
  async startManagedProcess(processName) {
    const processConfig = this.config.processes[processName];
    if (!processConfig) {
      throw new Error(`Unknown process: ${processName}`);
    }
    
    console.log(`[LEGAL-AI-SERVICE] Starting process: ${processName}`);
    
    const spawn = require('child_process').spawn;
    const command = processConfig.script.split(' ')[0];
    const args = processConfig.script.split(' ').slice(1);
    
    const child = spawn(command, args, {
      cwd: processConfig.cwd,
      env: { ...process.env, ...processConfig.env },
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: false,
      windowsHide: true
    });
    
    // Store process information
    this.processes.set(processName, {
      process: child,
      config: processConfig,
      startTime: Date.now(),
      restartCount: 0,
      status: 'starting',
      lastError: null,
      lastOutput: []
    });
    
    // Handle process events
    child.on('spawn', () => {
      console.log(`[LEGAL-AI-SERVICE] Process spawned: ${processName} (PID: ${child.pid})`);
      const processInfo = this.processes.get(processName);
      processInfo.status = 'running';
      processInfo.pid = child.pid;
    });
    
    child.on('exit', (code, signal) => {
      console.log(`[LEGAL-AI-SERVICE] Process exited: ${processName} (code: ${code}, signal: ${signal})`);
      const processInfo = this.processes.get(processName);
      processInfo.status = 'stopped';
      processInfo.exitCode = code;
      processInfo.exitSignal = signal;
      
      // Auto-restart if configured
      if (processConfig.restart && code !== 0 && processInfo.restartCount < processConfig.maxRestarts) {
        console.log(`[LEGAL-AI-SERVICE] Auto-restarting process: ${processName}`);
        setTimeout(() => {
          this.restartManagedProcess(processName);
        }, 5000);
      }
    });
    
    child.on('error', (error) => {
      console.error(`[LEGAL-AI-SERVICE] Process error: ${processName}:`, error);
      const processInfo = this.processes.get(processName);
      processInfo.lastError = error.message;
      processInfo.status = 'error';
    });
    
    // Capture output
    child.stdout.on('data', (data) => {
      const processInfo = this.processes.get(processName);
      const output = data.toString();
      processInfo.lastOutput.push({ timestamp: Date.now(), type: 'stdout', data: output });
      
      // Keep only last 100 output entries
      if (processInfo.lastOutput.length > 100) {
        processInfo.lastOutput = processInfo.lastOutput.slice(-100);
      }
    });
    
    child.stderr.on('data', (data) => {
      const processInfo = this.processes.get(processName);
      const output = data.toString();
      processInfo.lastOutput.push({ timestamp: Date.now(), type: 'stderr', data: output });
      
      if (processInfo.lastOutput.length > 100) {
        processInfo.lastOutput = processInfo.lastOutput.slice(-100);
      }
    });
    
    // Wait for process to be ready (simplified check)
    await this.sleep(1000);
    
    const processInfo = this.processes.get(processName);
    if (processInfo.status === 'running') {
      console.log(`[LEGAL-AI-SERVICE] Process started successfully: ${processName}`);
    } else {
      throw new Error(`Failed to start process: ${processName}`);
    }
  }
  
  async stopManagedProcess(processName) {
    const processInfo = this.processes.get(processName);
    if (!processInfo || !processInfo.process) {
      console.log(`[LEGAL-AI-SERVICE] Process not found or already stopped: ${processName}`);
      return;
    }
    
    console.log(`[LEGAL-AI-SERVICE] Stopping process: ${processName}`);
    
    try {
      // Graceful shutdown first
      processInfo.process.kill('SIGTERM');
      
      // Wait for graceful shutdown
      await this.sleep(5000);
      
      // Force kill if still running
      if (!processInfo.process.killed) {
        processInfo.process.kill('SIGKILL');
      }
      
      processInfo.status = 'stopped';
      console.log(`[LEGAL-AI-SERVICE] Process stopped: ${processName}`);
      
    } catch (error) {
      console.error(`[LEGAL-AI-SERVICE] Error stopping process ${processName}:`, error);
    }
  }
  
  async restartManagedProcess(processName) {
    console.log(`[LEGAL-AI-SERVICE] Restarting process: ${processName}`);
    
    const processInfo = this.processes.get(processName);
    if (processInfo) {
      processInfo.restartCount++;
    }
    
    await this.stopManagedProcess(processName);
    await this.sleep(2000);
    await this.startManagedProcess(processName);
  }
  
  handleIPCMessage(socket, data) {
    try {
      const message = JSON.parse(data.toString());
      console.log('[LEGAL-AI-SERVICE] IPC message received:', message.type);
      
      switch (message.type) {
        case 'ping':
          socket.write(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
          break;
          
        case 'status':
          socket.write(JSON.stringify({
            type: 'status_response',
            data: this.getServiceStatus()
          }));
          break;
          
        case 'restart_process':
          this.restartManagedProcess(message.processName);
          socket.write(JSON.stringify({
            type: 'restart_initiated',
            processName: message.processName
          }));
          break;
          
        case 'stop_service':
          socket.write(JSON.stringify({ type: 'shutdown_initiated' }));
          this.stopAllProcesses();
          break;
          
        case 'get_logs':
          socket.write(JSON.stringify({
            type: 'logs_response',
            data: this.getProcessLogs(message.processName)
          }));
          break;
          
        default:
          socket.write(JSON.stringify({
            type: 'error',
            message: `Unknown message type: ${message.type}`
          }));
      }
      
    } catch (error) {
      console.error('[LEGAL-AI-SERVICE] IPC message handling error:', error);
      socket.write(JSON.stringify({
        type: 'error',
        message: error.message
      }));
    }
  }
  
  // gRPC Service Implementation
  startProcess(call, callback) {
    const { processName } = call.request;
    
    this.startManagedProcess(processName)
      .then(() => {
        callback(null, {
          success: true,
          message: `Process ${processName} started successfully`
        });
      })
      .catch((error) => {
        callback(null, {
          success: false,
          message: error.message
        });
      });
  }
  
  stopProcess(call, callback) {
    const { processName } = call.request;
    
    this.stopManagedProcess(processName)
      .then(() => {
        callback(null, {
          success: true,
          message: `Process ${processName} stopped successfully`
        });
      })
      .catch((error) => {
        callback(null, {
          success: false,
          message: error.message
        });
      });
  }
  
  restartProcess(call, callback) {
    const { processName } = call.request;
    
    this.restartManagedProcess(processName)
      .then(() => {
        callback(null, {
          success: true,
          message: `Process ${processName} restarted successfully`
        });
      })
      .catch((error) => {
        callback(null, {
          success: false,
          message: error.message
        });
      });
  }
  
  getProcessStatus(call, callback) {
    const { processName } = call.request;
    const processInfo = this.processes.get(processName);
    
    if (!processInfo) {
      callback(null, {
        exists: false,
        status: 'not_found'
      });
      return;
    }
    
    callback(null, {
      exists: true,
      status: processInfo.status,
      pid: processInfo.pid || 0,
      startTime: processInfo.startTime,
      restartCount: processInfo.restartCount,
      lastError: processInfo.lastError || ''
    });
  }
  
  getSystemHealth(call, callback) {
    const health = {
      serviceStatus: this.isRunning ? 'running' : 'stopped',
      uptime: this.startTime ? Date.now() - this.startTime : 0,
      processCount: this.processes.size,
      processes: Array.from(this.processes.entries()).map(([name, info]) => ({
        name: name,
        status: info.status,
        pid: info.pid || 0,
        restartCount: info.restartCount
      })),
      systemResources: this.getSystemResources(),
      timestamp: Date.now()
    };
    
    callback(null, health);
  }
  
  executeCommand(call, callback) {
    const { command, args } = call.request;
    
    // Security: Only allow specific commands
    const allowedCommands = ['status', 'restart', 'logs', 'metrics'];
    
    if (!allowedCommands.includes(command)) {
      callback(null, {
        success: false,
        output: '',
        error: `Command not allowed: ${command}`
      });
      return;
    }
    
    try {
      let output = '';
      
      switch (command) {
        case 'status':
          output = JSON.stringify(this.getServiceStatus(), null, 2);
          break;
        case 'restart':
          if (args && args.length > 0) {
            this.restartManagedProcess(args[0]);
            output = `Restart initiated for ${args[0]}`;
          } else {
            output = 'Process name required for restart';
          }
          break;
        case 'logs':
          output = JSON.stringify(this.getProcessLogs(args[0]), null, 2);
          break;
        case 'metrics':
          output = JSON.stringify(this.getMetricsData(), null, 2);
          break;
      }
      
      callback(null, {
        success: true,
        output: output,
        error: ''
      });
      
    } catch (error) {
      callback(null, {
        success: false,
        output: '',
        error: error.message
      });
    }
  }
  
  getMetrics(call, callback) {
    const metrics = this.getMetricsData();
    callback(null, metrics);
  }
  
  setConfiguration(call, callback) {
    const { key, value } = call.request;
    
    try {
      // Update configuration (simplified)
      if (key.startsWith('process.')) {
        const processName = key.split('.')[1];
        const configKey = key.split('.')[2];
        
        if (this.config.processes[processName]) {
          this.config.processes[processName][configKey] = value;
        }
      }
      
      callback(null, {
        success: true,
        message: `Configuration updated: ${key} = ${value}`
      });
      
    } catch (error) {
      callback(null, {
        success: false,
        message: error.message
      });
    }
  }
  
  // Utility Methods
  performHealthChecks() {
    this.processes.forEach((processInfo, processName) => {
      if (processInfo.status === 'running' && processInfo.process) {
        // Check if process is still alive
        try {
          process.kill(processInfo.pid, 0); // Signal 0 checks if process exists
        } catch (error) {
          console.warn(`[LEGAL-AI-SERVICE] Process ${processName} appears to be dead`);
          processInfo.status = 'dead';
          
          // Auto-restart if configured
          if (processInfo.config.restart) {
            this.restartManagedProcess(processName);
          }
        }
      }
    });
  }
  
  monitorResources() {
    const usage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    // Log resource usage
    console.log(`[LEGAL-AI-SERVICE] Resource usage: Memory ${Math.round(usage.heapUsed / 1024 / 1024)}MB, CPU ${cpuUsage.user + cpuUsage.system}μs`);
    
    // Check for high memory usage
    if (usage.heapUsed > 1024 * 1024 * 1024) { // 1GB
      console.warn('[LEGAL-AI-SERVICE] High memory usage detected');
      
      // Trigger garbage collection
      if (global.gc) {
        global.gc();
      }
    }
  }
  
  getServiceStatus() {
    return {
      serviceName: this.config.serviceName,
      isRunning: this.isRunning,
      startTime: this.startTime,
      uptime: this.startTime ? Date.now() - this.startTime : 0,
      processCount: this.processes.size,
      processes: Array.from(this.processes.entries()).map(([name, info]) => ({
        name: name,
        status: info.status,
        pid: info.pid,
        startTime: info.startTime,
        restartCount: info.restartCount,
        lastError: info.lastError
      })),
      timestamp: Date.now()
    };
  }
  
  getProcessLogs(processName) {
    const processInfo = this.processes.get(processName);
    if (!processInfo) {
      return { error: 'Process not found' };
    }
    
    return {
      processName: processName,
      logs: processInfo.lastOutput.slice(-50), // Last 50 log entries
      status: processInfo.status
    };
  }
  
  getSystemResources() {
    const os = require('os');
    
    return {
      platform: os.platform(),
      arch: os.arch(),
      totalMemory: os.totalmem(),
      freeMemory: os.freemem(),
      cpuCount: os.cpus().length,
      uptime: os.uptime(),
      loadAvg: os.loadavg()
    };
  }
  
  getMetricsData() {
    return {
      service: {
        uptime: this.startTime ? Date.now() - this.startTime : 0,
        processCount: this.processes.size,
        totalRestarts: Array.from(this.processes.values())
          .reduce((sum, info) => sum + info.restartCount, 0)
      },
      processes: Array.from(this.processes.entries()).map(([name, info]) => ({
        name: name,
        status: info.status,
        uptime: Date.now() - info.startTime,
        restartCount: info.restartCount,
        memoryUsage: info.process ? info.process.memoryUsage?.() : null
      })),
      system: this.getSystemResources(),
      timestamp: Date.now()
    };
  }
  
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  // Public API Methods
  install() {
    console.log('[LEGAL-AI-SERVICE] Installing Windows service...');
    this.service.install();
  }
  
  uninstall() {
    console.log('[LEGAL-AI-SERVICE] Uninstalling Windows service...');
    this.service.uninstall();
  }
  
  start() {
    console.log('[LEGAL-AI-SERVICE] Starting Windows service...');
    this.service.start();
  }
  
  stop() {
    console.log('[LEGAL-AI-SERVICE] Stopping Windows service...');
    this.service.stop();
  }
  
  restart() {
    console.log('[LEGAL-AI-SERVICE] Restarting Windows service...');
    this.service.restart();
  }
}

module.exports = LegalAIWindowsService;

// CLI usage
if (require.main === module) {
  const command = process.argv[2];
  const service = new LegalAIWindowsService();
  
  switch (command) {
    case 'install':
      service.install();
      break;
    case 'uninstall':
      service.uninstall();
      break;
    case 'start':
      service.start();
      break;
    case 'stop':
      service.stop();
      break;
    case 'restart':
      service.restart();
      break;
    default:
      console.log('Usage: node legal-ai-service.js [install|uninstall|start|stop|restart]');
      console.log('Legal AI Windows Service Manager');
  }
}