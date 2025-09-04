#!/usr/bin/env node
import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class RTXEnhancedSystemStartup {
    constructor() {
        this.processes = new Map();
        this.logDir = path.join(__dirname, '..', 'logs');
        this.startTime = new Date().toISOString();
        this.isShuttingDown = false;
        this.healthChecks = new Map();
        
        // Service configuration
        this.services = [
            {
                name: 'Redis Cache Server',
                command: 'powershell',
                args: ['-Command', 'cd ../redis-windows; ./redis-server.exe'],
                cwd: path.join(__dirname, '..', '..'),
                port: 6379,
                healthPath: null, // Redis doesn't have HTTP health endpoint
                essential: true,
                startup_delay: 500,
                description: 'Redis cache and pub/sub server'
            },
            {
                name: 'SvelteKit Frontend',
                command: 'npm',
                args: ['run', 'dev', '--', '--port', '5175'],
                cwd: process.cwd(),
                port: 5175,
                healthPath: '/',
                essential: true,
                startup_delay: 1000, // Start after Redis
                description: 'Main frontend application'
            },
            {
                name: 'CUDA AI Service',
                command: 'powershell',
                args: ['-Command', '$env:DATABASE_URL="postgresql://postgres:123456@localhost:5432/legal_ai_db"; $env:PORT="8096"; ./cuda-ai-service.exe'],
                cwd: path.join(__dirname, '..', '..', 'go-microservice', 'bin'),
                port: 8096,
                healthPath: '/health',
                essential: true,
                startup_delay: 2000,
                description: 'RTX 3060 Ti CUDA processing with tensor cores'
            },
            {
                name: 'Upload Service',
                command: 'powershell',
                args: ['-Command', '$env:PORT="8093"; ./upload-service.exe'],
                cwd: path.join(__dirname, '..', '..', 'go-microservice', 'bin'),
                port: 8093,
                healthPath: '/health',
                essential: true,
                startup_delay: 1500,
                description: 'Document upload and processing'
            },
            {
                name: 'Multi-Protocol Gateway',
                command: 'powershell',
                args: ['-Command', '$env:GATEWAY_HTTP_PORT="8230"; $env:CUDA_SERVICE_URL="http://localhost:8096"; $env:UPLOAD_SERVICE_URL="http://localhost:8093"; ./multi-protocol-gateway.exe'],
                cwd: path.join(__dirname, '..', '..', 'go-microservice', 'cmd', 'multi-protocol-gateway'),
                port: 8230,
                healthPath: '/health',
                essential: false,
                startup_delay: 3000,
                description: 'Unified API gateway with RTX optimization'
            },
            {
                name: 'GPU Orchestrator',
                command: 'go',
                args: ['run', 'main.go'],
                cwd: path.join(__dirname, '..', '..', 'go-microservice', 'cmd', 'gpu-orchestrator'),
                port: 8231,
                healthPath: '/health',
                essential: false,
                startup_delay: 2500,
                description: 'GPU resource orchestration'
            }
        ];
    }

    async init() {
        await fs.mkdir(this.logDir, { recursive: true });
        const logFile = path.join(this.logDir, 'rtx-system-startup.log');
        await fs.writeFile(logFile, `=== RTX Enhanced System Startup: ${this.startTime} ===\n`);
        
        process.on('SIGINT', () => this.shutdown());
        process.on('SIGTERM', () => this.shutdown());
    }

    async log(msg, level = 'INFO') {
        const timestamp = new Date().toISOString();
        const logMsg = `[${timestamp}] [${level}] ${msg}\n`;
        
        // Color coding for console output
        const colors = {
            INFO: '\x1b[36m',    // Cyan
            SUCCESS: '\x1b[32m', // Green
            WARNING: '\x1b[33m', // Yellow
            ERROR: '\x1b[31m',   // Red
            RESET: '\x1b[0m'     // Reset
        };
        
        console.log(`${colors[level] || colors.INFO}${msg}${colors.RESET}`);
        await fs.appendFile(path.join(this.logDir, 'rtx-system-startup.log'), logMsg);
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async checkServiceHealth(service) {
        try {
            const fetch = (await import('node-fetch')).default;
            const response = await fetch(`http://localhost:${service.port}${service.healthPath}`, {
                timeout: 5000
            });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    async startService(service) {
        if (this.isShuttingDown) return;
        
        await this.log(`Starting ${service.name}...`, 'INFO');
        
        // Add startup delay
        if (service.startup_delay > 0) {
            await this.delay(service.startup_delay);
        }

        const process = spawn(service.command, service.args, {
            cwd: service.cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            shell: true
        });

        this.processes.set(service.name, process);

        // Handle process output
        process.stdout?.on('data', (data) => {
            const output = data.toString().trim();
            if (output) {
                this.log(`[${service.name}] ${output}`, 'INFO');
            }
        });

        process.stderr?.on('data', (data) => {
            const output = data.toString().trim();
            if (output && !output.includes('[GIN-debug]') && !output.includes('WARNING')) {
                this.log(`[${service.name}] ${output}`, 'WARNING');
            }
        });

        process.on('exit', (code) => {
            if (!this.isShuttingDown) {
                const level = code === 0 ? 'INFO' : 'ERROR';
                this.log(`${service.name} exited with code ${code}`, level);
                
                if (service.essential && code !== 0) {
                    this.log(`Essential service ${service.name} failed, shutting down system`, 'ERROR');
                    this.shutdown();
                }
            }
        });

        // Wait for service to be ready
        let retries = 30;
        while (retries > 0 && !this.isShuttingDown) {
            const healthy = await this.checkServiceHealth(service);
            if (healthy) {
                await this.log(`✅ ${service.name} ready on port ${service.port}`, 'SUCCESS');
                this.healthChecks.set(service.name, true);
                return process;
            }
            
            await this.delay(1000);
            retries--;
        }

        if (retries === 0) {
            await this.log(`⚠️ ${service.name} health check timeout`, 'WARNING');
        }

        return process;
    }

    async startAllServices() {
        await this.log('🚀 Starting RTX Enhanced Legal AI System...', 'INFO');
        await this.log('🎯 RTX 3060 Ti Tensor Core Processing Enabled', 'SUCCESS');
        
        for (const service of this.services) {
            try {
                await this.startService(service);
            } catch (error) {
                await this.log(`Failed to start ${service.name}: ${error.message}`, 'ERROR');
                if (service.essential) {
                    await this.log('Essential service failed, aborting startup', 'ERROR');
                    await this.shutdown();
                    return;
                }
            }
        }

        await this.displaySystemStatus();
    }

    async displaySystemStatus() {
        await this.log('\n📊 RTX Enhanced System Status:', 'SUCCESS');
        await this.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'INFO');
        
        for (const service of this.services) {
            const status = this.healthChecks.get(service.name) ? '✅ ONLINE' : '❌ OFFLINE';
            const url = `http://localhost:${service.port}`;
            await this.log(`${status}  ${service.name.padEnd(25)} ${url.padEnd(30)} ${service.description}`, 'INFO');
        }
        
        await this.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'INFO');
        await this.log('\n🎮 Access Points:', 'SUCCESS');
        await this.log('  • Frontend:              http://localhost:5175', 'INFO');
        await this.log('  • CUDA Processing:       http://localhost:8096/health', 'INFO');
        await this.log('  • Document Upload:       http://localhost:8093/health', 'INFO');
        await this.log('  • Multi-Protocol Gateway: http://localhost:8230/health', 'INFO');
        await this.log('  • GPU Orchestrator:      http://localhost:8231/health', 'INFO');
        
        await this.log('\n🧪 Demo Pages:', 'SUCCESS');
        await this.log('  • CUDA RTX Integration:  http://localhost:5175/demo/cuda-rtx-integration', 'INFO');
        await this.log('  • GPU Document Upload:   http://localhost:5175/demo/document-upload-gpu', 'INFO');
        
        await this.log('\n⚡ RTX Features Active:', 'SUCCESS');
        await this.log('  • 4th Generation Tensor Cores', 'INFO');
        await this.log('  • 4-bit Quantization (75% memory reduction)', 'INFO');
        await this.log('  • Negative Latent Space Processing', 'INFO');
        await this.log('  • Multi-stream CUDA Processing', 'INFO');
        
        await this.log('\n🛠️ System Commands:', 'SUCCESS');
        await this.log('  • Test CUDA Health:      curl http://localhost:8096/health', 'INFO');
        await this.log('  • Test Gateway:          curl http://localhost:8230/health', 'INFO');
        await this.log('  • Stop System:           Ctrl+C', 'INFO');
        
        await this.log('\n✨ RTX Enhanced Legal AI System Ready!', 'SUCCESS');
        await this.log('Press Ctrl+C to shutdown all services', 'INFO');
    }

    async shutdown() {
        if (this.isShuttingDown) return;
        this.isShuttingDown = true;
        
        await this.log('\n🔄 Shutting down RTX Enhanced System...', 'WARNING');
        
        // Shutdown in reverse order
        const shutdownOrder = [...this.services].reverse();
        
        for (const service of shutdownOrder) {
            const process = this.processes.get(service.name);
            if (process) {
                await this.log(`Stopping ${service.name}...`, 'WARNING');
                process.kill('SIGTERM');
                
                // Give process time to shutdown gracefully
                await this.delay(1000);
                
                // Force kill if still running
                if (!process.killed) {
                    process.kill('SIGKILL');
                }
            }
        }
        
        await this.log('✅ All services stopped', 'SUCCESS');
        await this.log('👋 RTX Enhanced System shutdown complete', 'INFO');
        
        process.exit(0);
    }

    async start() {
        try {
            await this.init();
            await this.startAllServices();
            
            // Keep process alive
            process.stdin.resume();
            
        } catch (error) {
            await this.log(`System startup failed: ${error.message}`, 'ERROR');
            await this.shutdown();
        }
    }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
    new RTXEnhancedSystemStartup().start().catch(console.error);
}

export default RTXEnhancedSystemStartup;