#!/usr/bin/env node
import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class SystemStartup {
    constructor() {
        this.processes = new Map();
        this.logDir = path.join(__dirname, '..', 'logs');
        this.startTime = new Date().toISOString();
        this.isShuttingDown = false;
    }

    async init() {
        await fs.mkdir(this.logDir, { recursive: true });
        const logFile = path.join(this.logDir, 'system-startup.log');
        await fs.writeFile(logFile, `=== System Startup: ${this.startTime} ===\n`);
        
        process.on('SIGINT', () => this.shutdown());
        process.on('SIGTERM', () => this.shutdown());
    }

    async log(msg) {
        const timestamp = new Date().toISOString();
        const logMsg = `[${timestamp}] ${msg}\n`;
        console.log(msg);
        await fs.appendFile(path.join(this.logDir, 'system-startup.log'), logMsg);
    }

    async startVite() {
        this.log('Starting Vite with enhanced configuration...');
        
        const viteProcess = spawn('npm', ['run', 'dev', '--', '--config', 'vite.config.enhanced.js'], {
            stdio: 'inherit',
            shell: true
        });

        this.processes.set('Vite', viteProcess);
        
        viteProcess.on('exit', (code) => {
            if (!this.isShuttingDown) {
                this.log(`Vite exited with code ${code}`);
            }
        });

        return viteProcess;
    }

    async shutdown() {
        if (this.isShuttingDown) return;
        this.isShuttingDown = true;
        
        this.log('Shutting down...');
        
        for (const [name, process] of this.processes) {
            process.kill('SIGTERM');
        }
        
        process.exit(0);
    }

    async start() {
        await this.init();
        this.log('Starting Enhanced Legal AI System...');
        
        await this.startVite();
        
        this.log('System started successfully');
        this.log('Frontend: http://localhost:5173');
        this.log('Press Ctrl+C to shutdown');
        
        process.stdin.resume();
    }
}

new SystemStartup().start().catch(console.error);
