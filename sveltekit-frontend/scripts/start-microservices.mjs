// start-microservices.mjs
// Orchestration script using ClusterMulticoreManager for dynamic, collision-free port assignment
import { resolve } from 'path';
import { platform } from 'os';
import { spawn } from 'child_process';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { ClusterMulticoreManager } from './cluster-multicore-manager.mjs';

const manager = new ClusterMulticoreManager();

const portRanges = {
    frontend:      { start: 3000, end: 3049 },
    nodeApi:       { start: 3050, end: 3099 },
    wsFanout:      { start: 8080, end: 8089 },
    gpuWorker:     { start: 8090, end: 8099 },
    wasmWorker:    { start: 8100, end: 8109 },
    goLlama:       { start: 8110, end: 8129 },
    quicGateway:   { start: 8130, end: 8139 },
    loadBalancer:  { start: 8140, end: 8145 }
};

const bin = (name) => {
    const ext = platform() === 'win32' ? '.exe' : '';
    return resolve(`go-microservices/bin/${name}${ext}`);
};

function spawnService({ name, cmd, args = [], port }) {
    try {
        const proc = spawn(cmd, args, { stdio: 'inherit' });
        proc.on('exit', (code, signal) => {
            console.warn(`${name} exited with code=${code} signal=${signal}`);
            // TODO: implement backoff/restart policy here if desired
        });
        proc.on('error', (err) => {
            console.error(`Failed to start ${name}:`, err);
        });
        return proc;
    } catch (err) {
        console.error(`spawnService error for ${name}:`, err);
        return null;
    }
}

(async () => {
    const ports = await manager.reservePorts(portRanges);

    const services = [
        { name: 'frontend', cmd: 'npx', args: ['tsx', 'src/lib/api/server.ts', `--port=${ports.frontend}`], port: ports.frontend, type: 'node' },
        { name: 'node-api', cmd: 'npx', args: ['tsx', 'src/lib/api/server.ts', `--port=${ports.nodeApi}`], port: ports.nodeApi, type: 'node' },
        { name: 'gpu-worker', cmd: 'npx', args: ['tsx', 'src/lib/services/gpu-worker.ts', `--port=${ports.gpuWorker}`], port: ports.gpuWorker, type: 'worker' },
        { name: 'wasm-worker', cmd: 'npx', args: ['tsx', 'src/lib/services/wasm-worker.ts', `--port=${ports.wasmWorker}`], port: ports.wasmWorker, type: 'worker' },
        { name: 'go-llama', cmd: bin('go-llama'), args: [`--port=${ports.goLlama}`], port: ports.goLlama, type: 'go' },
        { name: 'quic-gateway', cmd: bin('quic-gateway'), args: [`--port=${ports.quicGateway}`], port: ports.quicGateway, type: 'go' },
        { name: 'ws-fanout', cmd: 'npx', args: ['tsx', 'src/lib/services/ws-fanout-service.ts', `--port=${ports.wsFanout}`], port: ports.wsFanout, type: 'node' },
        { name: 'load-balancer', cmd: bin('load-balancer'), args: [`--port=${ports.loadBalancer}`], port: ports.loadBalancer, type: 'go' }
    ];

    // Persist port assignment so other processes (API/frontend) can read cluster status
    const outDir = '.vscode';
    if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
    writeFileSync(`${outDir}/cluster-status.json`, JSON.stringify({ ts: Date.now(), ports }, null, 2), 'utf8');

    // Spawn services
    const procs = new Map();
    for (const svc of services) {
        console.log(`Spawning ${svc.name} on port ${svc.port}: ${svc.cmd} ${svc.args?.join(' ') || ''}`);
        const p = spawnService(svc);
        if (p) procs.set(svc.name, p);
    }

    // Graceful shutdown: forward signals to children
    const shutdown = (sig) => {
        console.warn(`Received ${sig}, shutting down children...`);
        for (const [name, p] of procs.entries()) {
            try {
                p.kill('SIGTERM');
            } catch (e) {
                console.warn(`Error killing ${name}:`, e);
            }
        }
        // allow some time for children to exit
        setTimeout(() => process.exit(0), 2000);
    };

    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
})();
