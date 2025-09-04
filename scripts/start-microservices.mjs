#!/usr/bin/env node
// Multi-service spawn orchestrator with Vite-style dynamic port discovery
import { spawn } from 'node:child_process';
import { existsSync, writeFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import net from 'node:net';

const log = (msg, ...rest) => console.log(`[micro] ${msg}`, ...rest);
const warn = (msg, ...rest) => console.warn(`[micro] WARN ${msg}`, ...rest);
const err  = (msg, ...rest) => console.error(`[micro] ERR ${msg}`, ...rest);

// Dynamic port discovery like Vite
// Track allocated ports to prevent collisions
const allocatedPorts = new Set();

async function findAvailablePort(preferredPort, maxAttempts = 20) {
	for (let i = 0; i < maxAttempts; i++) {
		const port = preferredPort + i;
		
		// Skip if port is already allocated to another service
		if (allocatedPorts.has(port)) {
			continue;
		}
		
		try {
			await new Promise((resolve, reject) => {
				const server = net.createServer();
				server.listen(port, (err) => {
					if (err) {
						reject(err);
					} else {
						server.close(() => resolve());
					}
				});
				server.on('error', reject);
			});
			
			// Mark port as allocated
			allocatedPorts.add(port);
			
			if (port !== preferredPort) {
				log(`🔄 Port ${preferredPort} occupied, using ${port}`);
			} else {
				log(`✅ Using preferred port ${port}`);
			}
			return port;
		} catch (error) {
			// Port is occupied, continue
		}
	}
	throw new Error(`No available port found starting from ${preferredPort}`);
}

// Save dynamic port configuration for Vite integration
function saveDynamicPorts(portMap) {
	const config = {
		timestamp: new Date().toISOString(),
		ports: portMap,
		metadata: {
			generator: 'start-microservices.mjs',
			viteLike: true
		}
	};
	
	try {
		writeFileSync('.vscode/dynamic-ports.json', JSON.stringify(config, null, 2));
		log(`💾 Dynamic port configuration saved to .vscode/dynamic-ports.json`);
	} catch (error) {
		warn(`Failed to save port configuration: ${error.message}`);
	}
}

// Read optional env overrides (PowerShell friendly, set before pnpm run dev:full)
const DEFAULTS = {
	NODE_API_PORT: 3005,
	GPU_WORKER_PORT: 8094,
	WASM_WORKER_PORT: 8095,
	GO_LLAMA_PORT: 8096,
	QUIC_GATEWAY_PORT: 8097
};

function envNum(k){ return parseInt(process.env[k] || DEFAULTS[k] || '0', 10); }

const services = [
	{
		name: 'node-api',
		cmd: process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm',
		args: ['--filter', '@yorha/node-api', 'run', 'dev'].filter(Boolean),
		port: envNum('NODE_API_PORT'),
		type: 'node',
		health: '/',
		restart: true,
		env: { 'NODE_API_PORT': String(envNum('NODE_API_PORT')) }
	},
	{
		name: 'gpu-worker',
		cmd: 'npx',
		args: ['tsx','microservices/node-api/src/lib/services/gpu-worker.ts'],
		port: envNum('GPU_WORKER_PORT'),
		type: 'worker',
		restart: true
	},
	{
		name: 'wasm-worker',
		cmd: 'npx',
		args: ['tsx','microservices/node-api/src/lib/services/wasm-worker.ts'],
		port: envNum('WASM_WORKER_PORT'),
		type: 'worker',
		restart: true
	},
	{
		name: 'upload-service',
		cmd: resolve('go-microservice/bin/upload-service.exe'),
		args: [],
		preferredPort: 8093,
		type: 'go',
		restart: true,
		envTemplate: (port) => ({ 'UPLOAD_PORT': String(port), 'MINIO_ENDPOINT': 'localhost:9000' })
	},
	{
		name: 'simple-upload',
		cmd: resolve('go-microservice/bin/simple-upload-fixed.exe'),
		args: [],
		preferredPort: 8094,
		type: 'go',
		restart: true,
		envTemplate: (port) => ({ 'HTTP_PORT': String(port) })
	},
	{
		name: 'grpc-server',
		cmd: resolve('go-microservice/bin/grpc-server.exe'),
		args: [],
		port: 8084,
		type: 'go',
		restart: true,
		env: { 'GRPC_PORT': '8084' }
	},
	{
		name: 'load-balancer',
		cmd: resolve('go-microservice/bin/load-balancer.exe'),
		args: [],
		preferredPort: 8099,
		type: 'go',
		restart: true,
		envTemplate: (port) => ({ 'LB_PORT': String(port) })
	},
	{
		name: 'quic-gateway',
		cmd: resolve('go-microservice/bin/quic-gateway.exe'),
		args: [],
		preferredPort: envNum('QUIC_GATEWAY_PORT'),
		type: 'go',
		restart: true,
		envTemplate: (port) => ({ 
			'QUIC_PORT': String(port),
			'QUIC_HTTP3_PORT': String(port + 1)
		})
	},
	{
		name: 'ws-fanout',
		cmd: 'npx',
		args: ['tsx','microservices/node-api/src/lib/services/ws-fanout-service.ts'],
		port: 8080,
		type: 'worker',
		restart: true
	},
	{
		name: 'recommendations-service',
		cmd: resolve('go-microservice/bin/recommendations-service.exe'),
		args: [],
		preferredPort: 8105,
		type: 'go',
		restart: true,
		envTemplate: (port) => ({ 
			'RECOMMENDATIONS_PORT': String(port),
			'RECOMMENDATIONS_GRPC_PORT': String(port + 10),
			'REDIS_ADDR': 'localhost:6379',
			'DATABASE_URL': 'postgres://postgres:123456@localhost:5432/legal_ai_db'
		})
	},
	{
		name: 'langchain-legal',
		cmd: 'npx',
		args: ['tsx', 'microservices/langchain-legal/src/legal-summarization-service.ts'],
		preferredPort: 8106,
		type: 'worker',
		restart: true,
		envTemplate: (port) => ({ 
			'LANGCHAIN_PORT': String(port),
			'OLLAMA_URL': 'http://localhost:11434/v1'
		})
	}
];

const processes = new Map();

function waitForPort(port, timeoutMs=15000){
	return new Promise((resolve, reject)=>{
		const start = Date.now();
		const attempt = () => {
			const socket = new net.Socket();
			socket.setTimeout(1000);
			socket.once('connect', ()=>{ socket.destroy(); resolve(true); });
			socket.once('timeout', ()=>{ socket.destroy(); retry(); });
			socket.once('error', ()=>{ socket.destroy(); retry(); });
			socket.connect(port, '127.0.0.1');
		};
		const retry = () => {
			if (Date.now() - start > timeoutMs) return reject(new Error('timeout'));
			setTimeout(attempt, 250);
		};
		attempt();
	});
}

function spawnService(svc){
	if (!svc.cmd) { warn(`No command for ${svc.name}`); return null; }
	if (svc.type==='go' && !existsSync(svc.cmd)) { warn(`Binary missing for ${svc.name}: ${svc.cmd}`); return null; }
	// Auto-install node-api deps if missing (one-time)
	if (svc.name==='node-api') {
		const nm = resolve('microservices/node-api/node_modules');
		if (!existsSync(nm)) {
			warn('node-api node_modules missing – attempting pnpm install (filtered)');
			try {
				spawn(process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm', ['--filter','@yorha/node-api','install'], { stdio:'inherit', shell:true });
			} catch(e){ warn('auto install failed', e.message); }
		}
	}

	// Merge service environment variables with process environment
	const env = { ...process.env, ...(svc.env || {}) };

	const child = spawn(svc.cmd, svc.args, {
		stdio: 'inherit',
		shell: true,
		env: env
	});
	processes.set(svc.name, { svc, child, restarts: 0 });
	log(`spawned ${svc.name} (${svc.cmd} ${svc.args.join(' ')}) with env:`, svc.env || 'none');
	child.on('exit', (code, signal)=>{
		warn(`${svc.name} exited code=${code} signal=${signal}`);
		processes.delete(svc.name);
		if (svc.restart){
			const entry = processes.get(svc.name) || { restarts: 0 };
			const next = Math.min(10000, 1000 * Math.pow(2, entry.restarts||0));
			entry.restarts = (entry.restarts||0)+1;
			processes.set(svc.name, entry);
			log(`restart ${svc.name} in ${next}ms (attempt ${entry.restarts})`);
			setTimeout(()=> spawnService(svc), next).unref();
		}
	});
	return child;
}

async function startSequential(){
	// First, discover dynamic ports for services that need them
	const dynamicPortMap = {};
	
	log('🚀 Discovering dynamic ports...');
	for (const svc of services) {
		if (svc.preferredPort && svc.envTemplate) {
			try {
				const allocatedPort = await findAvailablePort(svc.preferredPort);
				svc.port = allocatedPort;
				svc.env = svc.envTemplate(allocatedPort);
				dynamicPortMap[svc.name.replace('-', '_')] = allocatedPort;
				log(`📡 ${svc.name}: port ${allocatedPort} allocated`);
			} catch (error) {
				err(`Failed to allocate port for ${svc.name}: ${error.message}`);
				// Fall back to original port
				svc.port = svc.preferredPort;
				svc.env = svc.envTemplate(svc.preferredPort);
			}
		}
	}
	
	// Save dynamic port configuration for Vite
	saveDynamicPorts(dynamicPortMap);
	
	// Now start services with allocated ports
	log('🚀 Starting services with dynamic ports...');
	for (const svc of services){
		const child = spawnService(svc);
		if (!child){
			warn(`skip waiting for ${svc.name}`);
			continue;
		}
		
		if (svc.port) {
			try {
				await waitForPort(svc.port, 15000);
				log(`✅ ${svc.name} ready on port ${svc.port}`);
			} catch(e){
				err(`❌ ${svc.name} failed to become ready on port ${svc.port} (${e.message})`);
			}
		}
	}
	log('🎉 All spawn attempts completed with dynamic ports');
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
let shuttingDown = false;
function shutdown(){
	if (shuttingDown) return; shuttingDown=true;
	log('Shutting down services...');
	for (const { child } of processes.values()){
		try { child.kill('SIGTERM'); } catch {}
	}
	setTimeout(()=>process.exit(0), 1500).unref();
}

startSequential();
