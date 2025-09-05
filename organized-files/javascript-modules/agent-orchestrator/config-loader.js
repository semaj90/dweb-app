/**
 * Configuration Loader for Agent Orchestrator
 * Loads and validates agent configuration with LLM optimization patterns
 */

import { readFileSync } from 'fs';
import { Worker } from 'worker_threads';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class ConfigLoader {
    constructor(configPath = null) {
        this.configPath = configPath || join(__dirname, 'agents-config.json');
        this.config = null;
        this.workers = new Map();
        this.isLoaded = false;
    }
    
    /**
     * Load and validate configuration
     */
    async loadConfig() {
        try {
            const configData = readFileSync(this.configPath, 'utf8');
            this.config = JSON.parse(configData);
            
            // Validate configuration
            this.validateConfig();
            
            // Initialize optimization features
            await this.initializeOptimizations();
            
            this.isLoaded = true;
            return this.config;
        } catch (error) {
            throw new Error(`Failed to load configuration: ${error.message}`);
        }
    }
    
    /**
     * Basic configuration validation
     */
    validateConfig() {
        if (!this.config.agents) {
            throw new Error('Configuration must include agents section');
        }
        
        if (!this.config.version) {
            throw new Error('Configuration must include version');
        }
        
        // Validate agent configurations
        for (const [agentName, agentConfig] of Object.entries(this.config.agents)) {
            if (!agentConfig.type) {
                throw new Error(`Agent ${agentName} must have a type`);
            }
            
            if (typeof agentConfig.enabled !== 'boolean') {
                throw new Error(`Agent ${agentName} must have enabled boolean`);
            }
        }
        
        console.log(`✅ Configuration validated: ${Object.keys(this.config.agents).length} agents`);
    }
    
    /**
     * Initialize optimization features based on configuration
     */
    async initializeOptimizations() {
        const globalOpts = this.config.optimization?.global;
        
        if (!globalOpts) {
            console.log('ℹ️ No global optimizations configured');
            return;
        }
        
        // Initialize worker threads for token processing
        if (globalOpts.workerThreads?.enabled) {
            await this.initializeWorkerThreads(globalOpts.workerThreads);
        }
        
        // Initialize caching layers
        if (globalOpts.caching?.strategy === 'multi-layer') {
            await this.initializeCaching(globalOpts.caching);
        }
        
        console.log('🚀 Optimizations initialized');
    }
    
    /**
     * Initialize worker thread pool for LLM stream processing
     */
    async initializeWorkerThreads(workerConfig) {
        const poolSize = workerConfig.poolSize || 4;
        const handlerScript = join(__dirname, workerConfig.handlerScript || 'llm-stream-handler.js');
        
        console.log(`🔧 Initializing ${poolSize} worker threads for LLM processing`);
        
        for (let i = 0; i < poolSize; i++) {
            try {
                const worker = new Worker(handlerScript, {
                    workerData: { 
                        workerId: i,
                        enableSIMD: true,
                        batchSize: 1024
                    }
                });
                
                // Set up worker message handling
                worker.on('message', (message) => {
                    if (message.type === 'ready') {
                        console.log(`✅ Worker ${message.workerId} ready (SIMD: ${message.simdSupported})`);
                    }
                });
                
                worker.on('error', (error) => {
                    console.error(`❌ Worker ${i} error:`, error);
                });
                
                this.workers.set(i, worker);
            } catch (error) {
                console.warn(`⚠️ Failed to create worker ${i}:`, error.message);
            }
        }
        
        console.log(`🎯 Worker thread pool initialized: ${this.workers.size}/${poolSize} workers`);
    }
    
    /**
     * Initialize multi-layer caching
     */
    async initializeCaching(cachingConfig) {
        console.log('🗄️ Initializing multi-layer caching');
        
        for (const layer of cachingConfig.layers || []) {
            try {
                switch (layer.type) {
                    case 'memory':
                        console.log(`  📝 Memory cache layer (Priority: ${layer.priority})`);
                        break;
                    case 'redis':
                        console.log(`  🔴 Redis cache layer (Priority: ${layer.priority})`);
                        // TODO: Initialize Redis connection
                        break;
                    case 'loki':
                        console.log(`  🟢 Loki cache layer (Priority: ${layer.priority})`);
                        // TODO: Initialize Loki.js
                        break;
                    default:
                        console.warn(`⚠️ Unknown cache layer type: ${layer.type}`);
                }
            } catch (error) {
                console.warn(`⚠️ Failed to initialize ${layer.type} cache:`, error.message);
            }
        }
    }
    
    /**
     * Get agent configuration
     */
    getAgentConfig(agentName) {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return this.config.agents[agentName];
    }
    
    /**
     * Get enabled agents
     */
    getEnabledAgents() {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return Object.entries(this.config.agents)
            .filter(([name, config]) => config.enabled)
            .map(([name, config]) => ({ name, ...config }));
    }
    
    /**
     * Get workflow configuration
     */
    getWorkflow(workflowName) {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return this.config.workflows?.[workflowName];
    }
    
    /**
     * Get available workflows
     */
    getWorkflows() {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return Object.keys(this.config.workflows || {});
    }
    
    /**
     * Get optimization settings
     */
    getOptimizations() {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return this.config.optimization;
    }
    
    /**
     * Get next available worker for token processing
     */
    getAvailableWorker() {
        if (this.workers.size === 0) {
            return null;
        }
        
        // Simple round-robin selection
        const workerIds = Array.from(this.workers.keys());
        const selectedId = workerIds[Math.floor(Math.random() * workerIds.length)];
        return this.workers.get(selectedId);
    }
    
    /**
     * Process tokens using worker thread
     */
    async processTokensWithWorker(tokens, action = 'processTokens') {
        const worker = this.getAvailableWorker();
        if (!worker) {
            throw new Error('No worker threads available');
        }
        
        return new Promise((resolve, reject) => {
            const messageId = `msg_${Date.now()}_${Math.random()}`;
            
            const messageHandler = (message) => {
                if (message.id === messageId) {
                    worker.off('message', messageHandler);
                    if (message.error) {
                        reject(new Error(message.error));
                    } else {
                        resolve(message.result);
                    }
                }
            };
            
            worker.on('message', messageHandler);
            worker.postMessage({
                id: messageId,
                action,
                data: tokens
            });
            
            // Timeout after 30 seconds
            setTimeout(() => {
                worker.off('message', messageHandler);
                reject(new Error('Worker timeout'));
            }, 30000);
        });
    }
    
    /**
     * Get bottleneck analysis from configuration
     */
    getBottleneckAnalysis() {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return this.config.optimization?.bottleneckAnalysis;
    }
    
    /**
     * Get monitoring configuration
     */
    getMonitoringConfig() {
        if (!this.isLoaded) {
            throw new Error('Configuration not loaded. Call loadConfig() first.');
        }
        
        return this.config.monitoring;
    }
    
    /**
     * Cleanup resources
     */
    async cleanup() {
        console.log('🧹 Cleaning up configuration resources');
        
        // Terminate worker threads
        for (const [id, worker] of this.workers) {
            try {
                await worker.terminate();
                console.log(`✅ Worker ${id} terminated`);
            } catch (error) {
                console.warn(`⚠️ Error terminating worker ${id}:`, error.message);
            }
        }
        
        this.workers.clear();
        this.isLoaded = false;
        
        console.log('✅ Cleanup complete');
    }
    
    /**
     * Reload configuration
     */
    async reloadConfig() {
        await this.cleanup();
        return this.loadConfig();
    }
    
    /**
     * Get configuration summary for debugging
     */
    getSummary() {
        if (!this.isLoaded) {
            return { status: 'not loaded' };
        }
        
        const enabledAgents = this.getEnabledAgents();
        const workflows = this.getWorkflows();
        
        return {
            status: 'loaded',
            version: this.config.version,
            agents: {
                total: Object.keys(this.config.agents).length,
                enabled: enabledAgents.length,
                types: enabledAgents.reduce((acc, agent) => {
                    acc[agent.type] = (acc[agent.type] || 0) + 1;
                    return acc;
                }, {})
            },
            workflows: workflows.length,
            workers: this.workers.size,
            optimizations: {
                workerThreads: !!this.config.optimization?.global?.workerThreads?.enabled,
                multiLayerCaching: this.config.optimization?.global?.caching?.strategy === 'multi-layer',
                tokenCompression: !!this.config.optimization?.global?.minimizeJsonPayload?.enabled
            }
        };
    }
}

export default ConfigLoader;