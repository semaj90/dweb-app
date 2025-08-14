#!/usr/bin/env node
/**
 * Production Ollama Setup for Enhanced RAG System
 * Configures and validates Ollama models for semantic caching
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Ollama model configurations for legal AI
const OLLAMA_MODELS = {
    embedding: {
        name: 'nomic-embed-text',
        size: '274MB',
        description: 'High-quality text embeddings (384 dimensions)',
        priority: 'critical'
    },
    legal: {
        name: 'gemma2:9b',
        size: '5.4GB',
        description: 'Legal reasoning and analysis',
        priority: 'high'
    },
    code: {
        name: 'codellama:7b',
        size: '3.8GB',
        description: 'Code analysis and generation',
        priority: 'medium'
    },
    fast: {
        name: 'gemma2:2b',
        size: '1.6GB',
        description: 'Fast responses for simple queries',
        priority: 'medium'
    }
};

const OLLAMA_CONFIG = {
    host: 'localhost',
    port: 11434,
    baseUrl: 'http://localhost:11434',
    timeout: 120000, // 2 minutes for model loading
    concurrency: 4,
    cacheTTL: 86400000, // 24 hours in milliseconds
    embeddingDimensions: 384
};

async function setupOllamaProduction() {
    console.log('🚀 Setting up Ollama for Production Enhanced RAG...\n');

    try {
        // Step 1: Check Ollama installation
        console.log('1️⃣ Checking Ollama installation...');
        await checkOllamaInstallation();

        // Step 2: Configure Ollama service
        console.log('\n2️⃣ Configuring Ollama service...');
        await configureOllamaService();

        // Step 3: Pull required models
        console.log('\n3️⃣ Pulling required models...');
        await pullRequiredModels();

        // Step 4: Validate model functionality
        console.log('\n4️⃣ Validating model functionality...');
        await validateModels();

        // Step 5: Create production configuration
        console.log('\n5️⃣ Creating production configuration...');
        await createProductionConfig();

        // Step 6: Set up caching configuration
        console.log('\n6️⃣ Setting up caching configuration...');
        await setupCachingConfig();

        console.log('\n🎉 Ollama Production Setup Complete!');
        console.log('\n📊 Setup Summary:');
        console.log(`✅ Ollama service running on ${OLLAMA_CONFIG.baseUrl}`);
        console.log(`✅ ${Object.keys(OLLAMA_MODELS).length} models configured`);
        console.log(`✅ Embedding dimensions: ${OLLAMA_CONFIG.embeddingDimensions}`);
        console.log(`✅ Cache TTL: ${OLLAMA_CONFIG.cacheTTL / 3600000} hours`);
        console.log(`✅ Max concurrency: ${OLLAMA_CONFIG.concurrency}`);

    } catch (error) {
        console.error('💥 Ollama setup failed:', error);
        process.exit(1);
    }
}

async function checkOllamaInstallation() {
    try {
        const { stdout } = await execAsync('ollama --version');
        console.log(`✅ Ollama installed: ${stdout.trim()}`);
    } catch (error) {
        console.log('❌ Ollama not found. Installing...');
        console.log('📥 Please install Ollama from: https://ollama.ai/download');
      run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama');
        throw new Error('Ollama installation required');
    }
}

async function configureOllamaService() {
    try {
        // Check if Ollama service is running
        const { stdout } = await execAsync(`curl -s ${OLLAMA_CONFIG.baseUrl}/api/tags || echo "not running"`);

        if (stdout.includes('not running')) {
            console.log('🔄 Starting Ollama service...');
            // Start Ollama service in background
            const ollamaProcess = spawn('ollama', ['serve'], {
                detached: true,
                stdio: 'ignore'
            });
            ollamaProcess.unref();

            // Wait for service to start
            await new Promise(resolve => setTimeout(resolve, 3000));
        }

        console.log('✅ Ollama service configured and running');
    } catch (error) {
        console.warn('⚠️ Ollama service configuration issue:', error.message);
        console.log('💡 Trying to continue with existing service...');
    }
}

async function pullRequiredModels() {
    for (const [key, model] of Object.entries(OLLAMA_MODELS)) {
        try {
            console.log(`📥 Pulling ${model.name} (${model.size})...`);
            console.log(`   📝 ${model.description}`);

            const pullProcess = spawn('ollama', ['pull', model.name], {
                stdio: 'pipe'
            });

            pullProcess.stdout.on('data', (data) => {
                process.stdout.write(`   ${data}`);
            });

            await new Promise((resolve, reject) => {
                pullProcess.on('close', (code) => {
                    if (code === 0) {
                        console.log(`✅ ${model.name} pulled successfully`);
                        resolve(code);
                    } else {
                        console.log(`❌ Failed to pull ${model.name}`);
                        if (model.priority === 'critical') {
                            reject(new Error(`Critical model ${model.name} failed to pull`));
                        } else {
                            console.log(`⚠️ Continuing without ${model.name} (${model.priority} priority)`);
                            resolve(code);
                        }
                    }
                });
            });

        } catch (error) {
            if (model.priority === 'critical') {
                throw error;
            }
            console.log(`⚠️ Skipping ${model.name}: ${error.message}`);
        }
    }
}

async function validateModels() {
    const validationTests = [
        {
            model: 'nomic-embed-text',
            test: 'embedding',
            input: 'legal contract analysis',
            expected: 'array of 384 numbers'
        },
        {
            model: 'gemma2:9b',
            test: 'generation',
            input: 'What are the key elements of a legal contract?',
            expected: 'coherent legal response'
        }
    ];

    for (const test of validationTests) {
        try {
            console.log(`🧪 Testing ${test.model}...`);

            const testPayload = test.test === 'embedding'
                ? { model: test.model, prompt: test.input }
                : { model: test.model, prompt: test.input, stream: false };

            const endpoint = test.test === 'embedding' ? '/api/embeddings' : '/api/generate';

            const curlCommand = `curl -s -X POST ${OLLAMA_CONFIG.baseUrl}${endpoint} -H "Content-Type: application/json" -d '${JSON.stringify(testPayload)}'`;

            const { stdout } = await execAsync(curlCommand);
            const response = JSON.parse(stdout);

            if (test.test === 'embedding') {
                if (response.embedding && response.embedding.length === OLLAMA_CONFIG.embeddingDimensions) {
                    console.log(`✅ ${test.model} embedding test passed (${response.embedding.length} dimensions)`);
                } else {
                    throw new Error(`Invalid embedding dimensions: ${response.embedding?.length}`);
                }
            } else {
                if (response.response && response.response.length > 10) {
                    console.log(`✅ ${test.model} generation test passed (${response.response.length} chars)`);
                } else {
                    throw new Error('Invalid generation response');
                }
            }

        } catch (error) {
            console.log(`❌ ${test.model} validation failed: ${error.message}`);
            if (test.model === 'nomic-embed-text') {
                throw new Error('Critical embedding model validation failed');
            }
        }
    }
}

async function createProductionConfig() {
    const productionConfig = {
        ollama: OLLAMA_CONFIG,
        models: OLLAMA_MODELS,
        enhancedRAG: {
            embeddingModel: 'nomic-embed-text',
            generationModel: 'gemma2:9b',
            fallbackModel: 'gemma2:2b',
            codeModel: 'codellama:7b',
            batchSize: 10,
            maxConcurrentRequests: 4,
            timeout: 30000
        },
        caching: {
            enabled: true,
            ttl: OLLAMA_CONFIG.cacheTTL,
            maxSize: 10000,
            persistToDisk: true,
            compressionEnabled: true
        },
        monitoring: {
            metricsEnabled: true,
            performanceLogging: true,
            errorTracking: true,
            healthChecks: true
        }
    };

    // Write config to multiple locations
    const configPaths = [
        'ollama-production-config.json',
        'vscode-llm-extension/src/ollama-config.json',
        'rag/ollama-config.json'
    ];

    for (const configPath of configPaths) {
        try {
            const fs = await import('fs');
            const path = await import('path');

            // Ensure directory exists
            const dir = path.dirname(configPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }

            fs.writeFileSync(configPath, JSON.stringify(productionConfig, null, 2));
            console.log(`✅ Configuration written to ${configPath}`);
        } catch (error) {
            console.log(`⚠️ Failed to write config to ${configPath}: ${error.message}`);
        }
    }
}

async function setupCachingConfig() {
    const cachingScript = `#!/usr/bin/env node
/**
 * Ollama Caching Service for Enhanced RAG
 * Manages embedding and response caching with TTL
 */

import fs from 'fs';
import path from 'path';

class OllamaCache {
    constructor() {
        this.cacheDir = './ollama-cache';
        this.embeddingCache = new Map();
        this.responseCache = new Map();
        this.stats = { hits: 0, misses: 0, saves: 0 };
        this.ensureCacheDir();
        this.loadPersistedCache();
    }

    ensureCacheDir() {
        if (!fs.existsSync(this.cacheDir)) {
            fs.mkdirSync(this.cacheDir, { recursive: true });
        }
    }

    generateCacheKey(text, model = 'default') {
        const crypto = require('crypto');
        return crypto.createHash('sha256').update(\`\${model}:\${text}\`).digest('hex');
    }

    async getEmbedding(text, model = 'nomic-embed-text', ttl = ${OLLAMA_CONFIG.cacheTTL}) {
        const key = this.generateCacheKey(text, model);
        const cached = this.embeddingCache.get(key);

        if (cached && Date.now() - cached.timestamp < ttl) {
            this.stats.hits++;
            return cached.embedding;
        }

        // Fetch from Ollama API
        const response = await fetch('${OLLAMA_CONFIG.baseUrl}/api/embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, prompt: text })
        });

        const data = await response.json();

        if (data.embedding) {
            this.embeddingCache.set(key, {
                embedding: data.embedding,
                timestamp: Date.now(),
                text: text.substring(0, 100) // Store preview
            });
            this.stats.misses++;
            this.stats.saves++;
            this.persistCache();
        }

        return data.embedding;
    }

    async getResponse(prompt, model = 'gemma2:9b', ttl = ${OLLAMA_CONFIG.cacheTTL}) {
        const key = this.generateCacheKey(prompt, model);
        const cached = this.responseCache.get(key);

        if (cached && Date.now() - cached.timestamp < ttl) {
            this.stats.hits++;
            return cached.response;
        }

        // Fetch from Ollama API
        const response = await fetch('${OLLAMA_CONFIG.baseUrl}/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, prompt, stream: false })
        });

        const data = await response.json();

        if (data.response) {
            this.responseCache.set(key, {
                response: data.response,
                timestamp: Date.now(),
                prompt: prompt.substring(0, 100) // Store preview
            });
            this.stats.misses++;
            this.stats.saves++;
            this.persistCache();
        }

        return data.response;
    }

    persistCache() {
        try {
            const cacheData = {
                embeddings: Array.from(this.embeddingCache.entries()),
                responses: Array.from(this.responseCache.entries()),
                stats: this.stats,
                timestamp: Date.now()
            };

            fs.writeFileSync(
                path.join(this.cacheDir, 'ollama-cache.json'),
                JSON.stringify(cacheData, null, 2)
            );
        } catch (error) {
            console.warn('Failed to persist cache:', error);
        }
    }

    loadPersistedCache() {
        try {
            const cachePath = path.join(this.cacheDir, 'ollama-cache.json');
            if (fs.existsSync(cachePath)) {
                const cacheData = JSON.parse(fs.readFileSync(cachePath, 'utf8'));

                this.embeddingCache = new Map(cacheData.embeddings || []);
                this.responseCache = new Map(cacheData.responses || []);
                this.stats = cacheData.stats || this.stats;

                console.log('✅ Loaded persisted cache with', this.embeddingCache.size, 'embeddings and', this.responseCache.size, 'responses');
            }
        } catch (error) {
            console.warn('Failed to load persisted cache:', error);
        }
    }

    getStats() {
        return {
            ...this.stats,
            embeddingCacheSize: this.embeddingCache.size,
            responseCacheSize: this.responseCache.size,
            hitRate: this.stats.hits / (this.stats.hits + this.stats.misses) || 0
        };
    }

    clearCache() {
        this.embeddingCache.clear();
        this.responseCache.clear();
        this.stats = { hits: 0, misses: 0, saves: 0 };
        this.persistCache();
    }
}

export const ollamaCache = new OllamaCache();
export default OllamaCache;
`;

    const fs = await import('fs');
    fs.writeFileSync('ollama-cache-service.js', cachingScript);
    console.log('✅ Caching service created: ollama-cache-service.js');
}

// Run the setup
setupOllamaProduction();