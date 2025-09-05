/**
 * Ollama Multi-Model Agent for Legal AI
 * Enhanced RAG System Integration with Multiple Model Support
 */

import { EventEmitter } from 'events';
import fetch from 'node-fetch';

export class OllamaAgent extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            endpoint: config.endpoint || process.env.OLLAMA_ENDPOINT || 'http://localhost:11434',
            defaultModel: config.defaultModel || 'llama3:latest',
            timeout: config.timeout || 30000,
            maxRetries: config.maxRetries || 3,
            temperature: config.temperature || 0.1,
            ...config
        };
        
        this.isInitialized = false;
        this.availableModels = new Map();
        this.capabilities = [
            'multi-model-support',
            'local-inference',
            'model-switching',
            'legal-analysis',
            'document-processing',
            'embeddings-generation'
        ];
    }
    
    async initialize() {
        try {
            // Check Ollama service
            const response = await fetch(`${this.config.endpoint}/api/tags`, {
                method: 'GET',
                timeout: this.config.timeout
            });
            
            if (!response.ok) {
                throw new Error(`Ollama service unavailable: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Store available models
            for (const model of data.models) {
                this.availableModels.set(model.name, {
                    name: model.name,
                    size: model.size,
                    digest: model.digest,
                    modified_at: model.modified_at,
                    details: model.details
                });
            }
            
            this.isInitialized = true;
            this.emit('initialized', { 
                agent: 'ollama', 
                status: 'ready', 
                models: Array.from(this.availableModels.keys()) 
            });
            
            console.log(`✅ Ollama initialized with ${this.availableModels.length} models`);
            return true;
        } catch (error) {
            this.emit('error', { agent: 'ollama', error: error.message });
            throw error;
        }
    }
    
    async generateWithModel(model, prompt, options = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        if (!this.availableModels.has(model)) {
            throw new Error(`Model ${this.config.model} not available. Available models: ${this.availableModels.join(', ')}`);
        }
        
        const requestOptions = {
            temperature: options.temperature || this.config.temperature,
            num_predict: options.maxTokens || 2048,
            num_ctx: options.contextWindow || 4096,
            ...options.modelOptions
        };
        
        try {
            const response = await fetch(`${this.config.endpoint}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: model,
                    prompt: prompt,
                    system: options.systemPrompt,
                    options: requestOptions,
                    stream: false
                }),
                timeout: this.config.timeout
            });
            
            if (!response.ok) {
                throw new Error(`Generation failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            const analysis = {
                agent: 'ollama',
                model: model,
                prompt: prompt,
                response: result.response,
                timestamp: new Date().toISOString(),
                performance: {
                    eval_count: result.eval_count,
                    eval_duration: result.eval_duration,
                    total_duration: result.total_duration,
                    tokens_per_second: result.eval_count / (result.eval_duration / 1000000000)
                },
                options: requestOptions
            };
            
            this.emit('generation-complete', analysis);
            return analysis;
        } catch (error) {
            this.emit('error', { agent: 'ollama', model: model, error: error.message });
            throw error;
        }
    }
    
    async compareLegalAnalysis(document, models = null) {
        if (!models) {
            // Use available legal-focused models or fallback to default
            const preferredModels = ['gemma3:latest', 'llama3:latest', 'mistral:latest'];
            models = preferredModels.filter(model => this.availableModels.has(model));
            
            if (models.length === 0) {
                models = [this.config.defaultModel];
            }
        }
        
        const analysisPrompt = `Analyze this legal document and provide:
1. Document classification
2. Key legal issues
3. Risk assessment
4. Recommendations

Document: ${document}`;
        
        const systemPrompt = "You are a legal expert. Provide structured, professional analysis.";
        
        const results = {};
        
        for (const model of models) {
            try {
                console.log(`Analyzing with model: ${this.config.model}`);
                const result = await this.generateWithModel(model, analysisPrompt, { 
                    systemPrompt: systemPrompt,
                    maxTokens: 1024
                });
                results[model] = result;
            } catch (error) {
                console.warn(`Failed to analyze with ${this.config.model}: ${error.message}`);
                results[model] = { error: error.message };
            }
        }
        
        return {
            agent: 'ollama',
            type: 'comparative-analysis',
            document: document.substring(0, 200) + '...',
            results: results,
            timestamp: new Date().toISOString()
        };
    }
    
    async generateEmbeddings(text, model = 'nomic-embed-text') {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        try {
            const response = await fetch(`${this.config.endpoint}/api/embeddings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: model,
                    prompt: text
                }),
                timeout: this.config.timeout
            });
            
            if (!response.ok) {
                throw new Error(`Embeddings generation failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            return {
                agent: 'ollama',
                type: 'embeddings',
                model: model,
                text: text.substring(0, 100) + '...',
                embeddings: result.embedding,
                dimensions: result.embedding.length,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.emit('error', { agent: 'ollama', task: 'embeddings', error: error.message });
            throw error;
        }
    }
    
    async pullModel(modelName) {
        try {
            console.log(`Pulling model: ${modelName}`);
            
            const response = await fetch(`${this.config.endpoint}/api/pull`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: modelName }),
                timeout: 300000 // 5 minutes for model download
            });
            
            if (!response.ok) {
                throw new Error(`Failed to pull model: ${response.status}`);
            }
            
            // Refresh available models
            await this.initialize();
            
            console.log(`✅ Model ${modelName} pulled successfully`);
            return true;
        } catch (error) {
            console.error(`❌ Failed to pull model ${modelName}: ${error.message}`);
            throw error;
        }
    }
    
    getAvailableModels() {
        return Array.from(this.availableModels.keys());
    }
    
    getModelInfo(modelName) {
        return this.availableModels.get(modelName);
    }
    
    getCapabilities() {
        return this.capabilities;
    }
    
    getStatus() {
        return {
            agent: 'ollama',
            initialized: this.isInitialized,
            endpoint: this.config.endpoint,
            availableModels: this.getAvailableModels(),
            defaultModel: this.config.defaultModel,
            capabilities: this.capabilities
        };
    }
}

export default OllamaAgent;
