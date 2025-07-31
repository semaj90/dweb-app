/**
 * Gemma3 Local LLM Agent for Legal AI
 * Enhanced RAG System Integration with GGUF Model Support
 */

import { EventEmitter } from 'events';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

export class GemmaAgent extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            endpoint: config.endpoint || process.env.OLLAMA_ENDPOINT || 'http://localhost:11434',
            model: config.model || 'gemma3:latest',
            modelPath: config.modelPath || process.env.GEMMA3_MODEL_PATH,
            temperature: config.temperature || 0.1,
            maxTokens: config.maxTokens || 2048,
            contextWindow: config.contextWindow || 8192,
            ...config
        };
        
        this.isInitialized = false;
        this.modelLoaded = false;
        this.capabilities = [
            'local-inference',
            'legal-document-analysis',
            'contract-review',
            'legal-qa',
            'document-summarization',
            'privacy-preserving-analysis'
        ];
    }
    
    async initialize() {
        try {
            // Check if Ollama is running
            const healthResponse = await fetch(`${this.config.endpoint}/api/tags`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (!healthResponse.ok) {
                throw new Error(`Ollama service unavailable: ${healthResponse.status}`);
            }
            
            // Check if Gemma3 model is available
            const models = await healthResponse.json();
            const gemmaModel = models.models.find(m => m.name.includes('gemma3') || m.name === this.config.model);
            
            if (!gemmaModel) {
                await this.loadModel();
            } else {
                this.modelLoaded = true;
                console.log(`Model ${this.config.model} is already loaded`);
            }
            
            this.isInitialized = true;
            this.emit('initialized', { agent: 'gemma', status: 'ready', model: this.config.model });
            return true;
        } catch (error) {
            this.emit('error', { agent: 'gemma', error: error.message });
            throw error;
        }
    }
    
    async loadModel() {
        if (this.config.modelPath && fs.existsSync(this.config.modelPath)) {
            console.log(`Loading Gemma3 model from: ${this.config.modelPath}`);
            
            // Create Ollama Modelfile
            const modelfile = `
FROM ${this.config.modelPath}

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER num_ctx ${this.config.contextWindow}
PARAMETER temperature ${this.config.temperature}

SYSTEM """You are a specialized legal AI assistant. You help with legal document analysis, contract review, and legal research. Always provide accurate, well-reasoned responses and cite relevant legal principles when applicable."""
`;
            
            // Save Modelfile
            const modelfilePath = path.join(path.dirname(this.config.modelPath), 'Modelfile-gemma3-legal');
            fs.writeFileSync(modelfilePath, modelfile);
            
            // Create model in Ollama
            const createResponse = await fetch(`${this.config.endpoint}/api/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: this.config.model,
                    modelfile: modelfile
                })
            });
            
            if (!createResponse.ok) {
                throw new Error(`Failed to create model: ${createResponse.status}`);
            }
            
            this.modelLoaded = true;
            console.log("✅ Gemma3 model loaded successfully");
        } else {
            throw new Error(`Model file not found: ${this.config.modelPath}`);
        }
    }
    
    async generateResponse(prompt, context = {}) {
        if (!this.isInitialized || !this.modelLoaded) {
            await this.initialize();
        }
        
        const systemPrompt = context.systemPrompt || 
            "You are a legal AI assistant. Provide accurate, well-reasoned legal analysis.";
        
        const fullPrompt = context.includeContext ? 
            `Context: ${context.contextData}\n\nUser Query: ${prompt}` : 
            prompt;
        
        try {
            const response = await fetch(`${this.config.endpoint}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: this.config.model,
                    prompt: fullPrompt,
                    system: systemPrompt,
                    options: {
                        temperature: this.config.temperature,
                        num_predict: this.config.maxTokens,
                        num_ctx: this.config.contextWindow
                    },
                    stream: false
                })
            });
            
            if (!response.ok) {
                throw new Error(`Generation failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            const analysis = {
                agent: 'gemma',
                model: this.config.model,
                prompt: prompt,
                response: result.response,
                timestamp: new Date().toISOString(),
                context: context,
                performance: {
                    eval_count: result.eval_count,
                    eval_duration: result.eval_duration,
                    total_duration: result.total_duration
                }
            };
            
            this.emit('generation-complete', analysis);
            return analysis;
        } catch (error) {
            this.emit('error', { agent: 'gemma', error: error.message, prompt: prompt.substring(0, 100) });
            throw error;
        }
    }
    
    async analyzeLegalDocument(document, analysisType = 'comprehensive') {
        const analysisPrompts = {
            comprehensive: `Analyze this legal document comprehensively. Identify:
1. Document type and purpose
2. Key parties involved
3. Main legal obligations
4. Important dates and deadlines
5. Potential legal issues or risks
6. Compliance requirements

Document: ${document}`,
            
            contract: `Review this contract and identify:
1. Contract type and governing law
2. Parties and their obligations
3. Payment terms and conditions
4. Termination clauses
5. Dispute resolution mechanisms
6. Risk factors and recommendations

Contract: ${document}`,
            
            compliance: `Analyze this document for compliance issues:
1. Regulatory requirements
2. Legal standards adherence
3. Potential violations
4. Recommended corrections
5. Compliance score and rationale

Document: ${document}`
        };
        
        const prompt = analysisPrompts[analysisType] || analysisPrompts.comprehensive;
        
        return await this.generateResponse(prompt, {
            analysisType: analysisType,
            includeContext: true,
            systemPrompt: "You are an expert legal analyst. Provide detailed, structured analysis with clear recommendations."
        });
    }
    
    async summarizeDocument(document, maxLength = 500) {
        const prompt = `Summarize this legal document in no more than ${maxLength} words. 
Focus on the most important legal points, obligations, and implications:

${document}`;
        
        return await this.generateResponse(prompt, {
            task: 'summarization',
            maxLength: maxLength,
            systemPrompt: "You are a legal document summarization expert. Create concise, accurate summaries."
        });
    }
    
    getCapabilities() {
        return this.capabilities;
    }
    
    getStatus() {
        return {
            agent: 'gemma',
            initialized: this.isInitialized,
            modelLoaded: this.modelLoaded,
            model: this.config.model,
            endpoint: this.config.endpoint,
            capabilities: this.capabilities
        };
    }
}

export default GemmaAgent;
