// High-performance local Gemma3 inference service with WebAssembly and Ollama integration
// Provides LLVM-quality performance for legal AI document analysis

import type { 
    Gemma3WasmModule, 
    Gemma3InferenceEngine, 
    Gemma3GenerationOptions,
    Gemma3GenerationResult,
    Gemma3ServiceConfig 
} from '../wasm/gemma3-inference.d.ts';

export interface LegalAnalysisResult {
    summary: string;
    keyTerms: string[];
    entities: Array<{
        type: string;
        value: string;
        confidence: number;
    }>;
    risks: Array<{
        type: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
    }>;
    recommendations: string[];
    confidence: number;
    processingTime: number;
    method: 'webassembly' | 'ollama' | 'hybrid';
}

export interface EmbeddingResult {
    embedding: number[];
    dimensions: number;
    model: string;
    processingTime: number;
}

export class Gemma3LocalService {
    private wasmModule: Gemma3WasmModule | null = null;
    private engine: Gemma3InferenceEngine | null = null;
    private config: Required<Gemma3ServiceConfig>;
    private initialized = false;
    private modelLoaded = false;
    
    // Fallback to Ollama service
    private ollamaBaseUrl = 'http://localhost:11434';
    private ollamaModel = 'gemma3-legal:latest';
    private nomicModel = 'nomic-embed-text:latest';
    
    // Performance tracking
    private stats = {
        totalRequests: 0,
        wasmRequests: 0,
        ollamaRequests: 0,
        avgResponseTime: 0,
        cacheHits: 0
    };

    constructor(config: Gemma3ServiceConfig = {}) {
        this.config = {
            modelUrl: config.modelUrl || '/models/gemma3-legal-weights.bin',
            wasmUrl: config.wasmUrl || '/static/wasm/gemma3-inference.js',
            enableWebGPU: config.enableWebGPU ?? true,
            enableThreading: config.enableThreading ?? true,
            maxCacheSize: config.maxCacheSize || 100,
            defaultTemperature: config.defaultTemperature || 0.1
        };
    }

    /**
     * Initialize the service with WebAssembly and fallback capabilities
     */
    async initialize(): Promise<boolean> {
        console.log('[Gemma3Service] Initializing local inference service...');
        
        try {
            // First, verify Ollama is available as fallback
            const ollamaHealthy = await this.checkOllamaHealth();
            if (!ollamaHealthy) {
                console.warn('[Gemma3Service] Ollama not available - WebAssembly only mode');
            }
            
            // Load WebAssembly module
            const wasmSuccess = await this.initializeWebAssembly();
            
            if (wasmSuccess) {
                console.log('[Gemma3Service] WebAssembly engine ready');
                this.initialized = true;
                return true;
            } else if (ollamaHealthy) {
                console.log('[Gemma3Service] Fallback to Ollama service');
                this.initialized = true;
                return true;
            } else {
                throw new Error('Neither WebAssembly nor Ollama available');
            }
            
        } catch (error: any) {
            console.error('[Gemma3Service] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Initialize WebAssembly inference engine
     */
    private async initializeWebAssembly(): Promise<boolean> {
        try {
            // Dynamic import to handle missing WebAssembly gracefully
            const Gemma3WasmModule = await import(this.config.wasmUrl);
            this.wasmModule = await (Gemma3WasmModule.default || Gemma3WasmModule)();
            
            if (!this.wasmModule) {
                throw new Error('Failed to load WebAssembly module');
            }
            
            this.engine = new this.wasmModule.Gemma3InferenceEngine();
            
            // Load model weights if available
            try {
                const weightsResponse = await fetch(this.config.modelUrl);
                if (weightsResponse.ok) {
                    const weights = await weightsResponse.arrayBuffer();
                    const loadResult = await this.engine.loadModelWeights(weights);
                    
                    if (loadResult.success) {
                        this.modelLoaded = true;
                        console.log(`[Gemma3Service] Model loaded: ${loadResult.parameters} parameters`);
                    } else {
                        console.warn('[Gemma3Service] Model weights not loaded:', loadResult.error);
                    }
                }
            } catch (error: any) {
                console.warn('[Gemma3Service] Model weights not available, using inference-only mode');
            }
            
            return true;
            
        } catch (error: any) {
            console.warn('[Gemma3Service] WebAssembly initialization failed:', error);
            return false;
        }
    }

    /**
     * Check if Ollama service is healthy
     */
    private async checkOllamaHealth(): Promise<boolean> {
        try {
            const response = await fetch(`${this.ollamaBaseUrl}/api/tags`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                const data = await response.json();
                const hasGemma3 = data.models?.some((model: any) => 
                    model.name.includes('gemma3-legal'));
                const hasNomic = data.models?.some((model: any) => 
                    model.name.includes('nomic-embed'));
                
                if (hasGemma3 && hasNomic) {
                    console.log('[Gemma3Service] Ollama healthy with required models');
                    return true;
                } else {
                    console.warn('[Gemma3Service] Ollama missing required models');
                    return false;
                }
            }
            return false;
        } catch (error: any) {
            console.warn('[Gemma3Service] Ollama health check failed:', error);
            return false;
        }
    }

    /**
     * Generate text using best available method (WebAssembly preferred, Ollama fallback)
     */
    async generate(prompt: string, options: Gemma3GenerationOptions = {}): Promise<Gemma3GenerationResult> {
        if (!this.initialized) {
            throw new Error('Service not initialized. Call initialize() first.');
        }

        const startTime = performance.now();
        this.stats.totalRequests++;

        try {
            // Prefer WebAssembly if model is loaded
            if (this.engine && this.modelLoaded) {
                const result = await this.generateWithWebAssembly(prompt, options);
                this.stats.wasmRequests++;
                return result;
            }
            
            // Fallback to Ollama
            const result = await this.generateWithOllama(prompt, options);
            this.stats.ollamaRequests++;
            return result;
            
        } catch (error: any) {
            console.error('[Gemma3Service] Generation failed:', error);
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                processing_time_ms: performance.now() - startTime
            };
        } finally {
            const responseTime = performance.now() - startTime;
            this.stats.avgResponseTime = (
                (this.stats.avgResponseTime * (this.stats.totalRequests - 1) + responseTime) / 
                this.stats.totalRequests
            );
        }
    }

    /**
     * Generate with WebAssembly engine
     */
    private async generateWithWebAssembly(
        prompt: string, 
        options: Gemma3GenerationOptions
    ): Promise<Gemma3GenerationResult> {
        if (!this.engine) {
            throw new Error('WebAssembly engine not available');
        }

        const result = await this.engine.generateText(prompt, {
            max_tokens: options.max_tokens || 1024,
            temperature: options.temperature || this.config.defaultTemperature,
            top_p: options.top_p || 0.9,
            use_cache: options.use_cache !== false
        });

        return {
            ...result,
            method: 'WebAssembly Gemma3 + LLVM optimizations'
        };
    }

    /**
     * Generate with Ollama fallback
     */
    private async generateWithOllama(
        prompt: string, 
        options: Gemma3GenerationOptions
    ): Promise<Gemma3GenerationResult> {
        const startTime = performance.now();
        
        const response = await fetch(`${this.ollamaBaseUrl}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: this.ollamaModel,
                prompt,
                options: {
                    num_predict: options.max_tokens || 1024,
                    temperature: options.temperature || this.config.defaultTemperature,
                    top_p: options.top_p || 0.9,
                },
                stream: false
            })
        });

        if (!response.ok) {
            throw new Error(`Ollama request failed: ${response.statusText}`);
        }

        const data = await response.json();
        const processingTime = performance.now() - startTime;

        return {
            success: true,
            text: data.response || '',
            tokens_generated: this.estimateTokens(data.response || ''),
            processing_time_ms: processingTime,
            tokens_per_second: this.estimateTokens(data.response || '') / (processingTime / 1000),
            method: 'Ollama Gemma3-Legal'
        };
    }

    /**
     * Analyze legal document with specialized prompting
     */
    async analyzeDocument(
        title: string,
        content: string,
        analysisType: 'comprehensive' | 'quick' | 'risk-focused' = 'comprehensive'
    ): Promise<LegalAnalysisResult> {
        if (!this.initialized) {
            throw new Error('Service not initialized');
        }

        const prompt = this.buildLegalAnalysisPrompt(title, content, analysisType);
        const startTime = performance.now();

        try {
            const result = await this.generate(prompt, {
                max_tokens: analysisType === 'quick' ? 512 : 2048,
                temperature: 0.1,
                use_cache: true
            });

            if (!result.success) {
                throw new Error(result.error || 'Generation failed');
            }

            const analysis = this.parseLegalAnalysisResponse(result.text || '');
            const processingTime = performance.now() - startTime;

            return {
                ...analysis,
                processingTime,
                method: result.method?.includes('WebAssembly') ? 'webassembly' : 'ollama'
            };

        } catch (error: any) {
            console.error('[Gemma3Service] Document analysis failed:', error);
            return {
                summary: 'Analysis failed due to technical error',
                keyTerms: [],
                entities: [],
                risks: [{
                    type: 'system',
                    severity: 'medium',
                    description: 'Unable to complete automated analysis'
                }],
                recommendations: ['Manual review recommended'],
                confidence: 0,
                processingTime: performance.now() - startTime,
                method: 'ollama'
            };
        }
    }

    /**
     * Generate embeddings using nomic-embed-text via Ollama
     */
    async generateEmbeddings(text: string): Promise<EmbeddingResult> {
        const startTime = performance.now();

        try {
            const response = await fetch(`${this.ollamaBaseUrl}/api/embeddings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: this.nomicModel,
                    prompt: text
                })
            });

            if (!response.ok) {
                throw new Error(`Embeddings request failed: ${response.statusText}`);
            }

            const data = await response.json();
            const processingTime = performance.now() - startTime;

            return {
                embedding: data.embedding || [],
                dimensions: data.embedding?.length || 384,
                model: this.nomicModel,
                processingTime
            };

        } catch (error: any) {
            console.error('[Gemma3Service] Embedding generation failed:', error);
            throw error;
        }
    }

    /**
     * Get comprehensive service statistics
     */
    getServiceStats() {
        const wasmStats = this.engine?.getPerformanceStats();
        
        return {
            initialized: this.initialized,
            wasmAvailable: !!this.engine,
            modelLoaded: this.modelLoaded,
            ollamaHealthy: this.stats.ollamaRequests > 0,
            
            requests: {
                total: this.stats.totalRequests,
                webAssembly: this.stats.wasmRequests,
                ollama: this.stats.ollamaRequests,
                cacheHits: this.stats.cacheHits
            },
            
            performance: {
                averageResponseTime: Math.round(this.stats.avgResponseTime),
                wasmTokensPerSecond: wasmStats?.average_tokens_per_second || 0,
                memoryUsage: wasmStats?.memory_usage_mb || 0
            },
            
            configuration: {
                webGPUEnabled: this.config.enableWebGPU,
                threadingEnabled: this.config.enableThreading,
                cacheSize: this.config.maxCacheSize
            }
        };
    }

    /**
     * Streaming generation for real-time responses
     */
    async *generateStream(
        prompt: string, 
        options: Gemma3GenerationOptions = {}
    ): AsyncGenerator<{ text: string; done: boolean }> {
        // For WebAssembly, we'll simulate streaming by chunking the response
        if (this.engine && this.modelLoaded) {
            const result = await this.generateWithWebAssembly(prompt, options);
            
            if (result.success && result.text) {
                // Simulate streaming by yielding words progressively
                const words = result.text.split(' ');
                let currentText = '';
                
                for (const word of words) {
                    currentText += (currentText ? ' ' : '') + word;
                    yield { text: currentText, done: false };
                    
                    // Small delay to simulate real streaming
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
                
                yield { text: currentText, done: true };
            }
            return;
        }

        // For Ollama, use native streaming
        const response = await fetch(`${this.ollamaBaseUrl}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: this.ollamaModel,
                prompt,
                options: {
                    num_predict: options.max_tokens || 1024,
                    temperature: options.temperature || this.config.defaultTemperature,
                },
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`Streaming request failed: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        if (reader) {
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        yield { text: fullText, done: true };
                        break;
                    }

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n').filter(line => line.trim());

                    for (const line of lines) {
                        try {
                            const data = JSON.parse(line);
                            if (data.response) {
                                fullText += data.response;
                                yield { text: fullText, done: data.done || false };
                            }
                        } catch (e: any) {
                            // Skip invalid JSON lines
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        }
    }

    /**
     * Clean up resources
     */
    dispose(): void {
        if (this.engine && this.wasmModule) {
            // Clean up WebAssembly resources
            this.engine = null;
            this.wasmModule = null;
        }
        
        this.initialized = false;
        this.modelLoaded = false;
        
        console.log('[Gemma3Service] Resources disposed');
    }

    // Private helper methods

    private buildLegalAnalysisPrompt(title: string, content: string, analysisType: string): string {
        const instructions = {
            comprehensive: 'Provide detailed analysis of all legal aspects including risks, compliance, and recommendations',
            quick: 'Provide concise summary focusing on key legal points and immediate concerns',
            'risk-focused': 'Focus specifically on identifying legal risks, compliance issues, and potential liabilities'
        };

        return `<|system|>You are a specialized legal AI assistant with expertise in contract analysis, compliance review, and risk assessment. Analyze the following legal document comprehensively.

Instructions: ${instructions[analysisType as keyof typeof instructions]}

Document Title: ${title}

Document Content:
${content.substring(0, 8000)}

Please provide your analysis in the following structured format:

<analysis>
<summary>[Provide a clear, comprehensive summary of the document]</summary>
<key_terms>[List important legal terms, separated by commas]</key_terms>
<entities>[List entities in TYPE:VALUE:CONFIDENCE format, one per line]</entities>
<risks>[List risks in TYPE:SEVERITY:DESCRIPTION format, one per line]</risks>
<recommendations>[List specific recommendations, one per line]</recommendations>
<confidence>[Provide confidence score from 0.0 to 1.0]</confidence>
</analysis>

<|assistant|>`;
    }

    private parseLegalAnalysisResponse(response: string): Omit<LegalAnalysisResult, 'processingTime' | 'method'> {
        const defaultAnalysis = {
            summary: '',
            keyTerms: [],
            entities: [],
            risks: [],
            recommendations: [],
            confidence: 0.5
        };

        try {
            // Extract structured sections
            const summaryMatch = response.match(/<summary>(.*?)<\/summary>/s);
            if (summaryMatch) {
                defaultAnalysis.summary = summaryMatch[1].trim();
            }

            const keyTermsMatch = response.match(/<key_terms>(.*?)<\/key_terms>/s);
            if (keyTermsMatch) {
                defaultAnalysis.keyTerms = keyTermsMatch[1]
                    .split(',')
                    .map(term => term.trim())
                    .filter(term => term);
            }

            const entitiesMatch = response.match(/<entities>(.*?)<\/entities>/s);
            if (entitiesMatch) {
                defaultAnalysis.entities = entitiesMatch[1]
                    .split('\n')
                    .filter(line => line.trim())
                    .map(line => {
                        const [type, value, confidenceStr] = line.split(':').map(s => s.trim());
                        return {
                            type: type || 'unknown',
                            value: value || '',
                            confidence: parseFloat(confidenceStr || '0.8')
                        };
                    })
                    .filter(entity => entity.value);
            }

            const risksMatch = response.match(/<risks>(.*?)<\/risks>/s);
            if (risksMatch) {
                defaultAnalysis.risks = risksMatch[1]
                    .split('\n')
                    .filter(line => line.trim())
                    .map(line => {
                        const [type, severity, description] = line.split(':').map(s => s.trim());
                        return {
                            type: type || 'general',
                            severity: (severity?.toLowerCase() as any) || 'medium',
                            description: description || ''
                        };
                    })
                    .filter(risk => risk.description);
            }

            const recommendationsMatch = response.match(/<recommendations>(.*?)<\/recommendations>/s);
            if (recommendationsMatch) {
                defaultAnalysis.recommendations = recommendationsMatch[1]
                    .split('\n')
                    .map(rec => rec.trim())
                    .filter(rec => rec);
            }

            const confidenceMatch = response.match(/<confidence>(.*?)<\/confidence>/s);
            if (confidenceMatch) {
                const confidence = parseFloat(confidenceMatch[1].trim());
                defaultAnalysis.confidence = isNaN(confidence) ? 0.5 : Math.max(0, Math.min(1, confidence));
            }

        } catch (error: any) {
            console.error('[Gemma3Service] Failed to parse analysis response:', error);
        }

        return defaultAnalysis;
    }

    private estimateTokens(text: string): number {
        // Rough estimation: ~4 characters per token for English text
        return Math.ceil(text.length / 4);
    }
}

// Export singleton instance for global use
export const gemma3Service = new Gemma3LocalService();

// Export types
export type { LegalAnalysisResult, EmbeddingResult, Gemma3ServiceConfig };