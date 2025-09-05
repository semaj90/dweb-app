// GPU-Gemma3 Orchestrator: Unified WebAssembly + Node GPU Service Integration
// Combines local WebAssembly inference with high-performance GPU processing

// Import types with proper fallbacks
import { gemma3Service } from './gemma3-local-service';

// Define types locally to avoid import issues
export interface Gemma3GenerationResult {
    text: string;
    processingTime: number;
    tokensGenerated: number;
}

export interface Gemma3ServiceConfig {
    modelUrl?: string;
    wasmUrl?: string;
    enableWebGPU?: boolean;
    enableThreading?: boolean;
    maxCacheSize?: number;
    defaultTemperature?: number;
}

export interface GPUServiceClient {
    processEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse>;
    performClustering(request: ClusteringRequest): Promise<ClusteringResponse>;
    computeSimilarity(request: SimilarityRequest): Promise<SimilarityResponse>;
    applyBoostTransform(request: BoostRequest): Promise<BoostResponse>;
    getHealthStatus(): Promise<HealthResponse>;
}

// HTTP-based GPU service client
class HTTPGPUServiceClient implements GPUServiceClient {
    constructor(private baseUrl: string) {}

    async processEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
        // Convert to GPU orchestrator format
        const gpuRequest = {
            jobId: globalThis.crypto?.randomUUID() || `job_${Date.now()}_${Math.random().toString(36).slice(2)}`,
            type: 'embedding',
            data: request.requests.map(r => r.text).join(' ').split(' ').map(Number),
            priority: 'normal'
        };

        const response = await fetch(`${this.baseUrl}/api/gpu/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gpuRequest)
        });

        if (!response.ok) {
            throw new Error(`GPU embedding failed: ${response.statusText}`);
        }

        const result = await response.json();
        return {
            embeddings: [{ values: result.result }],
            dimensions: result.result.length,
            processingTime: result.processingMs,
            batchSize: 1
        };
    }

    async performClustering(request: ClusteringRequest): Promise<ClusteringResponse> {
        // Use SOM training endpoint for clustering
        const gpuRequest = {
            jobId: globalThis.crypto?.randomUUID() || `job_${Date.now()}_${Math.random().toString(36).slice(2)}`,
            type: 'som_train',
            data: request.embeddings.map(e => e.values).flat(),
            options: {
                clusters: request.numClusters,
                iterations: request.maxIterations || 100
            },
            priority: 'normal'
        };

        const response = await fetch(`${this.baseUrl}/api/gpu/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gpuRequest)
        });

        if (!response.ok) {
            throw new Error(`GPU clustering failed: ${response.statusText}`);
        }

        const result = await response.json();
        return {
            assignments: result.result || [],
            centers: [],
            inertia: 0,
            iterations: request.maxIterations || 100,
            processingTime: result.processingMs
        };
    }

    async computeSimilarity(request: SimilarityRequest): Promise<SimilarityResponse> {
        const gpuRequest = {
            jobId: globalThis.crypto?.randomUUID() || `job_${Date.now()}_${Math.random().toString(36).slice(2)}`,
            type: 'similarity',
            data: [...request.embeddingsA[0].values, ...request.embeddingsB[0].values],
            options: {
                metric: request.metric || 'cosine',
                vector_a: request.embeddingsA[0].values,
                vector_b: request.embeddingsB[0].values
            },
            priority: 'normal'
        };

        const response = await fetch(`${this.baseUrl}/api/gpu/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gpuRequest)
        });

        if (!response.ok) {
            throw new Error(`GPU similarity failed: ${response.statusText}`);
        }

        const result = await response.json();
        return {
            scores: Array.isArray(result.result) ? result.result : [result.result],
            metric: request.metric || 'cosine',
            processingTime: result.processingMs
        };
    }

    async applyBoostTransform(request: BoostRequest): Promise<BoostResponse> {
        // Use rotation operation for boost transform
        const gpuRequest = {
            jobId: globalThis.crypto?.randomUUID() || `job_${Date.now()}_${Math.random().toString(36).slice(2)}`,
            type: 'rotation',
            data: request.embeddings[0].values,
            options: {
                boostFactors: request.boostFactors
            },
            priority: 'normal'
        };

        const response = await fetch(`${this.baseUrl}/api/gpu/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gpuRequest)
        });

        if (!response.ok) {
            throw new Error(`GPU boost transform failed: ${response.statusText}`);
        }

        const result = await response.json();
        return {
            transformedEmbeddings: [{ values: result.result }],
            boostFactors: request.boostFactors,
            processingTime: result.processingMs
        };
    }

    async getHealthStatus(): Promise<HealthResponse> {
        const response = await fetch(`${this.baseUrl}/api/gpu/status`);
        
        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }

        const result = await response.json();
        return {
            status: result.status === 'healthy' ? 'healthy' : 'unhealthy',
            uptime: Date.now() - (result.timestamp || Date.now()),
            metrics: {
                totalJobs: result.gpu_stats?.totalJobs?.toString() || '0',
                successfulJobs: result.gpu_stats?.successfulJobs?.toString() || '0',
                failedJobs: result.gpu_stats?.failedJobs?.toString() || '0',
                gpuModel: result.gpu_stats?.gpuModel || 'Unknown'
            }
        };
    }
}

export interface EmbeddingRequest {
    requests: Array<{ text: string; id?: string }>;
}

export interface EmbeddingResponse {
    embeddings: Array<{ values: number[] }>;
    dimensions: number;
    processingTime: number;
    batchSize: number;
}

export interface ClusteringRequest {
    embeddings: Array<{ values: number[] }>;
    numClusters: number;
    maxIterations?: number;
}

export interface ClusteringResponse {
    assignments: number[];
    centers: Array<{ values: number[] }>;
    inertia: number;
    iterations: number;
    processingTime: number;
}

export interface SimilarityRequest {
    embeddingsA: Array<{ values: number[] }>;
    embeddingsB: Array<{ values: number[] }>;
    metric?: 'cosine' | 'euclidean' | 'dot';
}

export interface SimilarityResponse {
    scores: number[];
    metric: string;
    processingTime: number;
}

export interface BoostRequest {
    embeddings: Array<{ values: number[] }>;
    boostFactors: number[];
}

export interface BoostResponse {
    transformedEmbeddings: Array<{ values: number[] }>;
    boostFactors: number[];
    processingTime: number;
}

export interface HealthResponse {
    status: string;
    uptime: number;
    metrics: Record<string, string>;
}

export interface DocumentProcessingPipeline {
    documentId: string;
    content: string;
    title: string;
    metadata: Record<string, any>;
    stage: 'preprocessing' | 'embedding' | 'analysis' | 'clustering' | 'storage' | 'complete';
    results?: {
        embeddings?: number[];
        analysis?: any;
        clusters?: number[];
        similarities?: number[];
        summary?: string;
    };
}

export interface OrchestrationConfig extends Gemma3ServiceConfig {
    nodeGpuServiceUrl?: string; // GPU orchestrator service
    enhancedRagServiceUrl?: string; // Enhanced RAG service  
    enableGpuAcceleration?: boolean;
    maxBatchSize?: number;
    clusteringThreshold?: number;
    similarityThreshold?: number;
    cacheResults?: boolean;
}

export class GPUGemma3Orchestrator {
    private gemma3Service = gemma3Service;
    private gpuClient: GPUServiceClient | null = null; // GPU orchestrator (8231)
    private ragClient: HTTPGPUServiceClient | null = null; // Enhanced RAG (8094)
    private config: Required<OrchestrationConfig & { enhancedRagServiceUrl: string }>;
    private initialized = false;
    private processingQueue: DocumentProcessingPipeline[] = [];
    private isProcessing = false;
    
    // Performance tracking
    private stats = {
        documentsProcessed: 0,
        totalProcessingTime: 0,
        gpuOperations: 0,
        wasmOperations: 0,
        cacheHits: 0,
        avgThroughput: 0
    };

    constructor(config: OrchestrationConfig = {}) {
        this.config = {
            modelUrl: config.modelUrl || '/models/gemma3-legal-weights.bin',
            wasmUrl: config.wasmUrl || '/static/wasm/gemma3-inference.js',
            enableWebGPU: config.enableWebGPU ?? true,
            enableThreading: config.enableThreading ?? true,
            maxCacheSize: config.maxCacheSize || 100,
            defaultTemperature: config.defaultTemperature || 0.1,
            nodeGpuServiceUrl: config.nodeGpuServiceUrl || 'http://localhost:8231',
            enhancedRagServiceUrl: config.enhancedRagServiceUrl || 'http://localhost:8094',
            enableGpuAcceleration: config.enableGpuAcceleration ?? true,
            maxBatchSize: config.maxBatchSize || 32,
            clusteringThreshold: config.clusteringThreshold || 0.7,
            similarityThreshold: config.similarityThreshold || 0.8,
            cacheResults: config.cacheResults ?? true
        };
    }

    async initialize(): Promise<boolean> {
        console.log('[GPUGemma3Orchestrator] Initializing unified inference system...');
        
        try {
            // Initialize Gemma3 WebAssembly service
            const gemma3Ready = await this.gemma3Service.initialize();
            if (!gemma3Ready) {
                console.warn('[GPUGemma3Orchestrator] Gemma3 service failed to initialize');
            }

            // Initialize both GPU services if enabled
            if (this.config.enableGpuAcceleration) {
                await Promise.all([
                    this.initializeGPUService(), // GPU orchestrator
                    this.initializeRAGService()  // Enhanced RAG
                ]);
            }

            // Start processing loop
            this.startProcessingLoop();

            this.initialized = true;
            console.log('[GPUGemma3Orchestrator] Unified system initialized successfully');
            return true;

        } catch (error: any) {
            console.error('[GPUGemma3Orchestrator] Initialization failed:', error);
            return false;
        }
    }

    private async initializeGPUService(): Promise<any> {
        try {
            console.log('[GPUGemma3Orchestrator] Connecting to GPU Orchestrator (port 8231)...');
            
            // Create HTTP-based GPU client for orchestrator
            this.gpuClient = new HTTPGPUServiceClient(this.config.nodeGpuServiceUrl);

            // Test connection
            const health = await this.gpuClient.getHealthStatus();
            if (health.status !== 'healthy') {
                throw new Error(`GPU orchestrator unhealthy: ${health.status}`);
            }

            console.log('[GPUGemma3Orchestrator] GPU Orchestrator connected successfully');
            console.log(`GPU Orchestrator - ${health.metrics.gpuModel} (${health.metrics.totalJobs} jobs completed)`);

        } catch (error: any) {
            console.warn('[GPUGemma3Orchestrator] GPU Orchestrator connection failed:', error);
            this.gpuClient = null;
        }
    }

    private async initializeRAGService(): Promise<any> {
        try {
            console.log('[GPUGemma3Orchestrator] Connecting to Enhanced RAG Service (port 8094)...');
            
            // Create HTTP client for enhanced RAG service
            this.ragClient = new HTTPGPUServiceClient(this.config.enhancedRagServiceUrl);

            // Test connection with system metrics
            const response = await fetch(`${this.config.enhancedRagServiceUrl}/api/system/metrics`);
            if (!response.ok) {
                throw new Error(`RAG service metrics unavailable: ${response.statusText}`);
            }

            const metrics = await response.json();
            console.log('[GPUGemma3Orchestrator] Enhanced RAG Service connected successfully');
            console.log(`RAG Service - Database: ${metrics.database?.connected ? 'Connected' : 'Disconnected'}, GPU: ${metrics.services?.gpu?.available ? 'Available' : 'Unavailable'}`);

        } catch (error: any) {
            console.warn('[GPUGemma3Orchestrator] Enhanced RAG Service connection failed:', error);
            this.ragClient = null;
        }
    }

    /**
     * Process legal document with full pipeline: Analysis + Embeddings + Clustering
     */
    async processLegalDocument(
        title: string,
        content: string,
        options: {
            analysisType?: 'comprehensive' | 'quick' | 'risk-focused';
            generateEmbeddings?: boolean;
            performClustering?: boolean;
            findSimilarDocuments?: boolean;
            storeResults?: boolean;
            userId?: string;
        } = {}
    ): Promise<{
        documentId: string;
        analysis?: any;
        embeddings?: number[];
        clusters?: number[];
        similarDocuments?: Array<{ id: string; similarity: number; title: string }>;
        processing: {
            totalTime: number;
            stages: Record<string, number>;
            method: string;
        };
        performance: {
            tokensPerSecond?: number;
            embeddingTime?: number;
            clusteringTime?: number;
            gpuUtilization?: boolean;
        };
    }> {
        if (!this.initialized) {
            throw new Error('Orchestrator not initialized');
        }

        const documentId = globalThis.crypto?.randomUUID() || `doc_${Date.now()}_${Math.random().toString(36).slice(2)}`;
        const startTime = performance.now();
        const stages: Record<string, number> = {};
        
        console.log(`[GPUGemma3Orchestrator] Processing document: ${title}`);

        try {
            // Stage 1: Legal Analysis with Gemma3
            let analysis = null;
            if (options.analysisType !== undefined) {
                const analysisStart = performance.now();
                
                analysis = await this.gemma3Service.analyzeDocument(
                    title,
                    content,
                    options.analysisType
                );

                stages.analysis = performance.now() - analysisStart;
                console.log(`[GPUGemma3Orchestrator] Analysis completed in ${Math.round(stages.analysis)}ms`);
            }

            // Stage 2: Generate Embeddings (Choose best available service)
            let embeddings: number[] | null = null;
            if (options.generateEmbeddings !== false) {
                const embeddingStart = performance.now();
                
                if (this.ragClient) {
                    // Use Enhanced RAG service for embeddings (has Ollama integration)
                    try {
                        const ragResponse = await fetch(`${this.config.enhancedRagServiceUrl}/api/embeddings`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ text: content })
                        });
                        
                        if (ragResponse.ok) {
                            const result = await ragResponse.json();
                            embeddings = result.embedding;
                            console.log('[GPUGemma3Orchestrator] Using Enhanced RAG for embeddings');
                        }
                    } catch (error: any) {
                        console.warn('[GPUGemma3Orchestrator] RAG embedding failed, trying GPU orchestrator');
                    }
                }
                
                if (!embeddings && this.gpuClient) {
                    // Fallback to GPU orchestrator service
                    embeddings = await this.generateEmbeddingsGPU([content]);
                    this.stats.gpuOperations++;
                    console.log('[GPUGemma3Orchestrator] Using GPU Orchestrator for embeddings');
                } else if (!embeddings) {
                    // Final fallback to Ollama via Gemma3 service
                    const embeddingResult = await this.gemma3Service.generateEmbeddings(content);
                    embeddings = embeddingResult.embedding;
                    this.stats.wasmOperations++;
                    console.log('[GPUGemma3Orchestrator] Using Gemma3 service for embeddings');
                }

                stages.embeddings = performance.now() - embeddingStart;
                console.log(`[GPUGemma3Orchestrator] Embeddings generated in ${Math.round(stages.embeddings)}ms`);
            }

            // Stage 3: Clustering (if requested and embeddings available)
            let clusters: number[] | null = null;
            if (options.performClustering && embeddings && this.gpuClient) {
                const clusteringStart = performance.now();
                
                clusters = await this.performDocumentClustering([embeddings], 8);
                stages.clustering = performance.now() - clusteringStart;
                console.log(`[GPUGemma3Orchestrator] Clustering completed in ${Math.round(stages.clustering)}ms`);
            }

            // Stage 4: Find Similar Documents (if requested)
            let similarDocuments: Array<{ id: string; similarity: number; title: string }> | null = null;
            if (options.findSimilarDocuments && embeddings) {
                const similarityStart = performance.now();
                
                similarDocuments = await this.findSimilarDocuments(embeddings, {
                    limit: 10,
                    threshold: this.config.similarityThreshold
                });

                stages.similarity = performance.now() - similarityStart;
                console.log(`[GPUGemma3Orchestrator] Similarity search completed in ${Math.round(stages.similarity)}ms`);
            }

            const totalTime = performance.now() - startTime;
            
            // Update statistics
            this.stats.documentsProcessed++;
            this.stats.totalProcessingTime += totalTime;
            this.stats.avgThroughput = this.stats.documentsProcessed / (this.stats.totalProcessingTime / 1000);

            return {
                documentId,
                analysis: analysis || undefined,
                embeddings: embeddings || undefined,
                clusters: clusters || undefined,
                similarDocuments: similarDocuments || undefined,
                processing: {
                    totalTime: Math.round(totalTime),
                    stages,
                    method: this.getProcessingMethod()
                },
                performance: {
                    tokensPerSecond: analysis?.processingTime ? 
                        (content.length / 4) / (analysis.processingTime / 1000) : undefined,
                    embeddingTime: stages.embeddings,
                    clusteringTime: stages.clustering,
                    gpuUtilization: this.stats.gpuOperations > 0
                }
            };

        } catch (error: any) {
            console.error('[GPUGemma3Orchestrator] Document processing failed:', error);
            throw error;
        }
    }

    /**
     * Process batch of documents efficiently
     */
    async processBatchDocuments(
        documents: Array<{ title: string; content: string; metadata?: any }>,
        options: {
            analysisType?: 'comprehensive' | 'quick' | 'risk-focused';
            generateEmbeddings?: boolean;
            performClustering?: boolean;
            maxConcurrency?: number;
        } = {}
    ): Promise<{
        results: Array<{
            documentId: string;
            title: string;
            analysis?: any;
            embeddings?: number[];
            error?: string;
        }>;
        clustering?: {
            assignments: number[];
            centers: number[][];
            numClusters: number;
        };
        processing: {
            totalTime: number;
            documentsPerSecond: number;
            parallelization: boolean;
        };
    }> {
        if (!this.initialized) {
            throw new Error('Orchestrator not initialized');
        }

        const startTime = performance.now();
        const maxConcurrency = Math.min(
            options.maxConcurrency || 4,
            this.config.maxBatchSize
        );

        console.log(`[GPUGemma3Orchestrator] Processing batch: ${documents.length} documents`);

        try {
            // Process documents in parallel batches
            const results: any[] = [];
            const embeddings: number[][] = [];

            for (let i = 0; i < documents.length; i += maxConcurrency) {
                const batch = documents.slice(i, i + maxConcurrency);
                
                const batchPromises = batch.map(async (doc, idx): Promise<any> => {
                    try {
                        const result = await this.processLegalDocument(
                            doc.title,
                            doc.content,
                            {
                                analysisType: options.analysisType,
                                generateEmbeddings: options.generateEmbeddings,
                                performClustering: false // Do clustering at the end
                            }
                        );

                        if (result.embeddings) {
                            embeddings.push(result.embeddings);
                        }

                        return {
                            documentId: result.documentId,
                            title: doc.title,
                            analysis: result.analysis,
                            embeddings: result.embeddings
                        };

                    } catch (error: any) {
                        console.error(`[GPUGemma3Orchestrator] Failed to process document ${doc.title}:`, error);
                        return {
                            documentId: globalThis.crypto?.randomUUID() || `doc_${Date.now()}_${Math.random().toString(36).slice(2)}`,
                            title: doc.title,
                            error: error instanceof Error ? error.message : 'Unknown error'
                        };
                    }
                });

                const batchResults = await Promise.all(batchPromises);
                results.push(...batchResults);

                // Progress update
                console.log(`[GPUGemma3Orchestrator] Processed ${i + batch.length}/${documents.length} documents`);
            }

            // Perform clustering on all embeddings if requested
            let clustering = null;
            if (options.performClustering && embeddings.length > 0 && this.gpuClient) {
                console.log('[GPUGemma3Orchestrator] Performing batch clustering...');
                
                const numClusters = Math.min(8, Math.ceil(embeddings.length / 4));
                const clusterResult = await this.performDocumentClustering(embeddings, numClusters);
                
                clustering = {
                    assignments: clusterResult,
                    centers: [], // Would need to compute centers
                    numClusters
                };
            }

            const totalTime = performance.now() - startTime;
            const documentsPerSecond = documents.length / (totalTime / 1000);

            return {
                results,
                clustering,
                processing: {
                    totalTime: Math.round(totalTime),
                    documentsPerSecond: Math.round(documentsPerSecond),
                    parallelization: maxConcurrency > 1
                }
            };

        } catch (error: any) {
            console.error('[GPUGemma3Orchestrator] Batch processing failed:', error);
            throw error;
        }
    }

    /**
     * Generate embeddings using GPU acceleration
     */
    private async generateEmbeddingsGPU(texts: string[]): Promise<number[]> {
        if (!this.gpuClient) {
            throw new Error('GPU client not available');
        }

        const request: EmbeddingRequest = {
            requests: texts.map((text, i) => ({ text, id: i.toString() }))
        };

        const response = await this.gpuClient.processEmbeddings(request);
        
        // Return first embedding (for single text processing)
        return response.embeddings[0]?.values || [];
    }

    /**
     * Perform document clustering using GPU
     */
    private async performDocumentClustering(
        embeddings: number[][],
        numClusters: number
    ): Promise<number[]> {
        if (!this.gpuClient) {
            throw new Error('GPU client not available');
        }

        const request: ClusteringRequest = {
            embeddings: embeddings.map(emb => ({ values: emb })),
            numClusters,
            maxIterations: 100
        };

        const response = await this.gpuClient.performClustering(request);
        return response.assignments;
    }

    /**
     * Find similar documents using vector similarity
     */
    private async findSimilarDocuments(
        queryEmbedding: number[],
        options: { limit: number; threshold: number }
    ): Promise<Array<{ id: string; similarity: number; title: string }>> {
        // This would integrate with your existing pgvector/qdrant setup
        // For now, return empty array
        return [];
    }

    private getProcessingMethod(): string {
        const methods = [];
        
        if (this.stats.wasmOperations > 0) {
            methods.push('WebAssembly Gemma3');
        }
        
        if (this.stats.gpuOperations > 0) {
            methods.push('GPU Accelerated');
        }

        return methods.join(' + ') || 'CPU Only';
    }

    private startProcessingLoop(): void {
        setInterval(async (): Promise<any> => {
            if (!this.isProcessing && this.processingQueue.length > 0) {
                await this.processQueueBatch();
            }
        }, 100); // Check every 100ms
    }

    private async processQueueBatch(): Promise<any> {
        if (this.processingQueue.length === 0) return;

        this.isProcessing = true;
        const batch = this.processingQueue.splice(0, this.config.maxBatchSize);

        try {
            console.log(`[GPUGemma3Orchestrator] Processing queue batch: ${batch.length} documents`);
            
            // Process batch...
            // Implementation would handle the queued documents
            
        } catch (error: any) {
            console.error('[GPUGemma3Orchestrator] Queue batch processing failed:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Get comprehensive system statistics
     */
    getSystemStats() {
        const gemma3Stats = this.gemma3Service.getServiceStats();
        
        return {
            orchestrator: {
                initialized: this.initialized,
                documentsProcessed: this.stats.documentsProcessed,
                totalProcessingTime: Math.round(this.stats.totalProcessingTime),
                avgThroughput: Math.round(this.stats.avgThroughput * 100) / 100,
                gpuOperations: this.stats.gpuOperations,
                wasmOperations: this.stats.wasmOperations,
                queueLength: this.processingQueue.length
            },
            gemma3: gemma3Stats,
            gpu: {
                available: !!this.gpuClient,
                accelerated: this.config.enableGpuAcceleration,
                serviceUrl: this.config.nodeGpuServiceUrl
            },
            configuration: {
                maxBatchSize: this.config.maxBatchSize,
                enableWebGPU: this.config.enableWebGPU,
                enableThreading: this.config.enableThreading,
                clusteringThreshold: this.config.clusteringThreshold,
                similarityThreshold: this.config.similarityThreshold
            }
        };
    }

    /**
     * Add document to processing queue
     */
    queueDocument(pipeline: Omit<DocumentProcessingPipeline, 'stage'>): void {
        this.processingQueue.push({
            ...pipeline,
            stage: 'preprocessing'
        });
    }

    /**
     * Clean up resources
     */
    async dispose(): Promise<any> {
        console.log('[GPUGemma3Orchestrator] Disposing orchestrator...');
        
        this.initialized = false;
        this.isProcessing = false;
        this.processingQueue = [];
        
        if (this.gpuClient) {
            // Close gRPC connection
            this.gpuClient = null;
        }
        
        this.gemma3Service.dispose();
        
        console.log('[GPUGemma3Orchestrator] Orchestrator disposed');
    }
}

// Export singleton instance
export const gpuGemma3Orchestrator = new GPUGemma3Orchestrator();

// Export types
export type {
    DocumentProcessingPipeline,
    OrchestrationConfig,
    EmbeddingRequest,
    EmbeddingResponse,
    ClusteringRequest,
    ClusteringResponse
};