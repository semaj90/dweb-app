// GPU-Gemma3 Orchestrator: Unified WebAssembly + Node GPU Service Integration
// Combines local WebAssembly inference with high-performance GPU processing

import type { Gemma3GenerationResult, Gemma3ServiceConfig } from './gemma3-local-service';
import { gemma3Service } from './gemma3-local-service';
import { createChannel, createClientFactory } from 'nice-grpc';
import { NodeGPUServiceDefinition } from '../grpc/gpu-service';

interface GPUServiceClient {
    processEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse>;
    performClustering(request: ClusteringRequest): Promise<ClusteringResponse>;
    computeSimilarity(request: SimilarityRequest): Promise<SimilarityResponse>;
    applyBoostTransform(request: BoostRequest): Promise<BoostResponse>;
    getHealthStatus(): Promise<HealthResponse>;
}

interface EmbeddingRequest {
    requests: Array<{ text: string; id?: string }>;
}

interface EmbeddingResponse {
    embeddings: Array<{ values: number[] }>;
    dimensions: number;
    processingTime: number;
    batchSize: number;
}

interface ClusteringRequest {
    embeddings: Array<{ values: number[] }>;
    numClusters: number;
    maxIterations?: number;
}

interface ClusteringResponse {
    assignments: number[];
    centers: Array<{ values: number[] }>;
    inertia: number;
    iterations: number;
    processingTime: number;
}

interface SimilarityRequest {
    embeddingsA: Array<{ values: number[] }>;
    embeddingsB: Array<{ values: number[] }>;
    metric?: 'cosine' | 'euclidean' | 'dot';
}

interface SimilarityResponse {
    scores: number[];
    metric: string;
    processingTime: number;
}

interface BoostRequest {
    embeddings: Array<{ values: number[] }>;
    boostFactors: number[];
}

interface BoostResponse {
    transformedEmbeddings: Array<{ values: number[] }>;
    boostFactors: number[];
    processingTime: number;
}

interface HealthResponse {
    status: string;
    uptime: number;
    metrics: Record<string, string>;
}

interface DocumentProcessingPipeline {
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

interface OrchestrationConfig extends Gemma3ServiceConfig {
    nodeGpuServiceUrl?: string;
    enableGpuAcceleration?: boolean;
    maxBatchSize?: number;
    clusteringThreshold?: number;
    similarityThreshold?: number;
    cacheResults?: boolean;
}

export class GPUGemma3Orchestrator {
    private gemma3Service = gemma3Service;
    private gpuClient: GPUServiceClient | null = null;
    private config: Required<OrchestrationConfig>;
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
            nodeGpuServiceUrl: config.nodeGpuServiceUrl || 'localhost:50052',
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

            // Initialize GPU service client if enabled
            if (this.config.enableGpuAcceleration) {
                await this.initializeGPUService();
            }

            // Start processing loop
            this.startProcessingLoop();

            this.initialized = true;
            console.log('[GPUGemma3Orchestrator] Unified system initialized successfully');
            return true;

        } catch (error) {
            console.error('[GPUGemma3Orchestrator] Initialization failed:', error);
            return false;
        }
    }

    private async initializeGPUService(): Promise<void> {
        try {
            console.log('[GPUGemma3Orchestrator] Connecting to Node GPU service...');
            
            const channel = createChannel(this.config.nodeGpuServiceUrl);
            const clientFactory = createClientFactory()
                .use(/* Add interceptors if needed */);
            
            this.gpuClient = clientFactory.create(NodeGPUServiceDefinition, channel) as GPUServiceClient;

            // Test connection
            const health = await this.gpuClient.getHealthStatus();
            if (health.status !== 'healthy') {
                throw new Error(`GPU service unhealthy: ${health.status}`);
            }

            console.log('[GPUGemma3Orchestrator] GPU service connected successfully');
            console.log(`GPU Service Uptime: ${health.uptime}s`);

        } catch (error) {
            console.warn('[GPUGemma3Orchestrator] GPU service connection failed:', error);
            console.log('[GPUGemma3Orchestrator] Continuing without GPU acceleration');
            this.gpuClient = null;
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

        const documentId = crypto.randomUUID();
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

            // Stage 2: Generate Embeddings (GPU-accelerated if available)
            let embeddings: number[] | null = null;
            if (options.generateEmbeddings !== false) {
                const embeddingStart = performance.now();
                
                if (this.gpuClient) {
                    // Use GPU service for high-performance embedding generation
                    embeddings = await this.generateEmbeddingsGPU([content]);
                    this.stats.gpuOperations++;
                } else {
                    // Fallback to Ollama nomic-embed
                    const embeddingResult = await this.gemma3Service.generateEmbeddings(content);
                    embeddings = embeddingResult.embedding;
                    this.stats.wasmOperations++;
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

        } catch (error) {
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
                
                const batchPromises = batch.map(async (doc, idx) => {
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

                    } catch (error) {
                        console.error(`[GPUGemma3Orchestrator] Failed to process document ${doc.title}:`, error);
                        return {
                            documentId: crypto.randomUUID(),
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

        } catch (error) {
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
        setInterval(async () => {
            if (!this.isProcessing && this.processingQueue.length > 0) {
                await this.processQueueBatch();
            }
        }, 100); // Check every 100ms
    }

    private async processQueueBatch(): Promise<void> {
        if (this.processingQueue.length === 0) return;

        this.isProcessing = true;
        const batch = this.processingQueue.splice(0, this.config.maxBatchSize);

        try {
            console.log(`[GPUGemma3Orchestrator] Processing queue batch: ${batch.length} documents`);
            
            // Process batch...
            // Implementation would handle the queued documents
            
        } catch (error) {
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
    async dispose(): Promise<void> {
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