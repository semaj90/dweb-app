// High-performance local Gemma3 API endpoint with full-stack integration
// Integrates WebAssembly inference with MinIO, Neo4j, pgvector, and Redis

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { gemma3Service } from '$lib/services/gemma3-local-service';
import { evidenceProcessingMachine } from '$lib/state/evidenceProcessingMachine';
import { interpret } from 'xstate';
import { minioService } from '$lib/services/minio-service';
import { neo4jService } from '$lib/services/neo4j-service';
import { pgVectorService } from '$lib/services/pgvector-service';
import { redisService } from '$lib/services/redis-service';

// Service instances
const evidenceProcessor = interpret(evidenceProcessingMachine).start();
let serviceInitialized = false;

// Initialize services on first request
async function ensureServicesInitialized(): Promise<void> {
    if (serviceInitialized) return;
    
    try {
        // Initialize core services in parallel
        await Promise.all([
            gemma3Service.initialize(),
            minioService.initialize(),
            neo4jService.initialize(),
            pgVectorService.initialize(),
            redisService.initialize()
        ]);
        
        serviceInitialized = true;
        console.log('[Gemma3API] All services initialized successfully');
        
    } catch (err) {
        console.error('[Gemma3API] Service initialization failed:', err);
        throw err;
    }
}

/**
 * POST /api/ai/gemma3-local - Generate text with local inference
 */
export const POST: RequestHandler = async ({ request, url }) => {
    await ensureServicesInitialized();
    
    const searchParams = url.searchParams;
    const endpoint = searchParams.get('endpoint') || 'generate';
    
    try {
        const body = await request.json();
        
        switch (endpoint) {
            case 'generate':
                return await handleGenerate(body);
            case 'analyze-document':
                return await handleDocumentAnalysis(body);
            case 'stream':
                return await handleStreamGenerate(body);
            case 'embeddings':
                return await handleEmbeddings(body);
            case 'upload-and-analyze':
                return await handleUploadAndAnalyze(body);
            case 'evidence-processing':
                return await handleEvidenceProcessing(body);
            default:
                throw error(400, `Unknown endpoint: ${endpoint}`);
        }
        
    } catch (err) {
        console.error('[Gemma3API] Request failed:', err);
        throw error(500, {
            message: err instanceof Error ? err.message : 'Internal server error',
            code: 'GEMMA3_REQUEST_FAILED'
        });
    }
};

/**
 * Handle text generation requests
 */
async function handleGenerate(body: {
    prompt: string;
    options?: {
        max_tokens?: number;
        temperature?: number;
        top_p?: number;
        use_cache?: boolean;
        context_id?: string;
    };
}): Promise<Response> {
    const { prompt, options = {} } = body;
    
    if (!prompt?.trim()) {
        throw error(400, 'Prompt is required');
    }
    
    // Check Redis cache first if context_id provided
    let cacheKey: string | null = null;
    if (options.context_id) {
        cacheKey = `gemma3:generate:${options.context_id}:${hashString(prompt)}`;
        const cached = await redisService.get(cacheKey);
        if (cached) {
            return json({
                ...cached,
                cached: true,
                timestamp: Date.now()
            });
        }
    }
    
    // Generate with Gemma3 service
    const result = await gemma3Service.generate(prompt, {
        max_tokens: options.max_tokens || 1024,
        temperature: options.temperature || 0.1,
        top_p: options.top_p || 0.9,
        use_cache: options.use_cache !== false
    });
    
    // Cache result if successful and context_id provided
    if (result.success && cacheKey) {
        await redisService.setex(cacheKey, 3600, result); // 1 hour TTL
    }
    
    return json({
        ...result,
        timestamp: Date.now(),
        cached: false
    });
}

/**
 * Handle document analysis with full pipeline integration
 */
async function handleDocumentAnalysis(body: {
    title: string;
    content: string;
    analysisType?: 'comprehensive' | 'quick' | 'risk-focused';
    storeResults?: boolean;
    evidenceId?: string;
    userId?: string;
}): Promise<Response> {
    const { 
        title, 
        content, 
        analysisType = 'comprehensive', 
        storeResults = false,
        evidenceId,
        userId 
    } = body;
    
    if (!title?.trim() || !content?.trim()) {
        throw error(400, 'Title and content are required');
    }
    
    const startTime = performance.now();
    
    try {
        // Analyze document with Gemma3
        const analysis = await gemma3Service.analyzeDocument(title, content, analysisType);
        
        // Generate embeddings for semantic search
        const embeddings = await gemma3Service.generateEmbeddings(
            `${title}\n\n${content.substring(0, 1000)}`
        );
        
        let storageResults = null;
        
        // Store results in integrated systems if requested
        if (storeResults && userId) {
            const documentId = evidenceId || crypto.randomUUID();
            
            // Parallel storage operations
            const storagePromises = [
                // Store in pgvector for semantic search
                pgVectorService.storeDocument({
                    id: documentId,
                    title,
                    content,
                    embedding: embeddings.embedding,
                    metadata: {
                        analysis: analysis,
                        userId,
                        timestamp: new Date().toISOString(),
                        analysisType
                    }
                }),
                
                // Store in Neo4j for graph relationships
                neo4jService.createDocumentNode({
                    id: documentId,
                    title,
                    userId,
                    entities: analysis.entities,
                    keyTerms: analysis.keyTerms,
                    risks: analysis.risks
                }),
                
                // Store raw content in MinIO
                minioService.uploadDocument({
                    bucketName: 'legal-documents',
                    objectName: `${userId}/${documentId}.txt`,
                    content: content,
                    metadata: {
                        title,
                        userId,
                        analysisType,
                        timestamp: new Date().toISOString()
                    }
                }),
                
                // Cache analysis in Redis
                redisService.setex(
                    `analysis:${documentId}`,
                    86400, // 24 hours
                    analysis
                )
            ];
            
            storageResults = await Promise.allSettled(storagePromises);
            
            // Update evidence processing state if evidenceId provided
            if (evidenceId) {
                evidenceProcessor.send({
                    type: 'ANALYSIS_COMPLETE',
                    data: {
                        evidenceId,
                        analysis,
                        embeddings: embeddings.embedding,
                        storageStatus: storageResults
                    }
                });
            }
        }
        
        const totalTime = performance.now() - startTime;
        
        return json({
            success: true,
            analysis,
            embeddings: {
                dimensions: embeddings.dimensions,
                model: embeddings.model
            },
            processing: {
                totalTime: Math.round(totalTime),
                analysisTime: Math.round(analysis.processingTime),
                embeddingTime: Math.round(embeddings.processingTime),
                method: analysis.method
            },
            storage: storeResults ? {
                stored: storageResults?.every(r => r.status === 'fulfilled'),
                documentId: evidenceId || crypto.randomUUID(),
                results: storageResults?.map(r => ({
                    status: r.status,
                    error: r.status === 'rejected' ? r.reason?.message : undefined
                }))
            } : null,
            timestamp: Date.now()
        });
        
    } catch (err) {
        console.error('[Gemma3API] Document analysis failed:', err);
        throw error(500, {
            message: err instanceof Error ? err.message : 'Analysis failed',
            code: 'DOCUMENT_ANALYSIS_FAILED'
        });
    }
}

/**
 * Handle streaming text generation
 */
async function handleStreamGenerate(body: {
    prompt: string;
    options?: {
        max_tokens?: number;
        temperature?: number;
        use_cache?: boolean;
    };
}): Promise<Response> {
    const { prompt, options = {} } = body;
    
    if (!prompt?.trim()) {
        throw error(400, 'Prompt is required');
    }
    
    // Create readable stream for SSE
    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            
            try {
                let fullText = '';
                
                for await (const chunk of gemma3Service.generateStream(prompt, {
                    max_tokens: options.max_tokens || 1024,
                    temperature: options.temperature || 0.1,
                    use_cache: options.use_cache !== false
                })) {
                    const data = {
                        text: chunk.text,
                        delta: chunk.text.substring(fullText.length),
                        done: chunk.done,
                        timestamp: Date.now()
                    };
                    
                    fullText = chunk.text;
                    
                    // Send SSE formatted data
                    const sseData = `data: ${JSON.stringify(data)}\n\n`;
                    controller.enqueue(encoder.encode(sseData));
                    
                    if (chunk.done) {
                        controller.close();
                        break;
                    }
                }
                
            } catch (err) {
                console.error('[Gemma3API] Streaming failed:', err);
                const errorData = {
                    error: err instanceof Error ? err.message : 'Streaming failed',
                    done: true,
                    timestamp: Date.now()
                };
                
                const sseData = `data: ${JSON.stringify(errorData)}\n\n`;
                controller.enqueue(encoder.encode(sseData));
                controller.close();
            }
        }
    });
    
    return new Response(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

/**
 * Handle embeddings generation
 */
async function handleEmbeddings(body: {
    text: string;
    cacheKey?: string;
}): Promise<Response> {
    const { text, cacheKey } = body;
    
    if (!text?.trim()) {
        throw error(400, 'Text is required');
    }
    
    // Check Redis cache
    if (cacheKey) {
        const cached = await redisService.get(`embeddings:${cacheKey}`);
        if (cached) {
            return json({
                ...cached,
                cached: true,
                timestamp: Date.now()
            });
        }
    }
    
    const embeddings = await gemma3Service.generateEmbeddings(text);
    
    // Cache embeddings
    if (cacheKey) {
        await redisService.setex(`embeddings:${cacheKey}`, 7200, embeddings); // 2 hours
    }
    
    return json({
        ...embeddings,
        cached: false,
        timestamp: Date.now()
    });
}

/**
 * Handle file upload and analysis workflow
 */
async function handleUploadAndAnalyze(body: {
    fileName: string;
    fileContent: string;
    mimeType?: string;
    userId: string;
    analysisType?: 'comprehensive' | 'quick' | 'risk-focused';
}): Promise<Response> {
    const { 
        fileName, 
        fileContent, 
        mimeType = 'text/plain',
        userId,
        analysisType = 'comprehensive'
    } = body;
    
    if (!fileName?.trim() || !fileContent?.trim() || !userId?.trim()) {
        throw error(400, 'fileName, fileContent, and userId are required');
    }
    
    const documentId = crypto.randomUUID();
    const startTime = performance.now();
    
    try {
        // Start evidence processing workflow
        evidenceProcessor.send({
            type: 'START_PROCESSING',
            data: {
                evidenceId: documentId,
                fileName,
                userId,
                contentType: mimeType,
                timestamp: Date.now()
            }
        });
        
        // Perform document analysis with full pipeline
        const analysisResult = await handleDocumentAnalysis({
            title: fileName,
            content: fileContent,
            analysisType,
            storeResults: true,
            evidenceId: documentId,
            userId
        });
        
        const analysisData = await analysisResult.json();
        
        // Update evidence processing state
        evidenceProcessor.send({
            type: 'UPLOAD_COMPLETE',
            data: {
                evidenceId: documentId,
                uploadSuccess: true,
                analysisComplete: true
            }
        });
        
        const totalTime = performance.now() - startTime;
        
        return json({
            success: true,
            documentId,
            fileName,
            userId,
            analysis: analysisData.analysis,
            storage: analysisData.storage,
            processing: {
                ...analysisData.processing,
                totalWorkflowTime: Math.round(totalTime)
            },
            evidenceProcessingState: evidenceProcessor.getSnapshot().value,
            timestamp: Date.now()
        });
        
    } catch (err) {
        console.error('[Gemma3API] Upload and analysis failed:', err);
        
        // Update evidence processing with error
        evidenceProcessor.send({
            type: 'ERROR',
            data: {
                evidenceId: documentId,
                error: err instanceof Error ? err.message : 'Unknown error'
            }
        });
        
        throw error(500, {
            message: err instanceof Error ? err.message : 'Upload and analysis failed',
            code: 'UPLOAD_ANALYSIS_FAILED'
        });
    }
}

/**
 * Handle evidence processing state queries
 */
async function handleEvidenceProcessing(body: {
    action: 'status' | 'history' | 'reset';
    evidenceId?: string;
}): Promise<Response> {
    const { action, evidenceId } = body;
    
    const currentState = evidenceProcessor.getSnapshot();
    
    switch (action) {
        case 'status':
            return json({
                state: currentState.value,
                context: currentState.context,
                evidenceId: evidenceId || null,
                timestamp: Date.now()
            });
            
        case 'history':
            // Get processing history from Redis
            const historyKey = evidenceId ? `evidence:${evidenceId}:history` : 'evidence:global:history';
            const history = await redisService.lrange(historyKey, 0, -1);
            
            return json({
                history: history.map(item => JSON.parse(item)),
                evidenceId: evidenceId || null,
                timestamp: Date.now()
            });
            
        case 'reset':
            evidenceProcessor.send({ type: 'RESET' });
            return json({
                success: true,
                message: 'Evidence processing state reset',
                timestamp: Date.now()
            });
            
        default:
            throw error(400, `Unknown action: ${action}`);
    }
}

/**
 * GET /api/ai/gemma3-local - Get service status and stats
 */
export const GET: RequestHandler = async ({ url }) => {
    const searchParams = url.searchParams;
    const info = searchParams.get('info');
    
    if (info === 'health') {
        return json({
            healthy: serviceInitialized,
            services: {
                gemma3: await checkServiceHealth(() => gemma3Service.getServiceStats()),
                minio: await checkServiceHealth(() => minioService.getHealth()),
                neo4j: await checkServiceHealth(() => neo4jService.getHealth()),
                pgvector: await checkServiceHealth(() => pgVectorService.getHealth()),
                redis: await checkServiceHealth(() => redisService.ping())
            },
            evidenceProcessing: {
                state: evidenceProcessor.getSnapshot().value,
                active: !evidenceProcessor.getSnapshot().done
            },
            timestamp: Date.now()
        });
    }
    
    if (info === 'stats') {
        return json({
            gemma3: gemma3Service.getServiceStats(),
            evidenceProcessing: {
                state: evidenceProcessor.getSnapshot().value,
                context: evidenceProcessor.getSnapshot().context
            },
            timestamp: Date.now()
        });
    }
    
    return json({
        service: 'Gemma3 Local Inference API',
        version: '1.0.0',
        initialized: serviceInitialized,
        endpoints: [
            'POST ?endpoint=generate - Generate text',
            'POST ?endpoint=analyze-document - Analyze legal documents',
            'POST ?endpoint=stream - Stream text generation',
            'POST ?endpoint=embeddings - Generate embeddings',
            'POST ?endpoint=upload-and-analyze - Upload and analyze workflow',
            'POST ?endpoint=evidence-processing - Evidence processing state',
            'GET ?info=health - Service health check',
            'GET ?info=stats - Performance statistics'
        ],
        timestamp: Date.now()
    });
};

// Utility functions
async function checkServiceHealth(healthCheck: () => Promise<any>): Promise<{ healthy: boolean; error?: string }> {
    try {
        await healthCheck();
        return { healthy: true };
    } catch (err) {
        return { 
            healthy: false, 
            error: err instanceof Error ? err.message : 'Unknown error' 
        };
    }
}

function hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
}