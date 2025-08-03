import * as vscode from 'vscode';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

// Import simdjson for high-performance JSON parsing
// Note: Using dynamic import to handle potential module loading issues
let simdjson: any = null;

export interface SIMDParseOptions {
    enableWorkerThreads: boolean;
    workerPoolSize: number;
    batchSize: number;
    timeout: number;
    validationLevel: 'none' | 'basic' | 'strict';
}

export interface ParseTask {
    id: string;
    jsonData: string;
    options: SIMDParseOptions;
}

export interface ParseResult {
    success: boolean;
    data?: any;
    error?: string;
    processingTime: number;
    memoryUsage?: number;
    method: 'simd' | 'native' | 'worker';
}

export interface EvidenceData {
    id: string;
    type: 'document' | 'testimony' | 'exhibit' | 'correspondence';
    content: string;
    metadata: {
        source: string;
        timestamp: string;
        author?: string;
        case_id?: string;
        relevance_score?: number;
        [key: string]: any;
    };
    annotations?: Array<{
        type: string;
        text: string;
        position: { start: number; end: number };
        category?: string;
    }>;
}

export class SIMDJSONParser {
    private options: SIMDParseOptions;
    private workers: Worker[] = [];
    private initialized = false;

    constructor(options?: Partial<SIMDParseOptions>) {
        this.options = {
            enableWorkerThreads: options?.enableWorkerThreads ?? true,
            workerPoolSize: options?.workerPoolSize ?? 4,
            batchSize: options?.batchSize ?? 100,
            timeout: options?.timeout ?? 10000,
            validationLevel: options?.validationLevel ?? 'basic'
        };
    }

    async initialize(): Promise<void> {
        if (this.initialized) return;

        try {
            // Try to load simdjson
            simdjson = await this.loadSimdJson();
            
            // Initialize worker pool if enabled
            if (this.options.enableWorkerThreads) {
                this.initializeWorkers();
            }

            this.initialized = true;
            console.log('SIMD JSON Parser initialized successfully');
        } catch (error) {
            console.warn('SIMD JSON initialization failed, falling back to native JSON:', error);
            // Set up fallback mode
            simdjson = null;
            this.options.enableWorkerThreads = false;
            this.initialized = true;
        }
    }

    private async loadSimdJson(): Promise<any> {
        try {
            // Dynamic import to handle potential module loading issues
            const simdModule = await import('simdjson');
            return simdModule;
        } catch (error) {
            console.warn('Failed to load simdjson module:', error);
            return null;
        }
    }

    private initializeWorkers(): void {
        for (let i = 0; i < this.options.workerPoolSize; i++) {
            const worker = new Worker(__filename, {
                workerData: { workerId: i, options: this.options }
            });

            worker.on('error', (error) => {
                console.error(`SIMD JSON worker ${i} error:`, error);
            });

            this.workers.push(worker);
        }
    }

    /**
     * Parse JSON evidence with SIMD optimization
     */
    async parseEvidence(jsonData: string): Promise<ParseResult> {
        if (!this.initialized) {
            await this.initialize();
        }

        const startTime = Date.now();
        const initialMemory = process.memoryUsage().heapUsed;

        try {
            let result: any;
            let method: 'simd' | 'native' | 'worker' = 'native';

            // Try SIMD parsing first
            if (simdjson && jsonData.length > 1000) {
                try {
                    result = await this.parseWithSimd(jsonData);
                    method = 'simd';
                } catch (simdError) {
                    console.warn('SIMD parsing failed, falling back to native:', simdError);
                    result = JSON.parse(jsonData);
                }
            } 
            // Use worker threads for large data
            else if (this.options.enableWorkerThreads && jsonData.length > 10000) {
                result = await this.parseWithWorker(jsonData);
                method = 'worker';
            }
            // Native JSON parsing for smaller data
            else {
                result = JSON.parse(jsonData);
            }

            // Validate structure if enabled
            if (this.options.validationLevel !== 'none') {
                this.validateEvidenceStructure(result, this.options.validationLevel);
            }

            const processingTime = Date.now() - startTime;
            const finalMemory = process.memoryUsage().heapUsed;
            const memoryUsage = finalMemory - initialMemory;

            return {
                success: true,
                data: result,
                processingTime,
                memoryUsage,
                method
            };

        } catch (error) {
            const processingTime = Date.now() - startTime;
            
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error),
                processingTime,
                method: 'native'
            };
        }
    }

    private async parseWithSimd(jsonData: string): Promise<any> {
        if (!simdjson) {
            throw new Error('SIMD JSON not available');
        }

        // Convert string to buffer for simdjson
        const buffer = Buffer.from(jsonData, 'utf8');
        return simdjson.parse(buffer);
    }

    private parseWithWorker(jsonData: string): Promise<any> {
        return new Promise((resolve, reject) => {
            if (this.workers.length === 0) {
                throw new Error('No workers available');
            }

            const worker = this.workers[0]; // Simple worker selection
            const taskId = `parse_${Date.now()}_${Math.random()}`;

            const timeout = setTimeout(() => {
                reject(new Error('Worker parsing timeout'));
            }, this.options.timeout);

            const handleMessage = (message: any) => {
                if (message.taskId === taskId) {
                    clearTimeout(timeout);
                    worker.off('message', handleMessage);
                    
                    if (message.error) {
                        reject(new Error(message.error));
                    } else {
                        resolve(message.result);
                    }
                }
            };

            worker.on('message', handleMessage);
            worker.postMessage({
                taskId,
                jsonData,
                options: this.options
            });
        });
    }

    /**
     * Parse multiple evidence documents in batch
     */
    async parseEvidenceBatch(jsonDocuments: string[]): Promise<ParseResult[]> {
        if (!this.initialized) {
            await this.initialize();
        }

        const results: ParseResult[] = [];
        const chunks = this.chunkArray(jsonDocuments, this.options.batchSize);

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            
            // Show progress for large batches
            if (jsonDocuments.length > 10) {
                vscode.window.setStatusBarMessage(
                    `🔄 Parsing evidence batch ${i + 1}/${chunks.length}...`,
                    2000
                );
            }

            const chunkPromises = chunk.map(jsonDoc => this.parseEvidence(jsonDoc));
            const chunkResults = await Promise.all(chunkPromises);
            results.push(...chunkResults);
        }

        return results;
    }

    private chunkArray<T>(array: T[], chunkSize: number): T[][] {
        const chunks: T[][] = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    }

    private validateEvidenceStructure(data: any, level: 'basic' | 'strict'): void {
        if (level === 'basic') {
            // Basic validation: check for required fields
            if (typeof data !== 'object' || !data) {
                throw new Error('Evidence data must be an object');
            }
        } else if (level === 'strict') {
            // Strict validation: validate full evidence structure
            if (!this.isValidEvidenceData(data)) {
                throw new Error('Evidence data does not match expected structure');
            }
        }
    }

    private isValidEvidenceData(data: any): data is EvidenceData {
        return (
            typeof data === 'object' &&
            data !== null &&
            typeof data.id === 'string' &&
            ['document', 'testimony', 'exhibit', 'correspondence'].includes(data.type) &&
            typeof data.content === 'string' &&
            typeof data.metadata === 'object' &&
            typeof data.metadata.source === 'string' &&
            typeof data.metadata.timestamp === 'string'
        );
    }

    /**
     * Extract and parse evidence from various JSON formats
     */
    async extractEvidenceFromJson(jsonData: string): Promise<{
        evidence: EvidenceData[];
        parseResult: ParseResult;
        extractionMetadata: {
            totalItems: number;
            validEvidence: number;
            errors: string[];
        };
    }> {
        const parseResult = await this.parseEvidence(jsonData);
        
        if (!parseResult.success || !parseResult.data) {
            return {
                evidence: [],
                parseResult,
                extractionMetadata: {
                    totalItems: 0,
                    validEvidence: 0,
                    errors: [parseResult.error || 'Unknown parsing error']
                }
            };
        }

        const evidence: EvidenceData[] = [];
        const errors: string[] = [];
        let totalItems = 0;

        try {
            // Handle different JSON structures
            const data = parseResult.data;

            if (Array.isArray(data)) {
                // Array of evidence items
                totalItems = data.length;
                for (let i = 0; i < data.length; i++) {
                    try {
                        const evidenceItem = this.normalizeEvidenceData(data[i], i);
                        evidence.push(evidenceItem);
                    } catch (error) {
                        errors.push(`Item ${i}: ${error instanceof Error ? error.message : String(error)}`);
                    }
                }
            } else if (data.evidence && Array.isArray(data.evidence)) {
                // Wrapped in evidence property
                totalItems = data.evidence.length;
                for (let i = 0; i < data.evidence.length; i++) {
                    try {
                        const evidenceItem = this.normalizeEvidenceData(data.evidence[i], i);
                        evidence.push(evidenceItem);
                    } catch (error) {
                        errors.push(`Evidence ${i}: ${error instanceof Error ? error.message : String(error)}`);
                    }
                }
            } else {
                // Single evidence item
                totalItems = 1;
                try {
                    const evidenceItem = this.normalizeEvidenceData(data, 0);
                    evidence.push(evidenceItem);
                } catch (error) {
                    errors.push(`Single item: ${error instanceof Error ? error.message : String(error)}`);
                }
            }

            return {
                evidence,
                parseResult,
                extractionMetadata: {
                    totalItems,
                    validEvidence: evidence.length,
                    errors
                }
            };

        } catch (error) {
            return {
                evidence: [],
                parseResult,
                extractionMetadata: {
                    totalItems: 0,
                    validEvidence: 0,
                    errors: [`Extraction error: ${error instanceof Error ? error.message : String(error)}`]
                }
            };
        }
    }

    private normalizeEvidenceData(rawData: any, index: number): EvidenceData {
        // Generate ID if missing
        const id = rawData.id || `evidence_${Date.now()}_${index}`;
        
        // Determine type
        const type = rawData.type || this.inferEvidenceType(rawData);
        
        // Extract content
        const content = rawData.content || rawData.text || rawData.description || '';
        
        // Build metadata
        const metadata = {
            source: rawData.source || rawData.metadata?.source || 'unknown',
            timestamp: rawData.timestamp || rawData.metadata?.timestamp || new Date().toISOString(),
            ...rawData.metadata
        };

        // Extract annotations if present
        const annotations = rawData.annotations || [];

        return {
            id,
            type,
            content,
            metadata,
            annotations
        };
    }

    private inferEvidenceType(data: any): EvidenceData['type'] {
        const content = (data.content || data.text || '').toLowerCase();
        
        if (content.includes('testimony') || content.includes('witness')) {
            return 'testimony';
        }
        if (content.includes('exhibit') || data.exhibit_number) {
            return 'exhibit';
        }
        if (content.includes('email') || content.includes('letter') || data.from || data.to) {
            return 'correspondence';
        }
        
        return 'document';
    }

    /**
     * Get performance metrics
     */
    getPerformanceMetrics(): {
        simdAvailable: boolean;
        workersInitialized: number;
        configuration: SIMDParseOptions;
    } {
        return {
            simdAvailable: simdjson !== null,
            workersInitialized: this.workers.length,
            configuration: this.options
        };
    }

    dispose(): void {
        // Terminate all workers
        for (const worker of this.workers) {
            worker.terminate();
        }
        this.workers = [];
        this.initialized = false;
    }
}

// Worker thread implementation
if (!isMainThread) {
    const { workerId, options } = workerData;

    parentPort?.on('message', async (message) => {
        const { taskId, jsonData, options: taskOptions } = message;
        
        try {
            // Parse JSON in worker thread
            const result = JSON.parse(jsonData);
            
            parentPort?.postMessage({
                taskId,
                result,
                workerId
            });
        } catch (error) {
            parentPort?.postMessage({
                taskId,
                error: error instanceof Error ? error.message : String(error),
                workerId
            });
        }
    });
}