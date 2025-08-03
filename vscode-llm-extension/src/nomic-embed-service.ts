import * as vscode from 'vscode';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import fetch from 'node-fetch';

export interface EmbeddingRequest {
    texts: string[];
    model?: string;
    task_type?: string;
}

export interface EmbeddingResponse {
    embeddings: number[][];
    usage: {
        prompt_tokens: number;
        total_tokens: number;
    };
}

export interface WorkerEmbeddingTask {
    id: string;
    texts: string[];
    model: string;
    config: NomicEmbedConfig;
}

export interface NomicEmbedConfig {
    baseUrl: string;
    model: string;
    workerThreads: number;
    batchSize: number;
    timeout: number;
}

export class NomicEmbedService {
    private config: NomicEmbedConfig;
    private workers: Worker[] = [];
    private requestQueue: Array<{
        resolve: (value: EmbeddingResponse) => void;
        reject: (reason: any) => void;
        task: WorkerEmbeddingTask;
    }> = [];

    constructor() {
        this.config = this.loadConfiguration();
        this.initializeWorkers();
    }

    private loadConfiguration(): NomicEmbedConfig {
        const config = vscode.workspace.getConfiguration('mcpContext7');
        return {
            baseUrl: config.get('nomicEmbedUrl', 'http://localhost:8080'),
            model: config.get('embeddingModel', 'nomic-embed-text-v1.5'),
            workerThreads: config.get('workerThreads', 4),
            batchSize: config.get('batchSize', 32),
            timeout: 30000
        };
    }

    private initializeWorkers() {
        for (let i = 0; i < this.config.workerThreads; i++) {
            const worker = new Worker(__filename, {
                workerData: { workerId: i }
            });

            worker.on('message', this.handleWorkerMessage.bind(this));
            worker.on('error', this.handleWorkerError.bind(this));
            
            this.workers.push(worker);
        }
    }

    private handleWorkerMessage(message: any) {
        const queueIndex = this.requestQueue.findIndex(item => item.task.id === message.taskId);
        if (queueIndex !== -1) {
            const { resolve, reject } = this.requestQueue[queueIndex];
            this.requestQueue.splice(queueIndex, 1);

            if (message.error) {
                reject(new Error(message.error));
            } else {
                resolve(message.result);
            }
        }
    }

    private handleWorkerError(error: Error) {
        console.error('Nomic Embed worker error:', error);
        vscode.window.showErrorMessage(`Embedding worker error: ${error.message}`);
    }

    /**
     * Embed multiple texts using multi-core processing
     */
    async embedTexts(texts: string[]): Promise<EmbeddingResponse> {
        const chunks = this.chunkTexts(texts, this.config.batchSize);
        const embeddings: number[][] = [];
        let totalTokens = 0;

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Embedding ${texts.length} texts with ${this.config.workerThreads} workers...`,
            cancellable: false
        }, async (progress) => {
            const chunkPromises = chunks.map(async (chunk, index) => {
                const task: WorkerEmbeddingTask = {
                    id: `embed_${Date.now()}_${index}`,
                    texts: chunk,
                    model: this.config.model,
                    config: this.config
                };

                progress.report({
                    increment: (100 / chunks.length),
                    message: `Processing chunk ${index + 1}/${chunks.length}`
                });

                return this.processEmbeddingTask(task);
            });

            const results = await Promise.all(chunkPromises);
            
            // Combine results
            for (const result of results) {
                embeddings.push(...result.embeddings);
                totalTokens += result.usage.total_tokens;
            }
        });

        return {
            embeddings,
            usage: {
                prompt_tokens: totalTokens,
                total_tokens: totalTokens
            }
        };
    }

    private chunkTexts(texts: string[], batchSize: number): string[][] {
        const chunks: string[][] = [];
        for (let i = 0; i < texts.length; i += batchSize) {
            chunks.push(texts.slice(i, i + batchSize));
        }
        return chunks;
    }

    private processEmbeddingTask(task: WorkerEmbeddingTask): Promise<EmbeddingResponse> {
        return new Promise((resolve, reject) => {
            // Find available worker (simple round-robin for now)
            const workerIndex = this.requestQueue.length % this.workers.length;
            const worker = this.workers[workerIndex];

            this.requestQueue.push({ resolve, reject, task });
            worker.postMessage(task);

            // Timeout handling
            setTimeout(() => {
                const queueIndex = this.requestQueue.findIndex(item => item.task.id === task.id);
                if (queueIndex !== -1) {
                    this.requestQueue.splice(queueIndex, 1);
                    reject(new Error(`Embedding task timeout: ${task.id}`));
                }
            }, this.config.timeout);
        });
    }

    /**
     * Embed markdown files from workspace
     */
    async embedMarkdownFiles(): Promise<{
        files: { path: string; chunks: number }[];
        totalEmbeddings: number;
        processingTime: number;
    }> {
        const startTime = Date.now();
        const workspaceFiles = await this.findMarkdownFiles();
        
        if (workspaceFiles.length === 0) {
            throw new Error('No markdown files found in workspace');
        }

        let totalEmbeddings = 0;
        const processedFiles: { path: string; chunks: number }[] = [];

        for (const file of workspaceFiles) {
            const content = await this.readFileContent(file);
            const chunks = this.splitIntoChunks(content);
            
            if (chunks.length > 0) {
                await this.embedTexts(chunks);
                processedFiles.push({ path: file.fsPath, chunks: chunks.length });
                totalEmbeddings += chunks.length;
            }
        }

        const processingTime = Date.now() - startTime;
        
        return {
            files: processedFiles,
            totalEmbeddings,
            processingTime
        };
    }

    private async findMarkdownFiles(): Promise<vscode.Uri[]> {
        // Look for specific files mentioned in problem statement
        const targetFiles = ['copilot.md', 'claude.md'];
        const foundFiles: vscode.Uri[] = [];

        // Find all markdown files in workspace
        const allMarkdownFiles = await vscode.workspace.findFiles('**/*.md', '**/node_modules/**');
        
        // Prioritize copilot.md and claude.md
        for (const targetFile of targetFiles) {
            const matchingFile = allMarkdownFiles.find(file => 
                file.fsPath.toLowerCase().includes(targetFile.toLowerCase())
            );
            if (matchingFile) {
                foundFiles.push(matchingFile);
            }
        }

        // Add other markdown files if requested
        const config = vscode.workspace.getConfiguration('mcpContext7');
        const embedAllMarkdown = config.get('embedAllMarkdownFiles', false);
        
        if (embedAllMarkdown) {
            foundFiles.push(...allMarkdownFiles.filter(file => 
                !foundFiles.some(existing => existing.fsPath === file.fsPath)
            ));
        }

        return foundFiles;
    }

    private async readFileContent(file: vscode.Uri): Promise<string> {
        const document = await vscode.workspace.openTextDocument(file);
        return document.getText();
    }

    private splitIntoChunks(content: string, maxChunkSize: number = 1000): string[] {
        const chunks: string[] = [];
        const paragraphs = content.split('\n\n');
        let currentChunk = '';

        for (const paragraph of paragraphs) {
            if (currentChunk.length + paragraph.length > maxChunkSize && currentChunk.length > 0) {
                chunks.push(currentChunk.trim());
                currentChunk = paragraph;
            } else {
                currentChunk += (currentChunk ? '\n\n' : '') + paragraph;
            }
        }

        if (currentChunk.trim()) {
            chunks.push(currentChunk.trim());
        }

        return chunks.filter(chunk => chunk.length > 50); // Filter out very short chunks
    }

    dispose() {
        for (const worker of this.workers) {
            worker.terminate();
        }
        this.workers = [];
        this.requestQueue = [];
    }
}

// Worker thread implementation
if (!isMainThread) {
    const { workerId } = workerData;

    parentPort?.on('message', async (task: WorkerEmbeddingTask) => {
        try {
            const result = await performEmbedding(task);
            parentPort?.postMessage({
                taskId: task.id,
                result,
                workerId
            });
        } catch (error) {
            parentPort?.postMessage({
                taskId: task.id,
                error: error instanceof Error ? error.message : String(error),
                workerId
            });
        }
    });

    async function performEmbedding(task: WorkerEmbeddingTask): Promise<EmbeddingResponse> {
        const { texts, model, config } = task;

        const requestBody: EmbeddingRequest = {
            texts,
            model,
            task_type: 'search_document'
        };

        const response = await fetch(`${config.baseUrl}/v1/embeddings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Nomic Embed API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json() as EmbeddingResponse;
        
        if (!data.embeddings || !Array.isArray(data.embeddings)) {
            throw new Error('Invalid response format from Nomic Embed API');
        }

        return data;
    }
}