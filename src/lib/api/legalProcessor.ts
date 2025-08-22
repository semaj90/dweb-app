
// Enhanced GPU+SIMD Legal API Client
export class LegalProcessorAPI {
    private baseUrl: string;

    constructor(baseUrl = '/api/legal') {
        this.baseUrl = baseUrl;
    }

    async similaritySearch(params: {
        queryEmbedding: number[];
        documentEmbeddings: number[][];
        documentIds: string[];
        topK?: number;
        useGPU?: boolean;
    }) {
        return this.post('similarity-search', {
            ...params,
            topK: params.topK || 10,
            useGPU: params.useGPU ?? true
        });
    }

    async ragSearch(params: {
        query: string;
        topK?: number;
        useGPU?: boolean;
        llmProvider?: 'ollama' | 'claude' | 'gemini' | 'llamacpp';
    }) {
        return this.post('rag-search', {
            ...params,
            topK: params.topK || 10,
            useGPU: params.useGPU ?? true,
            llmProvider: params.llmProvider || 'ollama'
        });
    }

    async llmRequest(params: {
        provider: string;
        model?: string;
        prompt: string;
        format?: string;
        options?: unknown;
    }) {
        return this.post('llm-request', params);
    }

    async analyzeDocument(params: {
        filePath: string;
        content: string;
        llmProvider?: string;
    }) {
        return this.post('analyze-document', {
            ...params,
            llmProvider: params.llmProvider || 'ollama'
        });
    }

    async batchProcessFiles(filePaths: string[]) {
        return this.post('batch-process-files', { filePaths });
    }

    async getHealth() {
        return this.get('health');
    }

    async getLLMEndpoints() {
        return this.get('llm-endpoints');
    }

    private async post(endpoint: string, data: unknown) {
        const response = await fetch(this.baseUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ endpoint, ...data })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return response.json();
    }

    private async get(endpoint: string) {
        const response = await fetch(`${this.baseUrl}?endpoint=${endpoint}`);
        return response.json();
    }
}

// Global instance
export const legalAPI = new LegalProcessorAPI();

// Types
export interface SimilarityResult {
    documentId: string;
    score: number;
    rank: number;
}

export interface LLMResponse {
    provider: string;
    model: string;
    response: string;
    timingMs: number;
    error?: string;
}

export interface AnalysisReport {
    filePath: string;
    severity: 'high' | 'medium' | 'low';
    issueSummary: string;
    recommendations: string[];
    todoList: string[];
    processingTimeMs: number;
    method: string;
}