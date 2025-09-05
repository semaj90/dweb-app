"use strict";
/**
 * Enhanced RAG Backend Client
 * Provides integration with the Enhanced RAG Backend API
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAGBackendClient = void 0;
const vscode = __importStar(require("vscode"));
const node_fetch_1 = __importDefault(require("node-fetch"));
class RAGBackendClient {
    constructor(config) {
        this.config = {
            baseUrl: config?.baseUrl || 'http://localhost:8000',
            timeout: config?.timeout || 30000,
            retries: config?.retries || 3
        };
        this.outputChannel = vscode.window.createOutputChannel('Enhanced RAG Backend');
    }
    /**
     * Check if the RAG backend is healthy
     */
    async healthCheck() {
        try {
            const response = await this.makeRequest('/health', {
                method: 'GET',
                timeout: 5000
            });
            return response.status === 'healthy';
        }
        catch (error) {
            this.outputChannel.appendLine(`Health check failed: ${error}`);
            return false;
        }
    }
    /**
     * Get detailed system metrics
     */
    async getSystemMetrics() {
        try {
            return await this.makeRequest('/health/detailed');
        }
        catch (error) {
            this.outputChannel.appendLine(`Failed to get system metrics: ${error}`);
            throw error;
        }
    }
    /**
     * Search documents using semantic/vector search
     */
    async searchDocuments(query) {
        try {
            this.outputChannel.appendLine(`Searching: "${query.query}" (${query.searchType || 'hybrid'})`);
            const response = await this.makeRequest('/api/v1/rag/search', {
                method: 'POST',
                body: JSON.stringify(query)
            });
            this.outputChannel.appendLine(`Found ${response.results?.length || 0} results`);
            return response;
        }
        catch (error) {
            this.outputChannel.appendLine(`Search failed: ${error}`);
            throw error;
        }
    }
    /**
     * Upload and process document
     */
    async uploadDocument(upload) {
        try {
            this.outputChannel.appendLine(`Uploading document: ${upload.fileName}`);
            const formData = new FormData();
            const blob = new Blob([upload.file]);
            formData.append('document', blob, upload.fileName);
            if (upload.title)
                formData.append('title', upload.title);
            if (upload.documentType)
                formData.append('documentType', upload.documentType);
            if (upload.caseId)
                formData.append('caseId', upload.caseId);
            if (upload.metadata)
                formData.append('metadata', JSON.stringify(upload.metadata));
            const response = await this.makeRequest('/api/v1/rag/upload', {
                method: 'POST',
                body: formData,
                headers: {} // Let fetch set Content-Type for FormData
            });
            this.outputChannel.appendLine(`Document processed: ${response.document?.id}`);
            return response;
        }
        catch (error) {
            this.outputChannel.appendLine(`Upload failed: ${error}`);
            throw error;
        }
    }
    /**
     * Execute multi-agent workflow
     */
    async executeWorkflow(request) {
        try {
            this.outputChannel.appendLine(`Executing workflow: ${request.workflowType}`);
            const response = await this.makeRequest('/api/v1/agents/workflow', {
                method: 'POST',
                body: JSON.stringify(request),
                timeout: request.options?.timeout || 120000 // 2 minutes default for workflows
            });
            this.outputChannel.appendLine(`Workflow completed in ${response.result?.metadata?.processingTime}ms`);
            return response;
        }
        catch (error) {
            this.outputChannel.appendLine(`Workflow failed: ${error}`);
            throw error;
        }
    }
    /**
     * Chat with AI agent
     */
    async chatWithAgent(messages, options) {
        try {
            this.outputChannel.appendLine(`Starting AI chat with ${messages.length} messages`);
            const response = await this.makeRequest('/api/v1/agents/chat', {
                method: 'POST',
                body: JSON.stringify({
                    messages,
                    options: options || {}
                })
            });
            return response;
        }
        catch (error) {
            this.outputChannel.appendLine(`Chat failed: ${error}`);
            throw error;
        }
    }
    /**
     * Analyze text with AI
     */
    async analyzeText(text, analysisType, options) {
        try {
            this.outputChannel.appendLine(`Analyzing text (${analysisType || 'general'}): ${text.substring(0, 100)}...`);
            const response = await this.makeRequest('/api/v1/rag/analyze', {
                method: 'POST',
                body: JSON.stringify({
                    text,
                    analysisType: analysisType || 'general',
                    options: options || {}
                })
            });
            return response;
        }
        catch (error) {
            this.outputChannel.appendLine(`Analysis failed: ${error}`);
            throw error;
        }
    }
    /**
     * Summarize text
     */
    async summarizeText(text, length, options) {
        try {
            this.outputChannel.appendLine(`Summarizing text (${length || 'medium'}): ${text.substring(0, 100)}...`);
            const response = await this.makeRequest('/api/v1/rag/summarize', {
                method: 'POST',
                body: JSON.stringify({
                    text,
                    length: length || 'medium',
                    options: options || {}
                })
            });
            return response;
        }
        catch (error) {
            this.outputChannel.appendLine(`Summarization failed: ${error}`);
            throw error;
        }
    }
    /**
     * Get RAG system statistics
     */
    async getRAGStats() {
        try {
            return await this.makeRequest('/api/v1/rag/stats');
        }
        catch (error) {
            this.outputChannel.appendLine(`Failed to get RAG stats: ${error}`);
            throw error;
        }
    }
    /**
     * Get document by ID
     */
    async getDocument(id, includeContent, includeChunks) {
        try {
            const params = new URLSearchParams();
            if (includeContent)
                params.set('includeContent', 'true');
            if (includeChunks)
                params.set('includeChunks', 'true');
            const url = `/api/v1/documents/${id}${params.toString() ? '?' + params.toString() : ''}`;
            return await this.makeRequest(url);
        }
        catch (error) {
            this.outputChannel.appendLine(`Failed to get document ${id}: ${error}`);
            throw error;
        }
    }
    /**
     * List documents with filters
     */
    async listDocuments(filters) {
        try {
            const params = new URLSearchParams();
            if (filters?.caseId)
                params.set('caseId', filters.caseId);
            if (filters?.documentType)
                params.set('documentType', filters.documentType);
            if (filters?.limit)
                params.set('limit', filters.limit.toString());
            if (filters?.offset)
                params.set('offset', filters.offset.toString());
            if (filters?.search)
                params.set('search', filters.search);
            const url = `/api/v1/documents${params.toString() ? '?' + params.toString() : ''}`;
            return await this.makeRequest(url);
        }
        catch (error) {
            this.outputChannel.appendLine(`Failed to list documents: ${error}`);
            throw error;
        }
    }
    /**
     * Find similar documents
     */
    async findSimilarDocuments(documentId, limit, threshold) {
        try {
            const params = new URLSearchParams();
            if (limit)
                params.set('limit', limit.toString());
            if (threshold)
                params.set('threshold', threshold.toString());
            const url = `/api/v1/rag/similar/${documentId}${params.toString() ? '?' + params.toString() : ''}`;
            return await this.makeRequest(url);
        }
        catch (error) {
            this.outputChannel.appendLine(`Failed to find similar documents: ${error}`);
            throw error;
        }
    }
    /**
     * Clear cache
     */
    async clearCache(pattern) {
        try {
            const params = new URLSearchParams();
            if (pattern)
                params.set('pattern', pattern);
            const url = `/api/v1/rag/cache${params.toString() ? '?' + params.toString() : ''}`;
            return await this.makeRequest(url, { method: 'DELETE' });
        }
        catch (error) {
            this.outputChannel.appendLine(`Failed to clear cache: ${error}`);
            throw error;
        }
    }
    /**
     * Make HTTP request with retry logic
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.config.baseUrl}${endpoint}`;
        const requestOptions = {
            method: options.method || 'GET',
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'VS-Code-Extension/1.0.0',
                ...options.headers
            },
            body: options.body,
            timeout: options.timeout || this.config.timeout
        };
        // Remove Content-Type for FormData
        if (options.body instanceof FormData) {
            delete requestOptions.headers['Content-Type'];
        }
        let lastError;
        for (let attempt = 1; attempt <= this.config.retries; attempt++) {
            try {
                this.outputChannel.appendLine(`Request attempt ${attempt}/${this.config.retries}: ${requestOptions.method} ${url}`);
                const response = await (0, node_fetch_1.default)(url, requestOptions);
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await response.json();
                }
                else {
                    return await response.text();
                }
            }
            catch (error) {
                lastError = error;
                this.outputChannel.appendLine(`Attempt ${attempt} failed: ${error}`);
                if (attempt < this.config.retries) {
                    const delay = Math.pow(2, attempt - 1) * 1000; // Exponential backoff
                    this.outputChannel.appendLine(`Retrying in ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        throw lastError;
    }
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.outputChannel.appendLine(`Updated RAG backend config: ${JSON.stringify(this.config)}`);
    }
    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }
    /**
     * Dispose resources
     */
    dispose() {
        this.outputChannel.dispose();
    }
}
exports.RAGBackendClient = RAGBackendClient;
//# sourceMappingURL=ragBackendClient.js.map