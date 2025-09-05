/**
 * Claude AI Agent for Legal Document Analysis
 * Enhanced RAG System Integration
 */

import { Anthropic } from '@anthropic-ai/sdk';
import { EventEmitter } from 'events';

export class ClaudeAgent extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            apiKey: config.apiKey || process.env.ANTHROPIC_API_KEY,
            model: config.model || 'claude-3-sonnet-20240229',
            maxTokens: config.maxTokens || 4096,
            temperature: config.temperature || 0.1,
            ...config
        };
        
        if (!this.config.apiKey) {
            throw new Error('Claude API key is required');
        }
        
        this.client = new Anthropic({
            apiKey: this.config.apiKey
        });
        
        this.isInitialized = false;
        this.capabilities = [
            'legal-analysis',
            'document-summarization',
            'precedent-search',
            'contract-review',
            'compliance-checking'
        ];
    }
    
    async initialize() {
        try {
            // Test connection
            await this.client.messages.create({
                model: this.config.model,
                max_tokens: 10,
                messages: [{ role: 'user', content: 'Test' }]
            });
            
            this.isInitialized = true;
            this.emit('initialized', { agent: 'claude', status: 'ready' });
            return true;
        } catch (error) {
            this.emit('error', { agent: 'claude', error: error.message });
            throw error;
        }
    }
    
    async analyzeLegalDocument(document, context = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        const prompt = `
        You are a legal AI assistant specializing in document analysis. 
        Analyze the following legal document and provide:
        
        1. Document type and classification
        2. Key legal concepts and clauses
        3. Potential issues or concerns
        4. Relevant precedents or statutes
        5. Summary of main points
        
        Context: ${JSON.stringify(context)}
        
        Document:
        ${document}
        
        Please provide a structured analysis in JSON format.
        `;
        
        try {
            const response = await this.client.messages.create({
                model: this.config.model,
                max_tokens: this.config.maxTokens,
                temperature: this.config.temperature,
                messages: [{ role: 'user', content: prompt }]
            });
            
            const analysis = {
                agent: 'claude',
                timestamp: new Date().toISOString(),
                content: response.content[0].text,
                usage: response.usage,
                confidence: this.calculateConfidence(response)
            };
            
            this.emit('analysis-complete', analysis);
            return analysis;
        } catch (error) {
            this.emit('error', { agent: 'claude', error: error.message, document: document.substring(0, 100) });
            throw error;
        }
    }
    
    async generateLegalBrief(caseData, requirements = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        const prompt = `
        Generate a comprehensive legal brief based on the following case data:
        
        Requirements: ${JSON.stringify(requirements)}
        Case Data: ${JSON.stringify(caseData)}
        
        Include:
        1. Executive Summary
        2. Statement of Facts
        3. Legal Issues
        4. Analysis and Arguments
        5. Conclusion and Recommendations
        
        Format as a professional legal document.
        `;
        
        try {
            const response = await this.client.messages.create({
                model: this.config.model,
                max_tokens: this.config.maxTokens,
                temperature: this.config.temperature,
                messages: [{ role: 'user', content: prompt }]
            });
            
            return {
                agent: 'claude',
                type: 'legal-brief',
                content: response.content[0].text,
                timestamp: new Date().toISOString(),
                usage: response.usage
            };
        } catch (error) {
            this.emit('error', { agent: 'claude', error: error.message });
            throw error;
        }
    }
    
    calculateConfidence(response) {
        // Simple confidence calculation based on response characteristics
        const contentLength = response.content[0].text.length;
        const hasStructure = response.content[0].text.includes('1.') || response.content[0].text.includes('##');
        
        let confidence = 0.5; // Base confidence
        
        if (contentLength > 1000) confidence += 0.2;
        if (hasStructure) confidence += 0.2;
        if (response.usage.input_tokens > 500) confidence += 0.1;
        
        return Math.min(confidence, 1.0);
    }
    
    getCapabilities() {
        return this.capabilities;
    }
    
    getStatus() {
        return {
            agent: 'claude',
            initialized: this.isInitialized,
            model: this.config.model,
            capabilities: this.capabilities
        };
    }
}

export default ClaudeAgent;
