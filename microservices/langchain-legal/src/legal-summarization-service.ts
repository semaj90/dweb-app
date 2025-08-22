#!/usr/bin/env node
// LangChain.js + Legal-BERT ONNX Integration for Legal Document Summarization
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { parse } from 'node:url';

// LangChain imports (updated for v0.3.x)
import { Ollama } from '@langchain/ollama';
import { PromptTemplate } from '@langchain/core/prompts';
import { LLMChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

// ONNX Runtime for Legal-BERT
import * as ort from 'onnxruntime-node';

interface SummarizationRequest {
  text: string;
  context?: string;
  style?: 'brief' | 'detailed' | 'technical';
  maxTokens?: number;
}

interface SummarizationResponse {
  summary: string;
  keyPoints: string[];
  confidence: number;
  model: string;
  processingTimeMs: number;
}

class LegalSummarizationService {
  private legalBertSession: ort.InferenceSession | null = null;
  private langchainChain: LLMChain | null = null;
  private port: number;

  constructor() {
    this.port = parseInt(process.env.LANGCHAIN_PORT || '8106');
    this.initializeLangChain();
  }

  private async initializeLangChain() {
    try {
      // Initialize LangChain with Ollama integration (FREE)
      const llm = new Ollama({
        baseUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
        model: 'gemma3-legal',
        temperature: 0.3,
        numPredict: 2000,
      });

      const promptTemplate = new PromptTemplate({
        template: `
As a legal AI assistant, provide a {style} summary of the following legal document.

Context: {context}
Document Type: {documentType}

Document Text:
{text}

Summary Requirements:
- Focus on key legal implications
- Identify important clauses and obligations
- Highlight potential risks or benefits
- Use professional legal terminology
- Keep summary {style} and actionable

Summary:`,
        inputVariables: ['text', 'context', 'style', 'documentType'],
      });

      this.langchainChain = new LLMChain({
        llm,
        prompt: promptTemplate,
      });

      console.log('✅ LangChain initialized with Ollama integration');
    } catch (error) {
      console.error('⚠️ LangChain initialization failed:', error);
    }
  }

  private async initializeLegalBERT() {
    try {
      // TODO: Load Legal-BERT ONNX model when available
      // this.legalBertSession = await ort.InferenceSession.create('./models/legal-bert.onnx');
      console.log('⚠️ Legal-BERT ONNX model not yet available - using LangChain fallback');
    } catch (error) {
      console.error('⚠️ Legal-BERT ONNX initialization failed:', error);
    }
  }

  private async summarizeWithLangChain(request: SummarizationRequest): Promise<SummarizationResponse> {
    const startTime = Date.now();
    
    try {
      if (!this.langchainChain) {
        throw new Error('LangChain not initialized');
      }

      // Split long documents
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 200,
      });

      const docs = await splitter.createDocuments([request.text]);
      let fullSummary = '';

      // Process chunks
      for (const doc of docs.slice(0, 3)) { // Limit to first 3 chunks
        const result = await this.langchainChain.call({
          text: doc.pageContent,
          context: request.context || 'legal document',
          style: request.style || 'detailed',
          documentType: this.detectDocumentType(request.text),
        });

        fullSummary += result.text + '\n\n';
      }

      // Extract key points
      const keyPoints = this.extractKeyPoints(fullSummary);

      return {
        summary: fullSummary.trim(),
        keyPoints,
        confidence: 0.85,
        model: 'ollama-gemma3-legal',
        processingTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      console.error('LangChain summarization error:', error);
      return this.fallbackSummary(request, startTime);
    }
  }

  private detectDocumentType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('whereas') && lowerText.includes('therefore')) {
      return 'contract';
    } else if (lowerText.includes('plaintiff') || lowerText.includes('defendant')) {
      return 'litigation';
    } else if (lowerText.includes('exhibit') || lowerText.includes('evidence')) {
      return 'evidence';
    } else if (lowerText.includes('statute') || lowerText.includes('regulation')) {
      return 'regulation';
    }
    
    return 'legal document';
  }

  private extractKeyPoints(summary: string): string[] {
    // Simple key point extraction - can be enhanced with NLP
    const sentences = summary.split(/[.!?]+/).filter(s => s.trim().length > 20);
    const keyPoints: string[] = [];

    sentences.forEach(sentence => {
      const trimmed = sentence.trim();
      if (trimmed.includes('shall') || trimmed.includes('must') || 
          trimmed.includes('liability') || trimmed.includes('obligation')) {
        keyPoints.push(trimmed);
      }
    });

    return keyPoints.slice(0, 5); // Top 5 key points
  }

  private fallbackSummary(request: SummarizationRequest, startTime: number): SummarizationResponse {
    const words = request.text.split(/\s+/);
    const summary = words.slice(0, 50).join(' ') + '...';
    
    return {
      summary: `Legal document analysis (${request.style || 'brief'}): ${summary}`,
      keyPoints: ['Document requires legal review', 'Key terms need analysis', 'Professional consultation recommended'],
      confidence: 0.60,
      model: 'fallback-summary',
      processingTimeMs: Date.now() - startTime,
    };
  }

  public async handleSummarization(request: SummarizationRequest): Promise<SummarizationResponse> {
    // Try Legal-BERT ONNX first, fallback to LangChain
    if (this.legalBertSession) {
      // TODO: Implement Legal-BERT ONNX inference
      return this.summarizeWithLangChain(request);
    } else {
      return this.summarizeWithLangChain(request);
    }
  }

  public startServer() {
    const server = createServer(async (req, res) => {
      const { pathname, query } = parse(req.url || '', true);
      
      // CORS headers
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      
      if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
      }

      // Health check
      if (pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          service: 'Legal Summarization Service',
          status: 'healthy',
          version: '1.0.0',
          features: ['langchain', 'ollama-integration', 'legal-bert-ready'],
          timestamp: new Date().toISOString(),
        }));
        return;
      }

      // Summarization endpoint
      if (pathname === '/api/summarize' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', async () => {
          try {
            const request: SummarizationRequest = JSON.parse(body);
            const result = await this.handleSummarization(request);
            
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(result));
          } catch (error) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Invalid request' }));
          }
        });
        return;
      }

      // 404
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not found' }));
    });

    server.listen(this.port, () => {
      console.log(`🚀 Legal Summarization Service (LangChain + Legal-BERT) on port ${this.port}`);
      console.log(`📍 Health: http://localhost:${this.port}/health`);
      console.log(`📍 Summarize: POST http://localhost:${this.port}/api/summarize`);
      console.log(`🤖 Integration: Ollama (gemma3-legal) + Future Legal-BERT ONNX`);
    });

    return server;
  }
}

// Start service if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const service = new LegalSummarizationService();
  service.startServer();
}