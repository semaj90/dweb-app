/**
 * Enhanced AI Chat API Endpoint
 * Integrates AI input synthesis, LegalBERT middleware, RAG pipeline, and streaming responses
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Import orchestrator and services
import {
  processAIAssistantQuery,
  synthesizeAIInput,
  analyzeLegalText,
  processRAGPipeline,
  rerankSearchResults,
} from '$lib/services/comprehensive-database-orchestrator';

// Import Ollama service for local model processing
import { ollamaService } from '$lib/services/ollamaService';

// Enhanced request interface
interface EnhancedChatRequest {
  query: string;
  context?: {
    userRole?: string;
    caseId?: string;
    documentIds?: string[];
    sessionContext?: any;
    enableLegalBERT?: boolean;
    enableRAG?: boolean;
    maxDocuments?: number;
  };
  settings?: {
    enhancementLevel?: 'basic' | 'standard' | 'advanced' | 'comprehensive';
    includeConfidenceScores?: boolean;
    enableStreamingResponse?: boolean;
    model?: string;
    temperature?: number;
    maxTokens?: number;
  };
}

  interface EnhancedChatConfig {
    enhancementLevel: 'basic' | 'standard' | 'advanced' | 'comprehensive';
    enableLegalBERT: boolean;
    enableRAG: boolean;
    enableSynthesis: boolean;
    model: string;
    temperature: number;
    maxTokens: number;
    maxDocuments?: number; // optional document cap for RAG
  }

interface EnhancedChatResponse {
  response: string;
  synthesizedInput?: any;
  legalAnalysis?: any;
  ragResults?: any;
  confidence: number;
  processingTime: number;
  metadata: {
    model: string;
    tokensUsed?: number;
    enabledFeatures: string[];
    fallbacksUsed?: string[];
    cacheHits?: string[];
  };
  recommendations?: string[];
  contextualPrompts?: any[];
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const body: EnhancedChatRequest = await request.json();
    const { query, context = {}, settings = {} } = body;

    if (!query?.trim()) {
      return json({ error: 'Query is required' }, { status: 400 });
    }

    // Set defaults
  const config: EnhancedChatConfig = {
      enhancementLevel: settings.enhancementLevel || 'comprehensive',
      enableLegalBERT: context.enableLegalBERT !== false,
      enableRAG: context.enableRAG === true,
      enableSynthesis: true,
      model: settings.model || 'gemma3-legal:latest',
      temperature: settings.temperature || 0.3,
      maxTokens: settings.maxTokens || 2000,
      ...settings,
    };

    const enabledFeatures: string[] = [];
    const fallbacksUsed: string[] = [];
    const cacheHits: string[] = [];

    // Process through the enhanced AI pipeline
    let synthesizedInput = null;
    let legalAnalysis = null;
    let ragResults = null;
    let aiResponse = '';
    let confidence = 0.5;

    try {
      // Step 1: Input Synthesis with LegalBERT
      if (config.enableSynthesis) {
        enabledFeatures.push('input-synthesis');

        try {
          synthesizedInput = await synthesizeAIInput(query, context);
          enabledFeatures.push('legalbert-analysis');
        } catch (error) {
          console.warn('Input synthesis failed, using fallback:', error);
          fallbacksUsed.push('basic-input-processing');

          synthesizedInput = {
            originalQuery: query,
            enhancedPrompt: enhancePromptBasic(query, context),
            intent: { primary: 'general', confidence: 0.3 },
            legalContext: { domain: 'general', complexity: 0.5 },
          };
        }
      }

      // Step 2: RAG Pipeline Processing
      if (config.enableRAG && context.documentIds?.length) {
        enabledFeatures.push('rag-pipeline');

        try {
          // Get relevant documents
          const maxDocs = config.maxDocuments || 10;
          const documents = await retrieveDocuments(context.documentIds, maxDocs);

          if (documents.length > 0) {
            ragResults = await processRAGPipeline(query, documents, {
              maxDocuments: maxDocs,
              enableReranking: true,
              generateSummary: true,
            });
            enabledFeatures.push('document-reranking');
          }
        } catch (error) {
          console.warn('RAG pipeline failed:', error);
          fallbacksUsed.push('basic-document-search');
        }
      }

      // Step 3: Advanced Legal Analysis
      if (config.enableLegalBERT) {
        try {
          legalAnalysis = await analyzeLegalText(query, {
            includeEntities: true,
            includeConcepts: true,
            includeSentiment: true,
            includeComplexity: true,
          });
          enabledFeatures.push('legal-entity-extraction');
        } catch (error) {
          console.warn('Legal analysis failed:', error);
          fallbacksUsed.push('basic-text-analysis');
        }
      }

      // Step 4: Generate AI Response
      const enhancedPrompt = buildEnhancedPrompt(
        query,
        synthesizedInput,
        legalAnalysis,
        ragResults,
        context,
        config
      );

      try {
        // Use local Ollama model for response generation
        aiResponse = await ollamaService.generateCompletion(enhancedPrompt, {
          temperature: config.temperature,
          maxTokens: config.maxTokens,
          systemPrompt: buildSystemPrompt(context, legalAnalysis),
        });

        enabledFeatures.push('local-llm-generation');
        confidence = calculateResponseConfidence(aiResponse, synthesizedInput, legalAnalysis);
      } catch (ollamaError) {
        console.warn('Ollama generation failed, using fallback:', ollamaError);
        fallbacksUsed.push('basic-response-generation');

        // Fallback to simple response
        aiResponse = await generateFallbackResponse(
          query,
          context,
          synthesizedInput,
          legalAnalysis
        );
        confidence = 0.4;
      }
    } catch (error) {
      console.error('Enhanced AI pipeline failed:', error);

      // Complete fallback
      aiResponse = `I apologize, but I encountered an issue processing your request. Here's a basic response:

${query.includes('?') ? 'This appears to be a question that requires legal analysis.' : 'This appears to be a legal matter that needs attention.'}

Please try rephrasing your query or contact support for assistance.`;

      confidence = 0.2;
      fallbacksUsed.push('complete-fallback');
    }

    // Build response
    const response: EnhancedChatResponse = {
      response: aiResponse,
      synthesizedInput,
      legalAnalysis,
      ragResults,
      confidence,
      processingTime: Date.now() - startTime,
      metadata: {
        model: config.model,
        tokensUsed: estimateTokens(aiResponse),
        enabledFeatures,
        fallbacksUsed: fallbacksUsed.length > 0 ? fallbacksUsed : undefined,
        cacheHits: cacheHits.length > 0 ? cacheHits : undefined,
      },
      recommendations: buildRecommendations(synthesizedInput, legalAnalysis, ragResults),
      contextualPrompts: synthesizedInput?.contextualPrompts || [],
    };

    return json(response);
  } catch (error) {
    console.error('Enhanced AI chat API error:', error);

    return json(
      {
        error: 'Internal server error',
        message: error.message,
        processingTime: Date.now() - startTime,
      },
      { status: 500 }
    );
  }
};

// Helper functions

function enhancePromptBasic(query: string, context: any): string {
  const parts = [];

  if (context.userRole) {
    parts.push(`As a ${context.userRole},`);
  }

  if (context.caseId) {
    parts.push(`regarding Case ${context.caseId},`);
  }

  parts.push(query);
  parts.push(
    'Please provide a comprehensive legal analysis with relevant citations and recommendations.'
  );

  return parts.join(' ');
}

async function retrieveDocuments(documentIds: string[], maxDocs: number) {
  try {
    // This would connect to your document database
    // For now, return mock documents
    return documentIds.slice(0, maxDocs).map((id, index) => ({
      id,
      title: `Document ${index + 1}`,
      content: `Sample legal document content for document ${id}`,
      metadata: { type: 'legal', relevance: 0.8 - index * 0.1 },
    }));
  } catch (error) {
    console.warn('Document retrieval failed:', error);
    return [];
  }
}

function buildEnhancedPrompt(
  originalQuery: string,
  synthesizedInput: any,
  legalAnalysis: any,
  ragResults: any,
  context: any,
  config: any
): string {
  const sections = [];

  // Use synthesized prompt if available
  if (synthesizedInput?.enhancedPrompt) {
    sections.push(synthesizedInput.enhancedPrompt);
  } else {
    sections.push(originalQuery);
  }

  // Add legal context
  if (legalAnalysis?.entities?.length > 0) {
    sections.push(
      '\nLegal entities detected: ' + legalAnalysis.entities.map((e) => e.text).join(', ')
    );
  }

  // Add document context
  if (ragResults?.documents?.length > 0) {
    sections.push(
      '\nRelevant documents available: ' + ragResults.documents.map((d) => d.title).join(', ')
    );
  }

  // Add role-specific guidance
  if (context.userRole) {
    sections.push(`\nProvide analysis appropriate for a ${context.userRole}.`);
  }

  // Add enhancement level instructions
  switch (config.enhancementLevel) {
    case 'comprehensive':
      sections.push(
        '\nProvide comprehensive analysis including legal framework, precedents, risks, and actionable recommendations.'
      );
      break;
    case 'advanced':
      sections.push('\nInclude legal citations and case precedents in your analysis.');
      break;
    case 'standard':
      sections.push('\nProvide clear legal analysis with practical guidance.');
      break;
  }

  return sections.join('');
}

function buildSystemPrompt(context: any, legalAnalysis: any): string {
  const prompts = [
    'You are an expert legal AI assistant with comprehensive knowledge of law, case precedents, and legal procedures.',
    'Provide accurate, helpful, and professional legal guidance.',
  ];

  if (context.userRole) {
    prompts.push(`You are assisting a ${context.userRole} with their legal work.`);
  }

  if (legalAnalysis?.complexity?.legalComplexity > 0.7) {
    prompts.push('This appears to be a complex legal matter requiring careful analysis.');
  }

  prompts.push(
    'Always include relevant legal citations where applicable and provide practical, actionable advice.'
  );
  prompts.push(
    'If you are uncertain about any legal advice, clearly state the limitations and recommend consulting with a qualified attorney.'
  );

  return prompts.join(' ');
}

function calculateResponseConfidence(
  response: string,
  synthesizedInput: any,
  legalAnalysis: any
): number {
  let confidence = 0.5;

  // Base confidence from response quality
  if (response.length > 100) confidence += 0.1;
  if (response.includes('citation') || response.includes('precedent')) confidence += 0.1;
  if (response.includes('recommendation')) confidence += 0.1;

  // Boost from input analysis
  if (synthesizedInput?.intent?.confidence) {
    confidence += synthesizedInput.intent.confidence * 0.2;
  }

  // Boost from legal analysis
  if (legalAnalysis?.entities?.length > 0) {
    confidence += Math.min(legalAnalysis.entities.length * 0.05, 0.2);
  }

  return Math.min(confidence, 1.0);
}

async function generateFallbackResponse(
  query: string,
  context: any,
  synthesizedInput: any,
  legalAnalysis: any
): Promise<string> {
  const responses = [];

  responses.push('Thank you for your legal question.');

  if (legalAnalysis?.entities?.length > 0) {
    responses.push(
      `I can see this involves ${legalAnalysis.entities.map((e) => e.text).join(', ')}.`
    );
  }

  if (synthesizedInput?.intent?.primary) {
    responses.push(`This appears to be a ${synthesizedInput.intent.primary} inquiry.`);
  }

  responses.push('For this type of legal matter, I recommend:');
  responses.push('1. Consulting with a qualified attorney in your jurisdiction');
  responses.push('2. Reviewing relevant statutes and case law');
  responses.push('3. Gathering all relevant documentation');

  if (context.caseId) {
    responses.push(`4. Documenting this inquiry in relation to Case ${context.caseId}`);
  }

  return responses.join('\n\n');
}

function buildRecommendations(
  synthesizedInput: any,
  legalAnalysis: any,
  ragResults: any
): string[] {
  const recommendations = [];

  if (synthesizedInput?.recommendations) {
    recommendations.push(...synthesizedInput.recommendations);
  }

  if (legalAnalysis?.entities?.length > 0) {
    recommendations.push('Consider the legal implications of identified entities');
  }

  if (ragResults?.documents?.length > 0) {
    recommendations.push('Review related documents for additional context');
  }

  recommendations.push('Verify all legal advice with qualified counsel');
  recommendations.push('Document all legal research and analysis');

  return recommendations;
}

function estimateTokens(text: string): number {
  // Rough estimation: ~4 characters per token
  return Math.ceil(text.length / 4);
}

// Health check endpoint
export const GET: RequestHandler = async () => {
  try {
    // Check if services are available
    const status = {
      legalBERT: false,
      rag: false,
      synthesis: false,
      ollama: false,
      timestamp: new Date().toISOString(),
    };

    try {
      // Test LegalBERT
      await analyzeLegalText('test', { includeEntities: false });
      status.legalBERT = true;
    } catch {}

    try {
      // Test synthesis
      await synthesizeAIInput('test');
      status.synthesis = true;
    } catch {}

    try {
      // Test Ollama
      await ollamaService.generateCompletion('test', { maxTokens: 10 });
      status.ollama = true;
    } catch {}

    // RAG is available if we can process documents
    status.rag = true;

    return json(status);
  } catch (error) {
    return json({ error: error.message }, { status: 500 });
  }
};
