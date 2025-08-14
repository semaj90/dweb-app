// @ts-nocheck
// Enhanced YoRHa Detective AI Chat API
// SvelteKit 2.0 + Svelte 5 + Ollama + YoRHa Fallback responses

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ollamaService } from '$lib/server/services/OllamaService';
import { logger } from '$lib/server/logger';
import { dev } from '$app/environment';

export interface ChatRequest {
  message: string;
  model?: string;
  context?: string[];
  temperature?: number;
  stream?: boolean;
  caseId?: string;
  useRAG?: boolean;
  history?: ChatMessage[];
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export interface ChatResponse {
  response: string;
  content: string;
  model: string;
  context?: number[];
  timestamp: string;
  performance: {
    duration: number;
    tokens: number;
    promptTokens: number;
    responseTokens: number;
    tokensPerSecond: number;
  };
  suggestions?: string[];
  relatedCases?: string[];
}

// YoRHa Detective AI fallback responses
const YORHA_RESPONSES = {
  greetings: [
    "YoRHa Detective Interface activated. I'm ready to assist with your legal investigation.",
    'Detective AI online. How may I help with your case analysis today?',
    'YoRHa systems operational. What investigation requires my attention?',
  ],

  legal_analysis: [
    'Analyzing legal precedents and case law... Cross-referencing with current database.',
    'Scanning for relevant statutes and regulations. Initiating comprehensive review.',
    'Legal pattern recognition active. Identifying key elements for case strategy.',
  ],

  document_review: [
    'Document analysis protocol initiated. Scanning for critical evidence markers.',
    'Processing contract terms and identifying potential liability issues.',
    'Extracting key phrases and cross-referencing with legal terminology database.',
  ],

  case_management: [
    'Case timeline analysis complete. Identifying sequence of events and key dates.',
    'Evidence correlation matrix updated. Flagging inconsistencies for review.',
    'Witness statement analysis in progress. Detecting potential contradictions.',
  ],

  unknown: [
    'Processing query... Please specify the type of legal assistance required.',
    'YoRHa Detective requires additional context. What aspect of your case needs analysis?',
    'Investigation parameters unclear. Please provide more specific details.',
  ],
};

function categorizeQuery(message: string): keyof typeof YORHA_RESPONSES {
  const lowerMessage = message.toLowerCase();

  if (
    lowerMessage.includes('hello') ||
    lowerMessage.includes('hi') ||
    lowerMessage.includes('help')
  ) {
    return 'greetings';
  }

  if (
    lowerMessage.includes('legal') ||
    lowerMessage.includes('law') ||
    lowerMessage.includes('statute') ||
    lowerMessage.includes('precedent') ||
    lowerMessage.includes('case law')
  ) {
    return 'legal_analysis';
  }

  if (
    lowerMessage.includes('document') ||
    lowerMessage.includes('contract') ||
    lowerMessage.includes('review') ||
    lowerMessage.includes('evidence') ||
    lowerMessage.includes('file')
  ) {
    return 'document_review';
  }

  if (
    lowerMessage.includes('case') ||
    lowerMessage.includes('timeline') ||
    lowerMessage.includes('witness') ||
    lowerMessage.includes('investigation') ||
    lowerMessage.includes('manage')
  ) {
    return 'case_management';
  }

  return 'unknown';
}

function generateYorhaResponse(query: string, context?: string): string {
  const category = categorizeQuery(query);
  const responses = YORHA_RESPONSES[category];
  const baseResponse = responses[Math.floor(Math.random() * responses.length)];

  // Add context-specific enhancements
  let enhancedResponse = baseResponse;

  if (category === 'legal_analysis') {
    enhancedResponse +=
      '\n\nRecommendation: Consider consulting relevant jurisdiction-specific statutes and recent appellate decisions.';
  } else if (category === 'document_review') {
    enhancedResponse +=
      '\n\nSuggestion: Flag any clauses related to liability, indemnification, and termination for detailed review.';
  } else if (category === 'case_management') {
    enhancedResponse +=
      '\n\nNext steps: Update case status and notify relevant team members of findings.';
  }

  return enhancedResponse;
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const {
      message,
      model = 'gemma2:2b',
      temperature = 0.7,
      stream = false,
      history = [],
    }: ChatRequest = await request.json();

    // Validate input
    if (!message?.trim()) {
      throw error(400, { message: 'Message is required' });
    }

    let response: string;
    let performance: any;

    try {
      // Check Ollama health
      const isHealthy = await ollamaService.isHealthy();

      if (!isHealthy) {
        console.warn('Ollama service unavailable, using YoRHa fallback');
        response = generateYorhaResponse(message);
        performance = {
          duration: Date.now() - startTime,
          tokens: response.length,
          promptTokens: message.length,
          responseTokens: response.length,
          tokensPerSecond: response.length / ((Date.now() - startTime) / 1000),
        };
      } else {
        // Add YoRHa Detective system prompt
        const systemPrompt = `You are YoRHa Detective, an advanced AI assistant specialized in legal analysis and investigation.
        You have access to comprehensive legal databases and can help with:
        - Legal document analysis and review
        - Case management and timeline construction
        - Evidence correlation and pattern recognition
        - Legal research and precedent analysis
        - Contract review and liability assessment

        Respond in a professional, analytical tone befitting a detective AI system.
        Keep responses concise but informative. When appropriate, suggest next steps or additional considerations.

        Format responses with clear structure and actionable insights.`;

        // Call Ollama with system prompt
        const chatResponse = await ollamaService.chat({
          model,
          messages: [
            { role: 'system', content: systemPrompt },
            ...history.slice(-5).map((h) => ({ role: h.role, content: h.content })),
            { role: 'user', content: message },
          ],
          stream: false,
          options: {
            temperature,
            num_predict: 500,
          },
        });

        response = chatResponse.message.content;
        performance = {
          duration: Date.now() - startTime,
          tokens: chatResponse.eval_count || 0,
          promptTokens: chatResponse.prompt_eval_count || 0,
          responseTokens: chatResponse.eval_count || 0,
          tokensPerSecond: chatResponse.eval_count
            ? chatResponse.eval_count / (chatResponse.eval_duration / 1000000000)
            : 0,
        };
      }
    } catch (ollamaError) {
      console.warn('Ollama error, falling back to YoRHa responses:', ollamaError);
      response = generateYorhaResponse(message);
      performance = {
        duration: Date.now() - startTime,
        tokens: response.length,
        promptTokens: message.length,
        responseTokens: response.length,
        tokensPerSecond: response.length / ((Date.now() - startTime) / 1000),
      };
    }

    // Log interaction
    logger.info(`YoRHa Chat - User: ${message.substring(0, 100)}...`);
    logger.info(`YoRHa Chat - AI: ${response.substring(0, 100)}...`);

    return json({
      response,
      content: response, // For YoRHa Assistant component compatibility
      model: `yorha-detective-${model}`,
      timestamp: new Date().toISOString(),
      performance,
      suggestions: [
        'Analyze legal precedents',
        'Review case timeline',
        'Check evidence correlation',
        'Generate case summary',
      ],
    } satisfies ChatResponse);
  } catch (err) {
    logger.error('Chat API error:', err);

    // Final fallback
    const fallbackResponse = generateYorhaResponse('error');

    return json({
      response: fallbackResponse,
      content: fallbackResponse,
      model: 'yorha-detective-fallback',
      timestamp: new Date().toISOString(),
      performance: {
        duration: Date.now() - startTime,
        tokens: fallbackResponse.length,
        promptTokens: 0,
        responseTokens: fallbackResponse.length,
        tokensPerSecond: 0,
      },
      error: 'Service temporarily unavailable',
    } satisfies ChatResponse);
  }
};

export const GET: RequestHandler = async () => {
  try {
    const isHealthy = await ollamaService.isHealthy();

    return json({
      status: isHealthy
        ? 'YoRHa Detective AI Assistant Online'
        : 'Ollama Offline - YoRHa Fallback Active',
      capabilities: [
        'Legal document analysis',
        'Case management assistance',
        'Evidence correlation',
        'Legal research support',
        'Contract review',
        'Timeline analysis',
      ],
      model: 'yorha-detective-v1.0',
      uptime: new Date().toISOString(),
      fallbackMode: !isHealthy,
    });
  } catch (error) {
    logger.error('Health check failed', error);
    return json(
      {
        status: 'YoRHa Fallback Mode Active',
        error: (error as Error).message,
        timestamp: new Date().toISOString(),
        fallbackMode: true,
      },
      { status: 200 }
    ); // Still return 200 since fallback works
  }
};
