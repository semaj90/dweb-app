import type { RequestHandler } from './$types.js';

/**
 * Ollama Chat API Endpoint
 * Handles streaming and non-streaming chat requests with legal context
 */

import { ollamaChatStream } from "$lib/services/ollamaChatStream";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();

    const {
      message,
      model: reqModel,
      temperature: reqTemperature,
      maxTokens: reqMaxTokens,
      stream: reqStream,
      systemPrompt,
      conversationId,
      useVectorSearch: reqUseVectorSearch,
      context = [],
    } = body;

    // Environment-backed defaults (check your .env for these keys)
    // OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_MAX_TOKENS, OLLAMA_STREAM, OLLAMA_USE_VECTOR_SEARCH
    const model = import.meta.env.OLLAMA_MODEL ?? reqModel ?? 'gemma3:legal:latest';
    const temperature =
      reqTemperature !== undefined
      ? Number(reqTemperature)
      : Number(import.meta.env.OLLAMA_TEMPERATURE ?? 0.7);
    const maxTokens =
      reqMaxTokens !== undefined
      ? Number(reqMaxTokens)
      : Number(import.meta.env.OLLAMA_MAX_TOKENS ?? 2048);
    const stream =
      reqStream !== undefined
      ? Boolean(reqStream)
      : (import.meta.env.OLLAMA_STREAM ?? 'false') === 'true';
    const useVectorSearch =
      reqUseVectorSearch !== undefined
      ? Boolean(reqUseVectorSearch)
      : (import.meta.env.OLLAMA_USE_VECTOR_SEARCH ?? 'true') === 'true';

    if (!message) {
      return json({ error: 'Message is required' }, { status: 400 });
    }

    // For streaming responses
    if (stream) {
      const encoder = new TextEncoder();
      const readable = new ReadableStream({
        async start(controller) {
          try {
            const streamGenerator = ollamaChatStream({
              message,
              model,
              temperature,
              maxTokens,
              systemPrompt,
              conversationId,
              useVectorSearch,
              context,
            });

            for await (const chunk of streamGenerator) {
              const data = `data: ${JSON.stringify(chunk)}\n\n`;
              controller.enqueue(encoder.encode(data));
            }

            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
          } catch (error: any) {
            console.error('Streaming error:', error);
            const errorChunk = {
              text: 'An error occurred while processing your request.',
              metadata: { type: 'error', error: error.message },
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
            controller.close();
          }
        },
      });

      return new Response(readable, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
        },
      });
    }

    // For non-streaming responses
    let fullResponse = '';
    let sources = [];
    let confidence = 0;

    const streamGenerator = ollamaChatStream({
      message,
      model,
      temperature,
      maxTokens,
      systemPrompt,
      conversationId,
      useVectorSearch,
      context,
    });

    for await (const chunk of streamGenerator) {
      if (chunk.metadata?.type === 'sources') {
        sources = chunk.metadata.sources || [];
      } else if (chunk.metadata?.type === 'text') {
        fullResponse += chunk.text;
        confidence = chunk.metadata.confidence || 0;
      }
    }

    return json({
      success: true,
      response: fullResponse,
      metadata: {
        model,
        confidence,
        sources,
        conversationId,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: any) {
    console.error('Chat API error:', error);
    return json(
      {
        success: false,
        error: 'Failed to process chat request',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
};