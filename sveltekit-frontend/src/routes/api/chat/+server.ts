// src/routes/api/chat/+server.ts
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { OllamaChatStream } from '$lib/services/ollamaChatStream.js';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { 
      message, 
      conversationId, 
      userId, 
      model = 'gemma3-legal-enhanced',
      stream = true,
      analysisType = null,
      caseData = null,
      evidenceData = null
    } = await request.json();
    
    if (!message || !userId) {
      throw error(400, 'Message and userId are required');
    }
    
    const chatStream = new OllamaChatStream(model);
    
    // Check if we need to create a new conversation
    let finalConversationId = conversationId;
    if (!finalConversationId) {
      const conversation = await chatStream.createConversation(
        userId, 
        'AI Chat Session', 
        'general'
      );
      finalConversationId = conversation.id;
    }
    
    if (!stream) {
      // Non-streaming response
      let fullResponse = '';
      const responseStream = chatStream.streamChat(finalConversationId, message);
      
      for await (const chunk of responseStream) {
        fullResponse += chunk;
      }
      
      return new Response(JSON.stringify({
        success: true,
        response: fullResponse,
        conversationId: finalConversationId,
        model,
        timestamp: new Date().toISOString()
      }), {
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
    
    // Streaming response
    const encoder = new TextEncoder();
    
    const readableStream = new ReadableStream({
      async start(controller) {
        try {
          let responseStream;
          
          // Choose the appropriate analysis method
          if (analysisType && caseData) {
            responseStream = chatStream.analyzeCaseStream(
              finalConversationId, 
              caseData, 
              analysisType
            );
          } else if (analysisType && evidenceData) {
            responseStream = chatStream.analyzeEvidenceStream(
              finalConversationId, 
              evidenceData, 
              analysisType
            );
          } else {
            responseStream = chatStream.streamChat(finalConversationId, message);
          }
          
          // Send initial metadata
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            type: 'start',
            conversationId: finalConversationId,
            model,
            timestamp: new Date().toISOString()
          })}\n\n`));
          
          // Stream the response
          for await (const chunk of responseStream) {
            const data = JSON.stringify({
              type: 'chunk',
              content: chunk,
              timestamp: new Date().toISOString()
            });
            controller.enqueue(encoder.encode(`data: ${data}\n\n`));
          }
          
          // Send completion signal
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            type: 'end',
            timestamp: new Date().toISOString()
          })}\n\n`));
          
          controller.close();
        } catch (err) {
          console.error('Streaming error:', err);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            type: 'error',
            error: err instanceof Error ? err.message : 'Unknown error',
            timestamp: new Date().toISOString()
          })}\n\n`));
          controller.close();
        }
      }
    });
    
    return new Response(readableStream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type'
      }
    });
    
  } catch (err) {
    console.error('Chat API error:', err);
    
    if (err instanceof Error && err.message.includes('Ollama')) {
      throw error(503, 'AI chat service unavailable');
    }
    
    throw error(500, `Chat failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

export const GET: RequestHandler = async () => {
  // Health check for chat service
  try {
    const chatStream = new OllamaChatStream();
    const isHealthy = await chatStream.healthCheck();
    const models = await chatStream.getAvailableModels();
    
    return new Response(JSON.stringify({
      status: 'Chat API endpoint',
      healthy: isHealthy,
      availableModels: models,
      methods: ['POST'],
      example: {
        message: 'Analyze this case for prosecution strengths',
        userId: 'user-123',
        stream: true,
        analysisType: 'strengths_weaknesses',
        caseData: { title: 'Sample Case', description: '...' }
      },
      timestamp: new Date().toISOString()
    }), {
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (err) {
    throw error(503, 'Chat service health check failed');
  }
};