import { type RequestHandler,  error } from '@sveltejs/kit';

const OLLAMA_URL = import.meta.env.OLLAMA_URL || 'http://localhost:11434';
const OLLAMA_MODEL = import.meta.env.OLLAMA_MODEL || 'gemma3-legal';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatRequest {
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { messages, stream = true, temperature = 0.7 }: ChatRequest = await request.json();

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      throw error(400, 'Messages array is required');
    }

    // Construct the conversation for Ollama
    const conversationText = messages
      .map(msg => `${msg.role === 'user' ? 'Human' : msg.role === 'assistant' ? 'Assistant' : 'System'}: ${msg.content}`)
      .join('\n\n');

    const finalPrompt = conversationText + '\n\nAssistant:';

    // Check if Ollama is available
    try {
      const healthCheck = await fetch(`${OLLAMA_URL}/api/version`, {
        signal: AbortSignal.timeout(3000)
      });
      
      if (!healthCheck.ok) {
        throw new Error('Ollama service unavailable');
      }
    } catch (healthError) {
      console.error('Ollama health check failed:', healthError);
      throw error(503, 'AI service is currently unavailable. Please ensure Ollama is running.');
    }

    // Make request to Ollama
    const ollamaResponse = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: finalPrompt,
        stream: true,
        options: {
          temperature,
          num_predict: 2048,
          top_p: 0.9,
          top_k: 40
        }
      })
    });

    if (!ollamaResponse.ok) {
      const errorText = await ollamaResponse.text();
      console.error('Ollama API error:', errorText);
      throw error(ollamaResponse.status, `Ollama API error: ${errorText}`);
    }

    // Return streaming response
    if (stream && ollamaResponse.body) {
      const encoder = new TextEncoder();
      
      const readable = new ReadableStream({
        start(controller) {
          const reader = ollamaResponse.body!.getReader();
          
          async function pump() {
            try {
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                  controller.close();
                  break;
                }

                const chunk = new TextDecoder().decode(value);
                const lines = chunk.split('\n').filter(line => line.trim());

                for (const line of lines) {
                  try {
                    const data = JSON.parse(line);
                    
                    if (data.response) {
                      // Format response for the frontend
                      const formattedResponse = `data: ${JSON.stringify({
                        message: {
                          content: data.response
                        },
                        done: data.done || false
                      })}\n\n`;
                      
                      controller.enqueue(encoder.encode(formattedResponse));
                    }
                    
                    if (data.done) {
                      controller.close();
                      break;
                    }
                  } catch (parseError) {
                    // Skip invalid JSON lines
                    console.warn('Failed to parse Ollama response line:', line);
                  }
                }
              }
            } catch (streamError) {
              console.error('Streaming error:', streamError);
              controller.error(streamError);
            }
          }
          
          pump();
        }
      });

      return new Response(readable, {
        headers: {
          'Content-Type': 'text/plain',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*'
        }
      });
    }

    // Fallback non-streaming response
    const reader = ollamaResponse.body!.getReader();
    let fullResponse = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = new TextDecoder().decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            fullResponse += data.response;
          }
        } catch (parseError) {
          // Skip invalid JSON lines
        }
      }
    }

    return new Response(JSON.stringify({
      message: {
        content: fullResponse
      },
      done: true
    }), {
      headers: {
        'Content-Type': 'application/json'
      }
    });

  } catch (err) {
    console.error('Chat API error:', err);
    
    if (err instanceof Error && 'status' in err) {
      throw err;
    }
    
    throw error(500, `Chat service error: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

// Health check endpoint
export const GET: RequestHandler = async () => {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/version`, {
      signal: AbortSignal.timeout(5000)
    });
    
    if (response.ok) {
      const data = await response.json();
      return new Response(JSON.stringify({
        status: 'healthy',
        service: 'ollama',
        version: data.version,
        url: OLLAMA_URL,
        model: OLLAMA_MODEL,
        timestamp: new Date().toISOString()
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (err) {
    return new Response(JSON.stringify({
      status: 'error',
      error: err instanceof Error ? err.message : 'Unknown error',
      url: OLLAMA_URL,
      timestamp: new Date().toISOString()
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};