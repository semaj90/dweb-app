import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const OLLAMA_BASE_URL = 'http://localhost:11434';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { message, conversationId, settings, contextInjection } = await request.json();

    if (!message || message.trim() === '') {
      return json({ error: 'Message is required' }, { status: 400 });
    }

    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: settings?.model || 'gemma3-legal',
        prompt: message,
        stream: false
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return json({ error: 'AI service unavailable', details: errorText }, { status: 503 });
    }

    const data = await response.json();

    return json({
      response: data.response,
      model: 'gemma3-legal',
      conversationId,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
};

export const GET: RequestHandler = async () => {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/version`);
    const isHealthy = response.ok;

    return json({
      status: isHealthy ? 'healthy' : 'unhealthy',
      service: 'ollama',
      model: 'gemma3-legal',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return json({
      status: 'error',
      error: error.message,
      timestamp: new Date().toISOString(),
    }, { status: 503 });
  }
};
