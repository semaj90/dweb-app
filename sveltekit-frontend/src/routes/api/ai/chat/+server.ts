import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const { message, model = 'gemma3-legal', temperature = 0.1 } = await request.json();

    if (!message) {
      return json({ error: 'Message required' }, { status: 400 });
    }

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: message,
        stream: false,
        options: {
          temperature,
          num_predict: 512,
          top_p: 0.9,
        }
      }),
      signal: AbortSignal.timeout(30000)
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const data = await response.json();

    return json({
      response: data.response,
      model,
      metadata: {
        provider: 'ollama',
        executionTime: Date.now() - startTime,
        done: data.done,
        tokens: data.eval_count || 0
      }
    });

  } catch (error) {
    return json(
      { error: `Chat failed: ${error.message}` },
      { status: 500 }
    );
  }
};