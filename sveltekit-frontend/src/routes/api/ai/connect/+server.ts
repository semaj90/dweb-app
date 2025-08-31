
import type { RequestHandler } from './$types';

async function fetchWithTimeout(input: RequestInfo, init: RequestInit = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(input, { ...init, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { model } = await request.json();

    // Check if Ollama is running
    const healthResponse = await fetchWithTimeout('http://localhost:11434/api/tags', {}, 10000);

    if (!healthResponse.ok) {
      throw error(503, 'Ollama service is not available');
    }

    const availableModels = await healthResponse.json();
    const modelList = availableModels.models?.map((m: any) => m.name) || [];

    // Verify requested model exists
    if (model && !modelList.includes(model)) {
      throw error(404, `Model '${model}' not found. Available models: ${modelList.join(', ')}`);
    }

    // Test the model with a simple request
    if (model) {
      const testResponse = await fetchWithTimeout(
        'http://localhost:11434/api/generate',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            prompt: 'Test connection',
            stream: false,
            options: { max_tokens: 1 }
          })
        },
        15000
      );

      if (!testResponse.ok) {
        throw error(503, `Failed to initialize model '${model}'`);
      }
    }

    return json({
      success: true,
      model: model || modelList[0] || 'none',
      availableModels: modelList,
      status: 'connected',
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('AI connection error:', err);

    // If this is a SvelteKit HttpError rethrow it
    if (err && typeof err === 'object' && 'status' in err) {
      throw err;
    }

    // Node/Fetch abort results in an AbortError
    if (err && (err.name === 'AbortError' || err.name === 'TimeoutError')) {
      throw error(504, 'Connection timeout. Please check if Ollama is running.');
    }

    if (err && (err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND')) {
      throw error(503, 'Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434');
    }

    throw error(500, `Connection failed: ${err?.message ?? String(err)}`);
  }
};
