import type { RequestHandler } from '@sveltejs/kit';
import { json, error } from '@sveltejs/kit';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { model } = await request.json();
    
    // Check if Ollama is running
    const healthResponse = await fetch('http://localhost:11434/api/tags', {
      signal: AbortSignal.timeout(10000)
    });
    
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
      const testResponse = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt: 'Test connection',
          stream: false,
          options: { max_tokens: 1 }
        }),
        signal: AbortSignal.timeout(15000)
      });
      
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
    
    if (err.status) {
      throw err;
    }
    
    if (err.name === 'TimeoutError') {
      throw error(504, 'Connection timeout. Please check if Ollama is running.');
    }
    
    if (err.code === 'ECONNREFUSED') {
      throw error(503, 'Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434');
    }
    
    throw error(500, `Connection failed: ${err.message}`);
  }
};
