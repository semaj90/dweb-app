import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Mock AI connection endpoint for development/testing
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { model } = await request.json();
    
    // Simulate successful connection with mock data
    return json({
      success: true,
      model: model || 'mock-legal-ai',
      availableModels: ['mock-legal-ai', 'gemma3-legal', 'mistral-7b', 'llama3.1-8b'],
      status: 'connected',
      timestamp: new Date().toISOString(),
      mock: true,
      message: 'Mock AI connection established for development'
    });
  } catch (error) {
    return json({
      success: false,
      error: 'Mock connection failed',
      message: (error as Error).message
    }, { status: 500 });
  }
};