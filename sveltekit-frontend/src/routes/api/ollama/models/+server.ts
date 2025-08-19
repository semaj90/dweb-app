import type { OllamaTagsResponse, OllamaModel } from "$lib/types/ollama";
/**
 * Ollama Models API Endpoint
 * Manages available models, status, and pulling new models
 */

import { json } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler

const OLLAMA_BASE_URL = import.meta.env.OLLAMA_URL || 'http://localhost:11434';

export const GET: RequestHandler = async () => {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data: OllamaTagsResponse = await response.json();
    
    // Add additional metadata for each model
    const enhancedModels = data.models.map((model: OllamaModel) => ({
      ...model,
      isLegal: model.name.includes('legal') || model.name.includes('law'),
      isEmbedding: model.name.includes('embed') || model.name.includes('nomic'),
      isChat: !model.name.includes('embed'),
      sizeGB: Math.round(model.size / (1024 * 1024 * 1024) * 100) / 100,
    }));

    return json({
      success: true,
      models: enhancedModels,
      count: enhancedModels.length,
      categories: {
        legal: enhancedModels.filter(m => m.isLegal).length,
        embedding: enhancedModels.filter(m => m.isEmbedding).length,
        chat: enhancedModels.filter(m => m.isChat).length,
      },
    });
  } catch (error) {
    console.error('Error fetching Ollama models:', error);
    return json(
      {
        success: false,
        error: 'Failed to fetch models from Ollama',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, modelName } = await request.json();

    switch (action) {
      case 'pull':
        if (!modelName) {
          return json({ success: false, error: 'Model name is required' }, { status: 400 });
        }

        const pullResponse = await fetch(`${OLLAMA_BASE_URL}/api/pull`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            name: modelName,
          }),
        });

        if (!pullResponse.ok) {
          throw new Error(`Failed to pull model: ${pullResponse.statusText}`);
        }

        return json({
          success: true,
          message: `Started pulling model: ${modelName}`,
          modelName,
        });

      case 'delete':
        if (!modelName) {
          return json({ success: false, error: 'Model name is required' }, { status: 400 });
        }

        const deleteResponse = await fetch(`${OLLAMA_BASE_URL}/api/delete`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            name: modelName,
          }),
        });

        if (!deleteResponse.ok) {
          throw new Error(`Failed to delete model: ${deleteResponse.statusText}`);
        }

        return json({
          success: true,
          message: `Deleted model: ${modelName}`,
          modelName,
        });

      default:
        return json({ success: false, error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Error in Ollama models API:', error);
    return json(
      {
        success: false,
        error: 'Failed to perform model operation',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
};