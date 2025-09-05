import { json, error } from '@sveltejs/kit';
import envConfig from '../../../../../env-config.mjs';
import type { RequestHandler } from './$types';

// POST /api/llm/summarize - Summarize text using Ollama
export const POST: RequestHandler = async ({ locals, request }): Promise<any> => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  try {
    const body = await request.json();
    const { text, type = 'evidence' } = body;

    if (!text) {
      throw error(400, 'Text is required');
    }

    // Create appropriate prompt based on type
    let prompt = '';
    switch (type) {
      case 'evidence':
        prompt = `Summarize this evidence for a legal report. Focus on key facts, relevant details, and potential legal significance:\n\n${text}`;
        break;
      case 'document':
        prompt = `Provide a concise summary of this legal document, highlighting the main points and important clauses:\n\n${text}`;
        break;
      case 'case':
        prompt = `Summarize this case information for a legal brief. Include key facts, issues, and conclusions:\n\n${text}`;
        break;
      default:
        prompt = `Summarize the following text:\n\n${text}`;
    }

    // Call Ollama API
    const response = await fetch(`${envConfig.OLLAMA_API_URL}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gemma3:latest-legal',
        prompt,
        stream: false,
        options: {
          temperature: 0.3,
          top_p: 0.9,
          max_tokens: 500
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const data = await response.json();

    return json({
      success: true,
      data: {
        summary: data.response || '',
        model: 'gemma3:latest-legal',
        type,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error generating summary:', err);
    throw error(500, 'Failed to generate summary');
  }
};