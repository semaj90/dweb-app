import { json, error } from '@sveltejs/kit';
import envConfig from '../../../../../env-config.mjs';
import type { RequestHandler } from './$types';

// POST /api/llm/chat - Interactive chat with Ollama
export const POST: RequestHandler = async ({ locals, request }): Promise<any> => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  try {
    const body = await request.json();
    const { message, context = [], caseId } = body;

    if (!message) {
      throw error(400, 'Message is required');
    }

    // Build conversation context
    let conversation = '';
    if (context && context.length > 0) {
      for (const msg of context.slice(-5)) { // Keep last 5 messages for context
        conversation += `${msg.role}: ${msg.content}\n`;
      }
    }

    // Create system prompt for legal AI assistant
    const systemPrompt = `You are a legal AI assistant helping prosecutors analyze evidence and build cases. You should:
1. Provide accurate legal analysis based on the evidence
2. Suggest relevant legal precedents when appropriate
3. Help identify key facts and potential weaknesses
4. Maintain professional legal terminology
5. Never provide advice that could compromise the integrity of the investigation

${caseId ? `You are currently working on Case ID: ${caseId}` : ''}

${conversation}
Human: ${message}
Assistant:`;

    // Call Ollama API
    const response = await fetch(`${envConfig.OLLAMA_API_URL}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gemma3:latest-legal',
        prompt: systemPrompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 1000
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
        message: data.response || '',
        model: 'gemma3:latest-legal',
        timestamp: new Date().toISOString(),
        caseId: caseId || null
      }
    });

  } catch (err: any) {
    console.error('Error in chat:', err);
    throw error(500, 'Failed to process chat message');
  }
};