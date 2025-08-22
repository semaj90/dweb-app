
// API route for AI chat interactions
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ollama, MODELS } from '$lib/ai/ollama';
import { vectorDB } from '$lib/ai/vector-db';
import { langchain } from '$lib/ai/langchain';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { 
      message, 
      model = MODELS.LEGAL_DETAILED,
      systemPrompt,
      useRAG = false,
      caseId,
      stream = false
    } = await request.json();
    
    if (!message) {
      return json({ error: 'Message is required' }, { status: 400 });
    }
    
    let response: string;
    let sources: unknown[] = [];
    
    if (useRAG) {
      // Use RAG with vector search
      const qaChain = langchain.createQAChain();
      const result = await qaChain.call({
        query: message,
      });
      
      response = result.text;
      sources = result.sourceDocuments?.map((doc: unknown) => ({
        content: doc.pageContent.substring(0, 200) + '...',
        metadata: doc.metadata
      })) || [];
      
    } else {
      // Direct LLM response
      const result = await ollama.generate(model, message, {
        system: systemPrompt,
        stream: false
      });
      response = result.response;
    }
    
    // Store in chat history if caseId provided
    if (caseId) {
      await vectorDB.storeChatMessage(caseId, 'user', message);
      await vectorDB.storeChatMessage(caseId, 'assistant', response, { sources });
    }
    
    return json({
      response,
      sources,
      model,
      timestamp: new Date()
    });
    
  } catch (error: unknown) {
    console.error('Chat API error:', error);
    return json({ 
      error: 'Failed to generate response' 
    }, { status: 500 });
  }
};

// GET endpoint for chat history
export const GET: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get('caseId');
    const limit = parseInt(url.searchParams.get('limit') || '50');
    
    if (!caseId) {
      return json({ error: 'Case ID is required' }, { status: 400 });
    }
    
    const history = await vectorDB.getChatHistory(caseId, limit);
    
    return json({ history });
    
  } catch (error: unknown) {
    console.error('Chat history error:', error);
    return json({ 
      error: 'Failed to fetch chat history' 
    }, { status: 500 });
  }
};
