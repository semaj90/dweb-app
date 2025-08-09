// @ts-nocheck
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { gemma3API } from '$lib/ai/gemma3-api';

// Query endpoint
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, ...params } = await request.json();
    
    switch (action) {
      case 'search':
        const results = await gemma3API.query(params.query, params.limit);
        return json({ success: true, results });
        
      case 'analyze':
        const analysis = await gemma3API.analyze(params.caseId, params.analysisType);
        return json({ success: true, analysis });
        
      case 'process':
        const document = await gemma3API.process(params.content, params.metadata);
        return json({ success: true, document });
        
      default:
        return json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Gemma3 API error:', error);
    return json({ error: (error as any)?.message || "Unknown error" }, { status: 500 });
  }
};

// Direct chat endpoint
export const GET: RequestHandler = async ({ url }) => {
  const prompt = url.searchParams.get('prompt');
  
  if (!prompt) {
    return json({ error: 'Missing prompt parameter' }, { status: 400 });
  }
  
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt,
        stream: false,
        options: {
          temperature: 0.7,
          num_predict: 512,
        }
      })
    });
    
    const data = await response.json();
    return json({ response: data.response });
  } catch (error: any) {
    return json({ error: 'Failed to query Gemma3' }, { status: 500 });
  }
};
