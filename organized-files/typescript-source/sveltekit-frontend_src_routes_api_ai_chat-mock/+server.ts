// @ts-nocheck
import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

// Mock AI chat endpoint for development/testing
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { message, sessionId, context, stream } = await request.json();
    
    // Simulate AI response with mock legal knowledge
    const mockResponses = [
      "As a legal AI assistant, I can help you with various legal matters. This is a mock response for development purposes.",
      "Legal precedent suggests that evidence admissibility depends on several factors including relevance, reliability, and proper chain of custody.",
      "For digital evidence analysis, it's important to consider metadata integrity, hash verification, and forensic imaging procedures.",
      "Constitutional protections in search and seizure cases are governed by the Fourth Amendment, with exceptions for exigent circumstances.",
      "Chain of custody requirements ensure evidence integrity from collection through court presentation, requiring detailed documentation."
    ];
    
    const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
    
    if (stream) {
      // For streaming responses, return a simple JSON for now
      return json({
        content: randomResponse,
        done: true,
        model: 'mock-legal-ai',
        confidence: 0.85,
        executionTime: Math.floor(Math.random() * 1000) + 500,
        sources: [
          {
            type: 'Legal Database',
            score: 0.92,
            title: 'Mock Legal Source'
          }
        ],
        citations: [
          {
            id: 'mock-1',
            title: 'Mock Legal Citation',
            relevance: 0.88,
            type: 'case'
          }
        ]
      });
    }
    
    return json({
      message: {
        id: crypto.randomUUID(),
        content: randomResponse,
        role: 'assistant',
        timestamp: new Date(),
        sources: [
          {
            type: 'Legal Database',
            score: 0.92,
            title: 'Mock Legal Source'
          }
        ],
        metadata: {
          model: 'mock-legal-ai',
          confidence: 0.85,
          executionTime: Math.floor(Math.random() * 1000) + 500,
          fromCache: false
        }
      },
      success: true,
      mock: true
    });
  } catch (error) {
    return json({
      success: false,
      error: 'Mock chat failed',
      message: (error as Error).message
    }, { status: 500 });
  }
};