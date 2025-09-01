import { describe, it, expect, vi } from 'vitest';

// Mock the heavy summarizer service BEFORE importing the route handler
vi.mock('$lib/services/comprehensive-ollama-summarizer', () => {
  return {
    comprehensiveOllamaSummarizer: {
      summarize: vi.fn(async (req: any) => ({
        summary: 'Mock summary for: ' + req.content.slice(0, 30),
        keyPoints: ['Point A', 'Point B'],
        confidence: 0.99,
        processingTime: 10,
        model: 'mock-model',
        metadata: {
          wordCount: req.content.split(/\s+/).length,
          complexity: 'low',
          topKeywords: ['mock', 'summary'],
          entities: []
        }
      }))
    }
  };
});

// Now import the POST handler (will use the mocked summarizer)
import { POST } from '../../../routes/api/ollama/comprehensive-summary/+server';

function makeRequest(body: any) {
  return new Request('http://localhost/api/ollama/comprehensive-summary', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: { 'Content-Type': 'application/json' }
  });
}

describe('POST /api/ollama/comprehensive-summary (mocked)', () => {
  it('returns 400 for invalid content', async () => {
    const response: any = await POST({ request: makeRequest({ content: 'short', type: 'document' }) } as any);
    expect(response.status).toBe(400);
  });

  it('returns success envelope for valid request (summary shape)', async () => {
    const response: any = await POST({ request: makeRequest({ content: 'This is a sufficiently long legal document content used for fast mocked summarization.', type: 'document' }) } as any);
    expect(response.status).toBe(200);
    const data = await response.json();
    expect(data.success).toBe(true);
    expect(typeof data.data.summary).toBe('string');
    expect(data.data.keyPoints.length).toBeGreaterThan(0);
  });
});
