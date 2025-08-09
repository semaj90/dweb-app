// @ts-nocheck
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { vectorSearchService } from '$lib/services/vector-search';

// Claude API integration for legal document analysis
async function queryClaudeWithContext(
  userQuery: string,
  context: string,
  sources: any[]
): Promise<{
  response: string;
  confidence: number;
  citedSources: string[];
}> {
  try {
    const prompt = `
You are a legal AI assistant helping with document analysis and case research.

Context from legal documents:
${context}

User Question: ${userQuery}

Please provide a comprehensive legal analysis based on the provided context. 
Include specific citations to the source documents when relevant.
Rate your confidence in the response from 0.0 to 1.0.

Respond in JSON format:
{
  "response": "Your detailed legal analysis here",
  "confidence": 0.85,
  "citedSources": ["filename1.pdf", "filename2.doc"],
  "legalCitations": ["relevant case law or statutes if applicable"],
  "keyFindings": ["bullet point summary of key findings"],
  "recommendations": ["actionable recommendations if applicable"]
}
    `;

    // Note: This would connect to Claude API if available
    // For now, return a structured mock response
    const mockResponse = {
      response: `Based on the provided legal documents, I've analyzed the query "${userQuery}". The search returned ${sources.length} relevant documents with high semantic similarity. Key findings include evidence patterns and relevant legal precedents found in the document context.`,
      confidence: Math.min(0.9, sources.reduce((acc, s) => acc + s.relevanceScore, 0) / sources.length),
      citedSources: sources.map(s: any => s.filename).filter(Boolean),
      legalCitations: [],
      keyFindings: [
        "High relevance documents identified through vector similarity",
        "Semantic search successfully retrieved contextual legal content",
        "Multiple source documents provide comprehensive coverage"
      ],
      recommendations: [
        "Review the cited source documents for detailed analysis",
        "Consider additional searches with refined terminology",
        "Validate findings with current legal precedents"
      ]
    };

    return mockResponse;
  } catch (error) {
    console.error('Claude API error:', error);
    throw error;
  }
}

// Gemini API integration for legal document analysis
async function queryGeminiWithContext(
  userQuery: string,
  context: string,
  sources: any[]
): Promise<{
  response: string;
  confidence: number;
  citedSources: string[];
}> {
  try {
    // Note: This would connect to Gemini API if available
    // For now, return a structured mock response
    const mockResponse = {
      response: `Gemini analysis for: "${userQuery}". Found ${sources.length} relevant legal documents. The semantic search successfully identified contextually relevant content with high confidence scores.`,
      confidence: Math.min(0.88, sources.reduce((acc, s) => acc + s.relevanceScore, 0) / sources.length),
      citedSources: sources.map(s: any => s.filename).filter(Boolean)
    };

    return mockResponse;
  } catch (error) {
    console.error('Gemini API error:', error);
    throw error;
  }
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { 
      query, 
      caseId, 
      model = 'claude', 
      threshold = 0.7, 
      limit = 10 
    } = await request.json();

    if (!query) {
      return json({ error: 'Query is required' }, { status: 400 });
    }

    // Perform vector search
    const searchResults = await vectorSearchService.search(query, {
      caseId,
      threshold,
      limit
    });

    if (searchResults.length === 0) {
      return json({
        response: "No relevant documents found for your query. Please try refining your search terms or check if documents have been properly indexed.",
        confidence: 0.0,
        sources: [],
        searchResults: []
      });
    }

    // Build context for LLM
    const { context, sources, relevanceScores } = await vectorSearchService.buildLegalContext(
      query,
      caseId,
      4000 // Max context length
    );

    // Query the specified model
    let aiResponse;
    if (model === 'claude') {
      aiResponse = await queryClaudeWithContext(query, context, sources);
    } else if (model === 'gemini') {
      aiResponse = await queryGeminiWithContext(query, context, sources);
    } else {
      return json({ error: 'Unsupported model. Use "claude" or "gemini"' }, { status: 400 });
    }

    // Enhanced response with metadata
    const enhancedResponse = {
      ...aiResponse,
      model,
      searchResults: searchResults.map(r: any => ({
        id: r.id,
        filename: r.filename,
        relevanceScore: r.relevanceScore,
        summary: r.summary,
        keywords: r.keywords
      })),
      searchMetadata: {
        totalResults: searchResults.length,
        averageRelevance: relevanceScores.reduce((a, b) => a + b, 0) / relevanceScores.length,
        threshold,
        caseId: caseId || null
      },
      timestamp: new Date().toISOString()
    };

    return json(enhancedResponse);

  } catch (error) {
    console.error('Vector search API error:', error);
    return json({ 
      error: 'Vector search failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const cacheStats = await vectorSearchService.getCacheStats();
    
    return json({
      status: 'Vector search service is running',
      cacheStats,
      endpoints: {
        search: 'POST /api/ai/vector-search',
        index: 'POST /api/ai/vector-search/index',
        stats: 'GET /api/ai/vector-search'
      },
      supportedModels: ['claude', 'gemini'],
      embeddingModels: ['ollama-nomic-embed-text', 'claude-fallback', 'gemini-embed']
    });

  } catch (error) {
    console.error('Vector search status error:', error);
    return json({ 
      error: 'Service unavailable',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};