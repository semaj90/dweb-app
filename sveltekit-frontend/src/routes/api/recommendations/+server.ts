
import { json } from "@sveltejs/kit";
import { enhancedSearchWithNeo4j } from "$lib/ai/custom-reranker";
import { mcpContext72GetLibraryDocs } from "$lib/mcp-context72-get-library-docs";
import { userRecommendationService } from '$lib/server/services/user-recommendation-service';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// Memory access helper for MCP integration
async function accessMemoryMCP(query: string, userContext: any) {
  // Simulate memory access - would integrate with actual MCP memory system
  return [
    { relatedId: 'memory-1', content: query, relevance: 0.8 },
    { relatedId: 'memory-2', content: userContext?.lastQuery, relevance: 0.6 }
  ].filter(m => m.content);
}

// Recommendation endpoint using enhanced reranker, Neo4j, and memory
export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const action = url.searchParams.get('action') || 'suggest';
    const body = await request.json();

    switch (action) {
      case 'suggest': {
        const {
          query,
          userContext,
          neo4jContext,
          limit = 5,
        } = body;
        
        // Run enhanced search with Neo4j context
        const reranked = await enhancedSearchWithNeo4j(
          query,
          userContext,
          neo4jContext,
          limit * 2
        );
        
        // Enrich with memory and docs for final scoring
        const memory = await accessMemoryMCP(query, userContext);
        const docs = await mcpContext72GetLibraryDocs("svelte", "runes");
        
        // Final scoring pass
        const recommendations = reranked
          .map((result) => {
            let score = result.rerankScore;
            if (memory.some((m) => m.relatedId === result.id)) score += 1;
            if (docs && docs.includes(result.intent)) score += 1;
            return { ...result, finalScore: score };
          })
          .sort((a, b) => b.finalScore - a.finalScore)
          .slice(0, limit);
        
        return json({ recommendations });
      }

      case 'resume': {
        const { userId } = body;
        
        if (!userId) {
          return json({
            success: false,
            error: 'userId is required'
          }, { status: 400 });
        }

        // Get user patterns and generate resume suggestions
        const patterns = await userRecommendationService.analyzeUserPatterns(userId);
        const suggestions = [
          `Continue reviewing ${patterns.preferredTopics[0]} cases from yesterday?`,
          `Complete the analysis for ${patterns.frequentCases[0]}?`,
          `Review the updated ${patterns.preferredTopics[1]} documents?`
        ].filter(Boolean);

        return json({
          suggestions,
          context: patterns.preferredTopics[0] || 'general-legal',
          lastActivity: new Date().toISOString(),
          userPattern: {
            complexity: patterns.queryComplexity,
            frequency: patterns.usageFrequency,
            activeHours: patterns.timePatterns.mostActiveHours
          }
        });
      }

      case 'trending': {
        // Mock trending searches - would query actual data in production
        return json({
          trending: [
            'contract indemnification clauses',
            'employment termination procedures', 
            'intellectual property licensing',
            'merger and acquisition due diligence',
            'privacy compliance regulations'
          ],
          period: '24h',
          timestamp: Date.now()
        });
      }

      case 'feedback': {
        const { userId, recommendationId, rating, feedback } = body;
        
        if (!userId || !recommendationId || rating === undefined) {
          return json({
            success: false,
            error: 'userId, recommendationId, and rating are required'
          }, { status: 400 });
        }

        // Store feedback for recommendation improvement
        // In production, this would update ML models
        console.log('Recommendation feedback:', { userId, recommendationId, rating, feedback });

        return json({
          success: true,
          message: 'Feedback recorded successfully',
          timestamp: Date.now()
        });
      }

      default:
        return json({
          success: false,
          error: `Unknown action: ${action}`
        }, { status: 400 });
    }

  } catch (error: any) {
    return json(
      { 
        success: false,
        error: error.message || "Failed to generate recommendations" 
      },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const userId = url.searchParams.get('userId');
    const type = url.searchParams.get('type') || 'resume';

    if (type === 'resume' && userId) {
      // "Pick up where you left off" functionality
      const patterns = await userRecommendationService.analyzeUserPatterns(userId);
      const suggestions = await userRecommendationService.generateRecommendations(userId, 3);

      return json({
        suggestions: suggestions.map(s => s.content),
        context: patterns.preferredTopics[0] || 'general-legal',
        lastActivity: new Date().toISOString(),
        recommendations: suggestions,
        userInsights: {
          complexity: patterns.queryComplexity,
          frequency: patterns.usageFrequency,
          topTopics: patterns.preferredTopics.slice(0, 3),
          activeHours: patterns.timePatterns.mostActiveHours
        }
      });
    }

    // Service overview
    return json({
      service: 'recommendation-engine',
      status: 'operational',
      endpoints: {
        suggest: '/api/recommendations?action=suggest (POST)',
        resume: '/api/recommendations?action=resume (POST)',
        trending: '/api/recommendations?action=trending (POST)',
        feedback: '/api/recommendations?action=feedback (POST)',
        user_resume: '/api/recommendations?userId={id}&type=resume (GET)'
      },
      capabilities: [
        '"Pick up where you left off" prompts',
        '"Did you mean" suggestions', 
        '"Others searched for" recommendations',
        'Context-aware assistance',
        'Learning user patterns',
        'Neo4j graph integration',
        'MCP Context7 documentation',
        'User behavior analytics'
      ],
      features: {
        selfPrompting: true,
        userPatterns: true,
        contextAware: true,
        graphEnhanced: true,
        machineLearning: true
      },
      timestamp: Date.now()
    });

  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      timestamp: Date.now()
    }, { status: 500 });
  }
};
