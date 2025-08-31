import { json } from '@sveltejs/kit';
import type { RequestEvent } from '@sveltejs/kit';
import { db } from '$lib/server/db/index.js';
import { chatSessions, chatMessages, chatRecommendations, users } from '$lib/server/db/schema-unified.js';
import { eq } from 'drizzle-orm';
import { v4 as uuidv4 } from 'uuid';
import { generateEnhancedEmbedding } from '$lib/server/ai/embeddings-enhanced.js';
import { insertChatMessageWithEmbedding, searchSimilarMessages } from '$lib/server/db/pgvector-utils.js';
import type { RequestHandler } from './$types';


/**
 * Database Persistence Test for AI Recommendations
 * Tests database operations, vector embeddings, and recommendations storage
 */
export async function POST({ request }: RequestEvent): Promise<any> {
  const startTime = Date.now();
  const testResults: any[] = [];
  
  try {
    const { testContent = "Test legal recommendation content for database persistence" } = await request.json();
    
    // Test 1: Create test user
    const testUser = await testCreateUser();
    testResults.push(testUser);
    
    if (!testUser.success) {
      return json({
        success: false,
        error: 'Failed to create test user',
        results: testResults
      });
    }
    
    const userId = testUser.data.userId;
    
    // Test 2: Create chat session
    const testSession = await testCreateChatSession(userId);
    testResults.push(testSession);
    
    if (!testSession.success) {
      return json({
        success: false,
        error: 'Failed to create chat session',
        results: testResults
      });
    }
    
    const sessionId = testSession.data.sessionId;
    
    // Test 3: Generate and store embedding
    const testEmbedding = await testEmbeddingGeneration(testContent);
    testResults.push(testEmbedding);
    
    // Test 4: Store chat message with embedding
    const testMessage = await testStoreMessage(sessionId, testContent, testEmbedding.data?.embedding);
    testResults.push(testMessage);
    
    if (!testMessage.success) {
      return json({
        success: false,
        error: 'Failed to store message',
        results: testResults
      });
    }
    
    const messageId = testMessage.data.messageId;
    
    // Test 5: Store recommendations
    const testRecommendations = await testStoreRecommendations(userId, messageId);
    testResults.push(testRecommendations);
    
    // Test 6: Vector similarity search
    const testVectorSearch = await testVectorSimilaritySearch(testEmbedding.data?.embedding);
    testResults.push(testVectorSearch);
    
    // Test 7: Retrieve recommendations
    const testRetrieveRecommendations = await testRetrieveRecommendations(userId);
    testResults.push(testRetrieveRecommendations);
    
    // Test 8: Update recommendation with feedback
    const testFeedback = await testUpdateRecommendationFeedback(userId);
    testResults.push(testFeedback);
    
    // Clean up test data
    await cleanupTestData(userId, sessionId);
    
    // Calculate overall results
    const successCount = testResults.filter(r => r.success).length;
    const totalTests = testResults.length;
    
    return json({
      success: successCount === totalTests,
      summary: {
        totalTests,
        passed: successCount,
        failed: totalTests - successCount,
        processingTime: Date.now() - startTime
      },
      results: testResults,
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    return json({
      success: false,
      error: 'Database persistence test failed',
      details: error instanceof Error ? error.message : 'Unknown error',
      results: testResults,
      processingTime: Date.now() - startTime
    }, { status: 500 });
  }
}

async function testCreateUser(): Promise<any> {
  const startTime = Date.now();
  
  try {
    const userId = uuidv4();
    const testEmail = `test-${Date.now()}@legal-ai.com`;
    
    await db.insert(users).values({
      id: userId,
      email: testEmail,
      displayName: 'Test User',
      role: 'user'
    });
    
    return {
      test: 'Create User',
      success: true,
      message: 'Test user created successfully',
      data: { userId, email: testEmail },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Create User',
      success: false,
      message: 'Failed to create test user',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testCreateChatSession(userId: string): Promise<any> {
  const startTime = Date.now();
  
  try {
    const sessionId = uuidv4();
    
    await db.insert(chatSessions).values({
      id: sessionId,
      userId,
      title: 'Test Session',
      context: { testSession: true },
      metadata: { 
        createdForTesting: true,
        timestamp: new Date().toISOString()
      }
    });
    
    return {
      test: 'Create Chat Session',
      success: true,
      message: 'Chat session created successfully',
      data: { sessionId },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Create Chat Session',
      success: false,
      message: 'Failed to create chat session',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testEmbeddingGeneration(content: string): Promise<any> {
  const startTime = Date.now();
  
  try {
    const embedding = await generateEnhancedEmbedding(content, {
      provider: 'nomic-embed',
      legalDomain: true,
      cache: true
    }) as number[];
    
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new Error('Invalid embedding generated');
    }
    
    return {
      test: 'Embedding Generation',
      success: true,
      message: `Generated ${embedding.length}-dimensional embedding`,
      data: { 
        embedding, 
        dimensions: embedding.length,
        sampleValues: embedding.slice(0, 5)
      },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Embedding Generation',
      success: false,
      message: 'Failed to generate embedding',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testStoreMessage(sessionId: string, content: string, embedding?: number[]): Promise<any> {
  const startTime = Date.now();
  
  try {
    const messageId = uuidv4();
    
    if (embedding && embedding.length > 0) {
      // Use pgvector utility function
      const success = await insertChatMessageWithEmbedding({
        id: messageId,
        sessionId,
        role: 'user',
        content,
        embedding,
        metadata: {
          testMessage: true,
          embeddingDimensions: embedding.length
        }
      });
      
      if (!success) {
        throw new Error('Failed to insert message with embedding');
      }
    } else {
      // Fallback without embedding
      await db.insert(chatMessages).values({
        id: messageId,
        sessionId,
        role: 'user',
        content,
        metadata: {
          testMessage: true,
          embeddingStatus: 'not_available'
        }
      });
    }
    
    return {
      test: 'Store Message',
      success: true,
      message: embedding ? 'Message stored with vector embedding' : 'Message stored without embedding',
      data: { 
        messageId,
        hasEmbedding: !!embedding,
        embeddingDimensions: embedding?.length || 0
      },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Store Message',
      success: false,
      message: 'Failed to store message',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testStoreRecommendations(userId: string, messageId: string): Promise<any> {
  const startTime = Date.now();
  
  try {
    const recommendations = [
      {
        id: uuidv4(),
        type: 'legal_analysis',
        content: 'Test recommendation for evidence authentication procedures',
        confidence: 0.85,
        metadata: {
          category: 'evidence_handling',
          source: 'test',
          reasoning: 'Test recommendation for database persistence'
        }
      },
      {
        id: uuidv4(),
        type: 'procedural_check',
        content: 'Verify chain of custody documentation is complete',
        confidence: 0.78,
        metadata: {
          category: 'compliance',
          source: 'test',
          reasoning: 'Test procedural recommendation'
        }
      }
    ];
    
    for (const rec of recommendations) {
      await db.insert(chatRecommendations).values({
        id: rec.id,
        userId,
        messageId,
        recommendationType: rec.type,
        content: rec.content,
        confidence: rec.confidence,
        metadata: rec.metadata
      });
    }
    
    return {
      test: 'Store Recommendations',
      success: true,
      message: `Stored ${recommendations.length} test recommendations`,
      data: { 
        recommendationsCount: recommendations.length,
        recommendationIds: recommendations.map(r => r.id)
      },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Store Recommendations',
      success: false,
      message: 'Failed to store recommendations',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testVectorSimilaritySearch(embedding?: number[]): Promise<any> {
  const startTime = Date.now();
  
  try {
    if (!embedding || embedding.length === 0) {
      return {
        test: 'Vector Similarity Search',
        success: false,
        message: 'No embedding available for similarity search',
        responseTime: Date.now() - startTime
      };
    }
    
    const similarResults = await searchSimilarMessages(embedding, {
      limit: 5,
      threshold: 0.1, // Low threshold to ensure we find our test message
      includeMetadata: true
    });
    
    return {
      test: 'Vector Similarity Search',
      success: true,
      message: `Found ${similarResults.length} similar messages`,
      data: {
        resultsCount: similarResults.length,
        topSimilarity: similarResults.length > 0 ? similarResults[0].similarity : 0,
        results: similarResults.map(r => ({
          id: r.id,
          similarity: r.similarity,
          contentPreview: r.content.substring(0, 50) + '...'
        }))
      },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Vector Similarity Search',
      success: false,
      message: 'Vector similarity search failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testRetrieveRecommendations(userId: string): Promise<any> {
  const startTime = Date.now();
  
  try {
    const recommendations = await db
      .select({
        id: chatRecommendations.id,
        content: chatRecommendations.content,
        type: chatRecommendations.recommendationType,
        confidence: chatRecommendations.confidence,
        metadata: chatRecommendations.metadata,
        feedback: chatRecommendations.feedback
      })
      .from(chatRecommendations)
      .where(eq(chatRecommendations.userId, userId))
      .limit(10);
    
    return {
      test: 'Retrieve Recommendations',
      success: true,
      message: `Retrieved ${recommendations.length} recommendations`,
      data: {
        recommendationsCount: recommendations.length,
        recommendations: recommendations.map(r => ({
          id: r.id,
          type: r.type,
          confidence: r.confidence,
          contentPreview: r.content.substring(0, 50) + '...',
          hasFeedback: !!r.feedback
        }))
      },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Retrieve Recommendations',
      success: false,
      message: 'Failed to retrieve recommendations',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function testUpdateRecommendationFeedback(userId: string): Promise<any> {
  const startTime = Date.now();
  
  try {
    // Get a test recommendation to update
    const recommendations = await db
      .select({ id: chatRecommendations.id })
      .from(chatRecommendations)
      .where(eq(chatRecommendations.userId, userId))
      .limit(1);
    
    if (recommendations.length === 0) {
      throw new Error('No recommendations found to update');
    }
    
    const recommendationId = recommendations[0].id;
    
    // Update with feedback
    const result = await db
      .update(chatRecommendations)
      .set({
        feedback: 'helpful',
        metadata: {
          feedbackUpdated: true,
          feedbackTimestamp: new Date().toISOString()
        }
      })
      .where(eq(chatRecommendations.id, recommendationId))
      .returning({ id: chatRecommendations.id });
    
    return {
      test: 'Update Recommendation Feedback',
      success: result.length > 0,
      message: 'Recommendation feedback updated successfully',
      data: {
        updatedRecommendationId: recommendationId,
        feedback: 'helpful'
      },
      responseTime: Date.now() - startTime
    };
  } catch (error: any) {
    return {
      test: 'Update Recommendation Feedback',
      success: false,
      message: 'Failed to update recommendation feedback',
      error: error instanceof Error ? error.message : 'Unknown error',
      responseTime: Date.now() - startTime
    };
  }
}

async function cleanupTestData(userId: string, sessionId: string): Promise<any> {
  try {
    // Clean up in reverse order of dependencies
    await db.delete(chatRecommendations).where(eq(chatRecommendations.userId, userId));
    await db.delete(chatMessages).where(eq(chatMessages.sessionId, sessionId));
    await db.delete(chatSessions).where(eq(chatSessions.id, sessionId));
    await db.delete(users).where(eq(users.id, userId));
    
    console.log('Test data cleaned up successfully');
  } catch (error: any) {
    console.warn('Failed to clean up test data:', error);
  }
}

/**
 * GET endpoint for quick health check
 */
export async function GET(): Promise<any> {
  try {
    // Quick database connection test
    await db.select().from(users).limit(1);
    
    return json({
      status: 'healthy',
      message: 'Database persistence test endpoint is operational',
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    return json({
      status: 'unhealthy',
      message: 'Database connection failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
}