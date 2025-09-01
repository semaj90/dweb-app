
import type { RequestHandler } from './$types';

// Test Full Re-embed + Re-rank Loop
// End-to-end testing of document changes, re-embedding, and re-ranking

import { documentUpdateLoop } from "$lib/services/documentUpdateLoop";
import { documents, documentVectors, queryVectors, cases, users } from "$lib/db/schema";
import { URL } from "url";

// ============================================================================
// TEST SCENARIOS
// ============================================================================

const testScenarios = {
  'minor_edit': {
    name: 'Minor Edit Test',
    description: 'Small text change to test low-priority update',
    originalContent: 'This contract establishes the terms and conditions for legal services. The contractor shall provide legal representation and advice.',
    modifiedContent: 'This agreement establishes the terms and conditions for legal services. The contractor shall provide legal representation and advice.',
    expectedPriority: 'low'
  },
  
  'major_revision': {
    name: 'Major Revision Test', 
    description: 'Significant content change to test high-priority update',
    originalContent: 'This contract establishes the terms and conditions for legal services. The contractor shall provide legal representation and advice.',
    modifiedContent: 'This employment agreement defines the relationship between the company and employee. The employee shall perform software development duties and maintain confidentiality of proprietary information.',
    expectedPriority: 'high'
  },
  
  'content_addition': {
    name: 'Content Addition Test',
    description: 'Adding new content to test medium-priority update',
    originalContent: 'This contract establishes the terms and conditions for legal services.',
    modifiedContent: 'This contract establishes the terms and conditions for legal services. Additional clauses include liability limitations, intellectual property provisions, and termination procedures. The contractor agrees to maintain professional standards.',
    expectedPriority: 'medium'
  },
  
  'complete_rewrite': {
    name: 'Complete Rewrite Test',
    description: 'Total content replacement to test critical-priority update',
    originalContent: 'This contract establishes the terms and conditions for legal services. The contractor shall provide legal representation and advice.',
    modifiedContent: 'CONFIDENTIAL SETTLEMENT AGREEMENT - This settlement resolves all claims between parties regarding patent infringement allegations. Payment terms are $500,000 over 12 months with mutual non-disclosure requirements.',
    expectedPriority: 'critical'
  }
};

// ============================================================================
// TEST EXECUTION
// ============================================================================

class UpdateLoopTester {
  private testResults: any = {};
  private testDocumentIds: string[] = [];

  async runFullTest(scenarioName?: string): Promise<any> {
    const startTime = Date.now();
    console.log('🧪 Starting full update loop test...');

    try {
      this.testResults = {
        timestamp: new Date().toISOString(),
        scenario: scenarioName || 'all',
        status: 'running',
        steps: {},
        performance: {},
        errors: []
      };

      // Step 1: Setup test environment
      await this.setupTestEnvironment();

      // Step 2: Run test scenarios
      if (scenarioName && testScenarios[scenarioName as keyof typeof testScenarios]) {
        await this.runTestScenario(scenarioName);
      } else {
        for (const [name, scenario] of Object.entries(testScenarios)) {
          await this.runTestScenario(name);
        }
      }

      // Step 3: Test search impact
      await this.testSearchImpact();

      // Step 4: Performance analysis
      await this.analyzePerformance();

      // Step 5: Cleanup
      await this.cleanup();

      this.testResults.status = 'completed';
      this.testResults.totalTime = Date.now() - startTime;

      return this.testResults;

    } catch (err: any) {
      this.testResults.status = 'failed';
      this.testResults.error = err instanceof Error ? err.message : 'Unknown error';
      this.testResults.totalTime = Date.now() - startTime;
      
      console.error('❌ Update loop test failed:', err);
      return this.testResults;
    }
  }

  private async setupTestEnvironment() {
    const stepStart = Date.now();
    console.log('🔧 Setting up test environment...');

    try {
      // Create test user
      const [testUser] = await db.insert(users).values({
        email: 'test@update-loop.com',
        name: 'Update Loop Test User',
        role: 'prosecutor',
        passwordHash: 'test-hash'
      }).onConflictDoUpdate({
        target: users.email,
        set: { name: 'Update Loop Test User (Updated)' }
      }).returning();

      // Create test case
      const [testCase] = await db.insert(cases).values({
        title: 'Update Loop Test Case',
        description: 'Test case for document update loop validation',
        status: 'active',
        priority: 'medium',
        createdBy: testUser.id
      }).returning();

      this.testResults.steps.setup = {
        status: 'success',
        userId: testUser.id,
        caseId: testCase.id,
        time: Date.now() - stepStart
      };

      this.testResults.testUserId = testUser.id;
      this.testResults.testCaseId = testCase.id;

      console.log('✅ Test environment ready');

    } catch (err: any) {
      this.testResults.steps.setup = {
        status: 'failed',
        error: err instanceof Error ? err.message : 'Setup failed'
      };
      throw err;
    }
  }

  private async runTestScenario(scenarioName: string) {
    const scenario = testScenarios[scenarioName as keyof typeof testScenarios];
    const stepStart = Date.now();
    
    console.log(`📝 Running scenario: ${scenario.name}`);

    try {
      // Create initial document
      const [document] = await db.insert(documents).values({
        caseId: this.testResults.testCaseId,
        filename: `${scenarioName}_test.txt`,
        filePath: `/test/${scenarioName}_test.txt`,
        extractedText: scenario.originalContent,
        createdBy: this.testResults.testUserId
      }).returning();

      this.testDocumentIds.push(document.id);

      // Initial embedding (simulate existing document)
      await documentUpdateLoop.queueDocumentUpdate(document.id, scenario.originalContent);
      await new Promise((resolve: any) => setTimeout(resolve, 1000)); // Wait for processing

      // Create some test queries that would return this document
      const testQuery = this.generateTestQuery(scenario.originalContent);
      const queryEmbedding = await documentUpdateLoop['embeddings'].embedQuery(testQuery);
      
      await db.insert(queryVectors).values({
        userId: this.testResults.testUserId,
        query: testQuery,
        embedding: queryEmbedding,
        resultCount: 1,
        clickedResults: [{ id: document.id, score: 0.8 }]
      });

      // NOW TEST THE UPDATE LOOP
      const changeDetection = await documentUpdateLoop.detectDocumentChanges(
        document.id, 
        scenario.modifiedContent
      );

      let updateResult;
      let rerankingResult;

      if (changeDetection) {
        // Test the full update loop
        updateResult = await documentUpdateLoop.reembedDocument(changeDetection);
        rerankingResult = await documentUpdateLoop.rerankAffectedQueries(document.id);
      }

      this.testResults.steps[scenarioName] = {
        status: 'success',
        documentId: document.id,
        changeDetected: !!changeDetection,
        detectedPriority: changeDetection?.priority,
        expectedPriority: scenario.expectedPriority,
        priorityMatch: changeDetection?.priority === scenario.expectedPriority,
        updateResult: updateResult ? {
          chunksUpdated: updateResult.chunksUpdated,
          processingTime: updateResult.processingTime
        } : null,
        rerankingResult: rerankingResult ? {
          queriesAffected: rerankingResult.length,
          avgImprovement: rerankingResult.reduce((sum, job) => sum + job.improvement, 0) / rerankingResult.length
        } : null,
        time: Date.now() - stepStart
      };

      console.log(`✅ Scenario ${scenario.name} completed`);

    } catch (err: any) {
      this.testResults.steps[scenarioName] = {
        status: 'failed',
        error: err instanceof Error ? err.message : 'Scenario failed'
      };
      
      console.error(`❌ Scenario ${scenario.name} failed:`, err);
    }
  }

  private generateTestQuery(content: string): string {
    // Extract key terms for realistic query
    const words = content.toLowerCase().split(' ').filter((word: any) => word.length > 4 && !['this', 'that', 'with', 'from', 'they', 'have', 'will', 'been'].includes(word)
    );
    
    return words.slice(0, 3).join(' ');
  }

  private async testSearchImpact() {
    const stepStart = Date.now();
    console.log('🔍 Testing search impact...');

    try {
      const searchTests = [];

      for (const documentId of this.testDocumentIds) {
        // Get document content
        const [doc] = await db
          .select()
          .from(documents)
          .where(eq(documents.id, documentId))
          .limit(1);

        if (!doc) continue;

        // Test search before and after
        const testQuery = this.generateTestQuery(doc.extractedText || '');
        const queryEmbedding = await documentUpdateLoop['embeddings'].embedQuery(testQuery);

        const searchResults = await db
          .select({
            documentId: documentVectors.documentId,
            similarity: sql<number>`1 - (${documentVectors.embedding} <=> ${queryEmbedding})`
          })
          .from(documentVectors)
          .where(
            sql`1 - (${documentVectors.embedding} <=> ${queryEmbedding}) > 0.3`
          )
          .orderBy(sql`${documentVectors.embedding} <=> ${queryEmbedding}`)
          .limit(10);

        const relevantResult = searchResults.find((r: any) => r.documentId === documentId);

        searchTests.push({
          documentId,
          query: testQuery,
          found: !!relevantResult,
          similarity: relevantResult?.similarity || 0,
          rank: relevantResult ? searchResults.findIndex((r: any) => r.documentId === documentId) + 1 : -1
        });
      }

      this.testResults.steps.searchImpact = {
        status: 'success',
        searchTests,
        documentsFound: searchTests.filter((t: any) => t.found).length,
        avgSimilarity: searchTests.reduce((sum, t) => sum + t.similarity, 0) / searchTests.length,
        time: Date.now() - stepStart
      };

      console.log('✅ Search impact testing completed');

    } catch (err: any) {
      this.testResults.steps.searchImpact = {
        status: 'failed',
        error: err instanceof Error ? err.message : 'Search testing failed'
      };
    }
  }

  private async analyzePerformance() {
    const stepStart = Date.now();
    console.log('📊 Analyzing performance...');

    try {
      // Get queue status
      const queueStatus = await documentUpdateLoop.getQueueStatus();

      // Calculate average processing times
      const processingTimes = Object.values(this.testResults.steps)
        .filter((step: any) => step.updateResult?.processingTime)
        .map((step: any) => step.updateResult.processingTime);

      const avgProcessingTime = processingTimes.length > 0 
        ? processingTimes.reduce((sum: number, time: number) => sum + time, 0) / processingTimes.length
        : 0;

      this.testResults.performance = {
        avgProcessingTime,
        totalDocumentsTested: this.testDocumentIds.length,
        successfulUpdates: Object.values(this.testResults.steps).filter((step: any) => step.status === 'success').length,
        queueStatus,
        priorityDistribution: Object.values(this.testResults.steps)
          .filter((step: any) => step.detectedPriority)
          .reduce((acc: any, step: any) => {
            acc[step.detectedPriority] = (acc[step.detectedPriority] || 0) + 1;
            return acc;
          }, {}),
        time: Date.now() - stepStart
      };

      console.log('✅ Performance analysis completed');

    } catch (err: any) {
      this.testResults.performance = {
        error: err instanceof Error ? err.message : 'Performance analysis failed'
      };
    }
  }

  private async cleanup() {
    const stepStart = Date.now();
    console.log('🧹 Cleaning up test data...');

    try {
      // Delete test documents and vectors
      for (const documentId of this.testDocumentIds) {
        await db.delete(documentVectors).where(eq(documentVectors.documentId, documentId));
        await db.delete(documents).where(eq(documents.id, documentId));
      }

      // Delete test queries
      await db.delete(queryVectors).where(eq(queryVectors.userId, this.testResults.testUserId));

      this.testResults.steps.cleanup = {
        status: 'success',
        documentsDeleted: this.testDocumentIds.length,
        time: Date.now() - stepStart
      };

      console.log('✅ Cleanup completed');

    } catch (err: any) {
      this.testResults.steps.cleanup = {
        status: 'failed',
        error: err instanceof Error ? err.message : 'Cleanup failed'
      };
    }
  }
}

// ============================================================================
// API HANDLERS
// ============================================================================

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { scenario = 'all', action = 'test' } = await request.json();

    const tester = new UpdateLoopTester();

    if (action === 'test') {
      const results = await tester.runFullTest(scenario);
      return json({
        success: true,
        data: results
      });
    }

    return json({ error: 'Unknown action' }, { status: 400 });

  } catch (err: any) {
    console.error('❌ Update loop test error:', err);
    return json({
      success: false,
      error: err instanceof Error ? err.message : 'Test failed'
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get('action') || 'info';

    if (action === 'info') {
      return json({
        success: true,
        data: {
          service: 'Document Update Loop Tester',
          scenarios: Object.keys(testScenarios),
          description: 'Tests the full re-embed + re-rank loop with various document change scenarios',
          endpoints: {
            'POST /': 'Run full test suite or specific scenario',
            'GET /?action=scenarios': 'List available test scenarios'
          }
        }
      });
    }

    if (action === 'scenarios') {
      return json({
        success: true,
        data: {
          scenarios: Object.entries(testScenarios).map(([key, scenario]) => ({
            key,
            name: scenario.name,
            description: scenario.description,
            expectedPriority: scenario.expectedPriority
          }))
        }
      });
    }

    return json({ error: 'Unknown action' }, { status: 400 });

  } catch (err: any) {
    return json({
      success: false,
      error: err instanceof Error ? err.message : 'Request failed'
    }, { status: 500 });
  }
};