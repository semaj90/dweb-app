/**
 * Contextual Prompting and Reinforcement Learning Validator
 * Comprehensive testing and validation system for AI contextual features
 * Integrates with BVector store for real-world performance validation
 */

import { EnhancedBVectorStore } from '../services/enhanced-bvector-store';
import { EmbeddingWorkerManager } from '../workers/embedding-worker';
import type { BVectorEntry, SearchOptions, SearchResult } from '../services/enhanced-bvector-store';

export interface ContextualPromptTest {
  id: string;
  description: string;
  userContext: {
    userId: string;
    userRole: 'prosecutor' | 'detective' | 'admin';
    currentCase: string;
    workflowStage: string;
    recentQueries: string[];
    preferences?: Record<string, any>;
  };
  baseQuery: string;
  expectedEnhancements: string[];
  minimumContextScore: number;
}

export interface ReinforcementLearningTest {
  id: string;
  description: string;
  userId: string;
  interactionSequence: Array<{
    query: string;
    expectedResults: number; // Expected number of results
    userActions: {
      clickedIndices: number[];
      rating: number; // 1-5 scale
      timeSpent: number; // seconds
      followUpQuery?: string;
    };
  }>;
  expectedImprovement: {
    personalizedRankingChange: number; // Expected change in ranking accuracy
    satisfactionIncrease: number; // Expected satisfaction score increase
    contextualBoostIncrease: number; // Expected contextual boost improvement
  };
}

export interface ValidationResult {
  testId: string;
  testType: 'contextual' | 'reinforcement';
  success: boolean;
  score: number;
  actualValue: number;
  expectedValue: number;
  details: Record<string, any>;
  error?: string;
}

export class ContextualRLValidator {
  private bvectorStore: EnhancedBVectorStore;
  private embeddingWorker: EmbeddingWorkerManager;
  private testData: Map<string, any> = new Map();

  constructor(bvectorStore: EnhancedBVectorStore) {
    this.bvectorStore = bvectorStore;
    this.embeddingWorker = new EmbeddingWorkerManager();
  }

  async validateContextualPrompting(tests: ContextualPromptTest[]): Promise<ValidationResult[]> {
    const results: ValidationResult[] = [];

    for (const test of tests) {
      try {
        console.log(`🧪 Running contextual prompt test: ${test.description}`);
        
        // Generate baseline search without context
        const baselineResults = await this.bvectorStore.search(test.baseQuery, {
          topK: 10,
          threshold: 0.1
        });

        // Generate contextual prompt
        const contextualPrompts = await this.bvectorStore.generateContextualPrompts(
          test.baseQuery,
          test.userContext
        );

        // Validate contextual enhancements
        const enhancementScore = this.validateContextualEnhancements(
          contextualPrompts.enhancedPrompt,
          test.expectedEnhancements
        );

        // Test contextual search with enhanced prompt
        const contextualResults = await this.bvectorStore.contextualSearch(
          test.baseQuery,
          {
            userContext: test.userContext,
            topK: 10,
            enableRL: false // Pure contextual test
          }
        );

        // Calculate contextual boost effectiveness
        const contextualBoostScore = this.calculateContextualBoostScore(
          baselineResults,
          contextualResults
        );

        // Validate context-aware ranking
        const rankingImprovement = this.validateContextualRanking(
          contextualResults,
          test.userContext
        );

        const overallScore = (enhancementScore + contextualBoostScore + rankingImprovement) / 3;
        const success = overallScore >= test.minimumContextScore;

        results.push({
          testId: test.id,
          testType: 'contextual',
          success,
          score: overallScore,
          actualValue: overallScore,
          expectedValue: test.minimumContextScore,
          details: {
            enhancementScore,
            contextualBoostScore,
            rankingImprovement,
            enhancedPromptLength: contextualPrompts.enhancedPrompt.length,
            contextualReferencesCount: contextualPrompts.contextualReferences.length,
            baselineResultsCount: baselineResults.length,
            contextualResultsCount: contextualResults.length,
            averageContextualBoost: contextualResults.reduce((sum, result) => 
              sum + result.contextualBoost, 0) / contextualResults.length
          }
        });

      } catch (error) {
        results.push({
          testId: test.id,
          testType: 'contextual',
          success: false,
          score: 0,
          actualValue: 0,
          expectedValue: test.minimumContextScore,
          details: {},
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return results;
  }

  async validateReinforcementLearning(tests: ReinforcementLearningTest[]): Promise<ValidationResult[]> {
    const results: ValidationResult[] = [];

    for (const test of tests) {
      try {
        console.log(`🧠 Running reinforcement learning test: ${test.description}`);
        
        // Clear user history for clean test
        await this.bvectorStore.clearUserHistory(test.userId);
        
        // Capture baseline performance
        const baselineResults = await this.bvectorStore.searchWithRL(
          test.interactionSequence[0].query,
          {
            userId: test.userId,
            topK: 10,
            usePersonalization: false
          }
        );

        const baselineAccuracy = this.calculateRankingAccuracy(
          baselineResults,
          test.interactionSequence[0].userActions.clickedIndices
        );

        // Simulate user interaction sequence
        let cumulativeSatisfaction = 0;
        let interactionCount = 0;

        for (const interaction of test.interactionSequence) {
          // Perform search
          const searchResults = await this.bvectorStore.searchWithRL(
            interaction.query,
            {
              userId: test.userId,
              topK: 10,
              usePersonalization: true
            }
          );

          // Record user feedback
          await this.bvectorStore.recordUserFeedback({
            userId: test.userId,
            query: interaction.query,
            results: searchResults,
            selectedIndices: interaction.userActions.clickedIndices,
            userSatisfaction: interaction.userActions.rating / 5, // Normalize to 0-1
            contextualRelevance: this.calculateContextualRelevance(
              searchResults,
              interaction.userActions.clickedIndices
            ),
            followUpSuccess: !interaction.userActions.followUpQuery,
            timeSpent: interaction.userActions.timeSpent
          });

          // Update user preferences based on interactions
          await this.bvectorStore.updateUserPreferences(test.userId, {
            searchPatterns: [interaction.query],
            satisfactionHistory: [interaction.userActions.rating / 5],
            clickPatterns: interaction.userActions.clickedIndices,
            timeSpentHistory: [interaction.userActions.timeSpent]
          });

          cumulativeSatisfaction += interaction.userActions.rating / 5;
          interactionCount++;

          // Simulate follow-up query if present
          if (interaction.userActions.followUpQuery) {
            await this.bvectorStore.searchWithRL(
              interaction.userActions.followUpQuery,
              {
                userId: test.userId,
                topK: 5,
                usePersonalization: true
              }
            );
          }
        }

        // Test final performance after learning
        const finalResults = await this.bvectorStore.searchWithRL(
          test.interactionSequence[0].query,
          {
            userId: test.userId,
            topK: 10,
            usePersonalization: true
          }
        );

        const finalAccuracy = this.calculateRankingAccuracy(
          finalResults,
          test.interactionSequence[0].userActions.clickedIndices
        );

        // Calculate improvements
        const rankingImprovement = finalAccuracy - baselineAccuracy;
        const averageSatisfaction = cumulativeSatisfaction / interactionCount;
        
        const averageContextualBoost = finalResults.reduce((sum, result) => 
          sum + result.contextualBoost, 0) / finalResults.length;

        // Validate against expected improvements
        const rankingSuccess = rankingImprovement >= test.expectedImprovement.personalizedRankingChange;
        const satisfactionSuccess = averageSatisfaction >= test.expectedImprovement.satisfactionIncrease;
        const contextualSuccess = averageContextualBoost >= test.expectedImprovement.contextualBoostIncrease;

        const overallScore = (
          (rankingSuccess ? 1 : 0) +
          (satisfactionSuccess ? 1 : 0) +
          (contextualSuccess ? 1 : 0)
        ) / 3;

        const success = overallScore >= 0.67; // At least 2/3 criteria met

        results.push({
          testId: test.id,
          testType: 'reinforcement',
          success,
          score: overallScore,
          actualValue: overallScore,
          expectedValue: 0.67,
          details: {
            baselineAccuracy,
            finalAccuracy,
            rankingImprovement,
            averageSatisfaction,
            averageContextualBoost,
            interactionsProcessed: interactionCount,
            rankingSuccess,
            satisfactionSuccess,
            contextualSuccess,
            userPreferences: await this.bvectorStore.getUserPreferences(test.userId)
          }
        });

      } catch (error) {
        results.push({
          testId: test.id,
          testType: 'reinforcement',
          success: false,
          score: 0,
          actualValue: 0,
          expectedValue: 0.67,
          details: {},
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return results;
  }

  private validateContextualEnhancements(enhancedPrompt: string, expectedEnhancements: string[]): number {
    let foundEnhancements = 0;
    const lowercasePrompt = enhancedPrompt.toLowerCase();

    for (const enhancement of expectedEnhancements) {
      if (lowercasePrompt.includes(enhancement.toLowerCase())) {
        foundEnhancements++;
      }
    }

    return foundEnhancements / expectedEnhancements.length;
  }

  private calculateContextualBoostScore(baseline: SearchResult[], contextual: SearchResult[]): number {
    if (contextual.length === 0) return 0;

    const averageBoost = contextual.reduce((sum, result) => 
      sum + (result.contextualBoost || 0), 0) / contextual.length;

    // Score based on average boost (expecting positive values)
    return Math.min(averageBoost / 0.5, 1); // Max score at 0.5 boost
  }

  private validateContextualRanking(results: SearchResult[], userContext: any): number {
    if (results.length === 0) return 0;

    let contextualRelevanceScore = 0;
    
    // Check if results are properly ranked by contextual relevance
    for (let i = 0; i < results.length - 1; i++) {
      const current = results[i];
      const next = results[i + 1];
      
      // Higher ranked results should have better contextual alignment
      if (this.isContextuallyRelevant(current, userContext) >= 
          this.isContextuallyRelevant(next, userContext)) {
        contextualRelevanceScore += 1;
      }
    }

    return contextualRelevanceScore / Math.max(results.length - 1, 1);
  }

  private isContextuallyRelevant(result: SearchResult, userContext: any): number {
    let relevance = 0;

    // Check domain alignment
    if (result.metadata.legalDomain && userContext.preferences?.domains?.includes(result.metadata.legalDomain)) {
      relevance += 0.3;
    }

    // Check case type alignment
    if (result.metadata.caseType && userContext.currentCase?.includes(result.metadata.caseType)) {
      relevance += 0.3;
    }

    // Check role-specific relevance
    if (result.metadata.userRole === userContext.userRole) {
      relevance += 0.2;
    }

    // Check recent query alignment
    if (userContext.recentQueries) {
      for (const recentQuery of userContext.recentQueries) {
        if (result.content.toLowerCase().includes(recentQuery.toLowerCase())) {
          relevance += 0.2;
          break;
        }
      }
    }

    return Math.min(relevance, 1);
  }

  private calculateRankingAccuracy(results: SearchResult[], clickedIndices: number[]): number {
    if (clickedIndices.length === 0) return 0;

    // Calculate Mean Reciprocal Rank (MRR)
    let reciprocalSum = 0;
    
    for (const clickedIndex of clickedIndices) {
      if (clickedIndex < results.length) {
        reciprocalSum += 1 / (clickedIndex + 1); // +1 because rank starts at 1
      }
    }

    return reciprocalSum / clickedIndices.length;
  }

  private calculateContextualRelevance(results: SearchResult[], clickedIndices: number[]): number {
    if (clickedIndices.length === 0 || results.length === 0) return 0;

    let totalContextualBoost = 0;
    
    for (const index of clickedIndices) {
      if (index < results.length) {
        totalContextualBoost += results[index].contextualBoost || 0;
      }
    }

    return totalContextualBoost / clickedIndices.length;
  }

  // Predefined test cases for common legal AI scenarios
  static getStandardContextualTests(): ContextualPromptTest[] {
    return [
      {
        id: 'prosecutor-contract-dispute',
        description: 'Prosecutor handling contract dispute case',
        userContext: {
          userId: 'prosecutor-001',
          userRole: 'prosecutor',
          currentCase: 'employment-contract-dispute-2024',
          workflowStage: 'evidence-gathering',
          recentQueries: [
            'employment contract terms',
            'breach of contract damages',
            'non-compete clauses'
          ]
        },
        baseQuery: 'What are the legal remedies for contract violations?',
        expectedEnhancements: [
          'employment',
          'prosecutor',
          'damages',
          'evidence',
          'contract breach'
        ],
        minimumContextScore: 0.8
      },
      {
        id: 'detective-digital-evidence',
        description: 'Detective investigating digital evidence case',
        userContext: {
          userId: 'detective-002',
          userRole: 'detective',
          currentCase: 'cybercrime-investigation-2024',
          workflowStage: 'evidence-analysis',
          recentQueries: [
            'digital forensics procedures',
            'chain of custody requirements',
            'metadata preservation'
          ]
        },
        baseQuery: 'How do I preserve digital evidence integrity?',
        expectedEnhancements: [
          'digital',
          'detective',
          'forensics',
          'chain of custody',
          'metadata'
        ],
        minimumContextScore: 0.75
      }
    ];
  }

  static getStandardRLTests(): ReinforcementLearningTest[] {
    return [
      {
        id: 'personalized-contract-search',
        description: 'Learning user preferences for contract law searches',
        userId: 'rl-test-user-contract',
        interactionSequence: [
          {
            query: 'contract liability terms',
            expectedResults: 10,
            userActions: {
              clickedIndices: [0, 2, 4], // User prefers these results
              rating: 4,
              timeSpent: 45
            }
          },
          {
            query: 'employment agreement violations',
            expectedResults: 8,
            userActions: {
              clickedIndices: [1, 3],
              rating: 3,
              timeSpent: 30,
              followUpQuery: 'damages for employment violations'
            }
          },
          {
            query: 'breach of contract remedies',
            expectedResults: 12,
            userActions: {
              clickedIndices: [0, 1, 2, 5],
              rating: 5,
              timeSpent: 60
            }
          }
        ],
        expectedImprovement: {
          personalizedRankingChange: 0.15, // 15% improvement in MRR
          satisfactionIncrease: 0.7, // 70% satisfaction
          contextualBoostIncrease: 0.3 // 30% contextual boost
        }
      },
      {
        id: 'evidence-analysis-learning',
        description: 'Learning user patterns for evidence analysis',
        userId: 'rl-test-user-evidence',
        interactionSequence: [
          {
            query: 'digital evidence chain of custody',
            expectedResults: 8,
            userActions: {
              clickedIndices: [0, 1],
              rating: 3,
              timeSpent: 35
            }
          },
          {
            query: 'forensic evidence preservation',
            expectedResults: 10,
            userActions: {
              clickedIndices: [0, 2, 3],
              rating: 4,
              timeSpent: 50
            }
          },
          {
            query: 'evidence tampering detection',
            expectedResults: 6,
            userActions: {
              clickedIndices: [1, 2],
              rating: 5,
              timeSpent: 40
            }
          }
        ],
        expectedImprovement: {
          personalizedRankingChange: 0.20, // 20% improvement in MRR
          satisfactionIncrease: 0.8, // 80% satisfaction
          contextualBoostIncrease: 0.25 // 25% contextual boost
        }
      }
    ];
  }
}

// Export validation utilities
export { ContextualRLValidator, type ValidationResult };