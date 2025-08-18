// Reinforcement Learning Service for YoRHa AI Assistant
import { db } from '../db';
import { conversations, documents, userActivity, knowledgeBase } from '../db/schema';
import { eq, and, gte, sql, desc } from 'drizzle-orm';
import { EnhancedVectorService } from './vector.service';
import { QueueService } from './queue.service';

interface FeedbackData {
  queryId: string;
  responseId: string;
  userId: string;
  rating: number; // 1-5
  helpful: boolean;
  accurate: boolean;
  complete: boolean;
  feedback?: string;
  corrections?: string;
  preferredResponse?: string;
}

interface LearningMetrics {
  accuracy: number;
  helpfulness: number;
  completeness: number;
  userSatisfaction: number;
  responseTime: number;
  sourcesQuality: number;
}

export class ReinforcementLearningService {
  private vectorService: EnhancedVectorService;
  private queueService: QueueService;
  
  // Learning configuration
  private config = {
    minFeedbackForLearning: 10,
    learningRate: 0.01,
    explorationRate: 0.1,
    rewardDecay: 0.95,
    batchSize: 32,
    updateFrequency: 100 // Update model after N interactions
  };
  
  // Performance tracking
  private metrics: Map<string, LearningMetrics> = new Map();
  private rewardHistory: number[] = [];
  private policyUpdates: number = 0;

  constructor() {
    this.vectorService = new EnhancedVectorService();
    this.queueService = new QueueService();
    this.initializeMetrics();
  }

  // Initialize performance metrics
  private async initializeMetrics() {
    // Load historical metrics from database
    const historicalMetrics = await db.execute(sql`
      SELECT 
        category,
        AVG(rating) as avg_rating,
        AVG(CASE WHEN helpful THEN 1 ELSE 0 END) as helpfulness,
        AVG(CASE WHEN accurate THEN 1 ELSE 0 END) as accuracy,
        AVG(response_time) as avg_response_time
      FROM feedback_data
      WHERE created_at > NOW() - INTERVAL '30 days'
      GROUP BY category
    `);

    for (const metric of historicalMetrics.rows) {
      this.metrics.set(metric.category, {
        accuracy: metric.accuracy,
        helpfulness: metric.helpfulness,
        completeness: 0.8, // Default
        userSatisfaction: metric.avg_rating / 5,
        responseTime: metric.avg_response_time,
        sourcesQuality: 0.85 // Default
      });
    }
  }

  // Process user feedback for reinforcement learning
  async processFeedback(feedback: FeedbackData): Promise<void> {
    try {
      // Calculate reward signal
      const reward = this.calculateReward(feedback);
      
      // Store feedback in database
      await this.storeFeedback(feedback, reward);
      
      // Update Q-values for state-action pairs
      await this.updateQValues(feedback, reward);
      
      // Update retrieval policy
      await this.updateRetrievalPolicy(feedback);
      
      // Update response generation policy
      await this.updateGenerationPolicy(feedback);
      
      // Retrain embeddings if needed
      if (this.shouldRetrainEmbeddings()) {
        await this.retrainEmbeddings(feedback);
      }
      
      // Update metrics
      await this.updateMetrics(feedback);
      
      // Trigger policy update if threshold reached
      this.policyUpdates++;
      if (this.policyUpdates >= this.config.updateFrequency) {
        await this.performPolicyUpdate();
        this.policyUpdates = 0;
      }
      
      // Queue analytics
      await this.queueService.publishMessage('analytics', {
        event: 'rl_feedback_processed',
        userId: feedback.userId,
        reward,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to process feedback:', error);
      throw error;
    }
  }

  // Calculate reward signal from feedback
  private calculateReward(feedback: FeedbackData): number {
    let reward = 0;
    
    // Base reward from rating (normalized to -1 to 1)
    reward += (feedback.rating - 3) / 2;
    
    // Bonus for helpful responses
    if (feedback.helpful) reward += 0.3;
    
    // Bonus for accurate responses
    if (feedback.accurate) reward += 0.4;
    
    // Bonus for complete responses
    if (feedback.complete) reward += 0.3;
    
    // Penalty for corrections needed
    if (feedback.corrections) reward -= 0.5;
    
    // Apply reward decay based on response time
    const timeFactor = this.config.rewardDecay;
    reward *= timeFactor;
    
    // Store in history
    this.rewardHistory.push(reward);
    
    return reward;
  }

  // Update Q-values for reinforcement learning
  private async updateQValues(feedback: FeedbackData, reward: number): Promise<void> {
    // Get the original query and response
    const interaction = await db.query.conversations.findFirst({
      where: eq(conversations.id, feedback.queryId)
    });
    
    if (!interaction) return;
    
    // Extract state features
    const state = this.extractStateFeatures(interaction);
    
    // Extract action features (retrieval and generation choices)
    const action = this.extractActionFeatures(interaction);
    
    // Update Q-value using Bellman equation
    const currentQ = await this.getQValue(state, action);
    const maxFutureQ = await this.getMaxFutureQValue(state);
    
    const newQ = currentQ + this.config.learningRate * (
      reward + this.config.rewardDecay * maxFutureQ - currentQ
    );
    
    // Store updated Q-value
    await this.storeQValue(state, action, newQ);
  }

  // Update retrieval policy based on feedback
  private async updateRetrievalPolicy(feedback: FeedbackData): Promise<void> {
    // Get the sources used in the response
    const response = await db.execute(sql`
      SELECT sources, retrieval_strategy
      FROM rag_responses
      WHERE id = ${feedback.responseId}
    `);
    
    if (!response.rows[0]) return;
    
    const sources = response.rows[0].sources;
    const strategy = response.rows[0].retrieval_strategy;
    
    // Update source quality scores
    for (const source of sources) {
      const qualityUpdate = feedback.accurate ? 0.1 : -0.1;
      
      await db.execute(sql`
        UPDATE documents
        SET quality_score = quality_score + ${qualityUpdate},
            usage_count = usage_count + 1,
            last_used = NOW()
        WHERE id = ${source.documentId}
      `);
    }
    
    // Update retrieval strategy preferences
    if (feedback.helpful && feedback.accurate) {
      await this.increaseStrategyWeight(strategy);
    } else {
      await this.decreaseStrategyWeight(strategy);
    }
  }

  // Update generation policy based on feedback
  private async updateGenerationPolicy(feedback: FeedbackData): Promise<void> {
    if (feedback.preferredResponse) {
      // Learn from user's preferred response
      const embedding = await this.vectorService.generateEmbedding(
        feedback.preferredResponse,
        'search_document'
      );
      
      // Store as positive example
      await db.insert(knowledgeBase).values({
        category: 'user_preference',
        title: 'Preferred Response Pattern',
        content: feedback.preferredResponse,
        tags: ['reinforcement_learning', 'positive_example'],
        metadata: {
          userId: feedback.userId,
          originalQuery: feedback.queryId,
          reward
        },
        embedding: sql`${embedding}::vector(768)`
      });
    }
    
    if (feedback.corrections) {
      // Learn from corrections
      const correctionEmbedding = await this.vectorService.generateEmbedding(
        feedback.corrections,
        'search_document'
      );
      
      // Store corrections for future reference
      await db.insert(knowledgeBase).values({
        category: 'corrections',
        title: 'Response Correction',
        content: feedback.corrections,
        tags: ['reinforcement_learning', 'correction'],
        metadata: {
          userId: feedback.userId,
          originalQuery: feedback.queryId
        },
        embedding: sql`${correctionEmbedding}::vector(768)`
      });
    }
  }

  // Decide if embeddings should be retrained
  private shouldRetrainEmbeddings(): boolean {
    // Check if we have enough feedback
    if (this.rewardHistory.length < this.config.minFeedbackForLearning) {
      return false;
    }
    
    // Check if performance is declining
    const recentRewards = this.rewardHistory.slice(-10);
    const avgRecentReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
    const overallAvgReward = this.rewardHistory.reduce((a, b) => a + b, 0) / this.rewardHistory.length;
    
    return avgRecentReward < overallAvgReward * 0.9;
  }

  // Retrain embeddings with feedback
  private async retrainEmbeddings(feedback: FeedbackData): Promise<void> {
    // Get positive and negative examples
    const positiveExamples = await db.execute(sql`
      SELECT content, embedding
      FROM knowledge_base
      WHERE category = 'user_preference'
        AND metadata->>'reward' > '0.5'
      ORDER BY created_at DESC
      LIMIT 100
    `);
    
    const negativeExamples = await db.execute(sql`
      SELECT content, embedding
      FROM knowledge_base
      WHERE category = 'corrections'
      ORDER BY created_at DESC
      LIMIT 100
    `);
    
    // Create training batch
    const trainingData = {
      positive: positiveExamples.rows,
      negative: negativeExamples.rows,
      userId: feedback.userId
    };
    
    // Queue for fine-tuning
    await this.queueService.publishMessage('model_training', {
      type: 'embedding_finetuning',
      data: trainingData,
      timestamp: new Date().toISOString()
    });
  }

  // Perform policy update
  private async performPolicyUpdate(): Promise<void> {
    console.log('Performing policy update...');
    
    // Get recent interactions
    const recentInteractions = await db.execute(sql`
      SELECT 
        q.query,
        r.response,
        f.rating,
        f.helpful,
        f.accurate,
        f.complete
      FROM queries q
      JOIN responses r ON q.id = r.query_id
      JOIN feedback_data f ON r.id = f.response_id
      WHERE f.created_at > NOW() - INTERVAL '7 days'
      ORDER BY f.created_at DESC
      LIMIT ${this.config.batchSize}
    `);
    
    // Calculate policy gradients
    const policyGradients = await this.calculatePolicyGradients(recentInteractions.rows);
    
    // Update retrieval weights
    await this.updateRetrievalWeights(policyGradients);
    
    // Update generation parameters
    await this.updateGenerationParameters(policyGradients);
    
    // Store policy version
    await db.execute(sql`
      INSERT INTO policy_versions (version, metrics, gradients, created_at)
      VALUES (
        ${Date.now()},
        ${JSON.stringify(Object.fromEntries(this.metrics))},
        ${JSON.stringify(policyGradients)},
        NOW()
      )
    `);
  }

  // Active learning - identify areas needing improvement
  async identifyLearningOpportunities(): Promise<any[]> {
    // Find queries with low confidence or poor feedback
    const opportunities = await db.execute(sql`
      SELECT 
        q.query,
        AVG(r.confidence) as avg_confidence,
        AVG(f.rating) as avg_rating,
        COUNT(*) as frequency
      FROM queries q
      JOIN responses r ON q.id = r.query_id
      LEFT JOIN feedback_data f ON r.id = f.response_id
      WHERE r.created_at > NOW() - INTERVAL '7 days'
      GROUP BY q.query
      HAVING AVG(r.confidence) < 0.7 OR AVG(f.rating) < 3
      ORDER BY frequency DESC
      LIMIT 20
    `);
    
    return opportunities.rows;
  }

  // Get recommended actions for improving responses
  async getImprovementRecommendations(queryPattern: string): Promise<any> {
    // Analyze similar queries with good feedback
    const embedding = await this.vectorService.generateEmbedding(queryPattern, 'search_query');
    
    const similarSuccessful = await db.execute(sql`
      SELECT 
        q.query,
        r.response,
        r.sources,
        f.rating,
        1 - (q.embedding <=> ${embedding}::vector(768)) as similarity
      FROM queries q
      JOIN responses r ON q.id = r.query_id
      JOIN feedback_data f ON r.id = f.response_id
      WHERE f.rating >= 4
        AND f.helpful = true
        AND f.accurate = true
      ORDER BY similarity DESC
      LIMIT 10
    `);
    
    // Extract patterns from successful responses
    const patterns = this.extractResponsePatterns(similarSuccessful.rows);
    
    return {
      queryPattern,
      recommendedPatterns: patterns,
      suggestedSources: this.extractTopSources(similarSuccessful.rows),
      confidence: this.calculateConfidence(patterns)
    };
  }

  // Extract response patterns for learning
  private extractResponsePatterns(responses: any[]): any[] {
    const patterns = [];
    
    for (const response of responses) {
      patterns.push({
        structure: this.analyzeStructure(response.response),
        sourceUsage: this.analyzeSourceUsage(response.sources),
        length: response.response.length,
        rating: response.rating
      });
    }
    
    return patterns;
  }

  // Calculate policy gradients for updates
  private async calculatePolicyGradients(interactions: any[]): Promise<any> {
    const gradients = {
      retrieval: {},
      generation: {},
      ranking: {}
    };
    
    for (const interaction of interactions) {
      const reward = (interaction.rating - 3) / 2; // Normalize
      
      // Calculate gradients for each component
      gradients.retrieval[interaction.strategy] = 
        (gradients.retrieval[interaction.strategy] || 0) + reward;
      
      gradients.generation[interaction.model] = 
        (gradients.generation[interaction.model] || 0) + reward;
      
      gradients.ranking[interaction.ranker] = 
        (gradients.ranking[interaction.ranker] || 0) + reward;
    }
    
    // Normalize gradients
    for (const key of Object.keys(gradients)) {
      const sum = Object.values(gradients[key]).reduce((a: number, b: number) => a + b, 0);
      for (const subkey of Object.keys(gradients[key])) {
        gradients[key][subkey] /= sum || 1;
      }
    }
    
    return gradients;
  }

  // Update retrieval weights based on learning
  private async updateRetrievalWeights(gradients: any): Promise<void> {
    for (const [strategy, gradient] of Object.entries(gradients.retrieval)) {
      await db.execute(sql`
        UPDATE retrieval_strategies
        SET weight = weight * (1 + ${gradient as number * this.config.learningRate})
        WHERE name = ${strategy}
      `);
    }
  }

  // Update generation parameters
  private async updateGenerationParameters(gradients: any): Promise<void> {
    for (const [param, gradient] of Object.entries(gradients.generation)) {
      await db.execute(sql`
        UPDATE generation_params
        SET value = value * (1 + ${gradient as number * this.config.learningRate})
        WHERE name = ${param}
      `);
    }
  }

  // Store feedback in database
  private async storeFeedback(feedback: FeedbackData, reward: number): Promise<void> {
    await db.execute(sql`
      INSERT INTO feedback_data (
        query_id,
        response_id,
        user_id,
        rating,
        helpful,
        accurate,
        complete,
        feedback,
        corrections,
        preferred_response,
        reward,
        created_at
      ) VALUES (
        ${feedback.queryId},
        ${feedback.responseId},
        ${feedback.userId},
        ${feedback.rating},
        ${feedback.helpful},
        ${feedback.accurate},
        ${feedback.complete},
        ${feedback.feedback},
        ${feedback.corrections},
        ${feedback.preferredResponse},
        ${reward},
        NOW()
      )
    `);
  }

  // Get Q-value for state-action pair
  private async getQValue(state: any, action: any): Promise<number> {
    const result = await db.execute(sql`
      SELECT q_value
      FROM q_values
      WHERE state_hash = ${this.hashState(state)}
        AND action_hash = ${this.hashAction(action)}
    `);
    
    return result.rows[0]?.q_value || 0;
  }

  // Store Q-value
  private async storeQValue(state: any, action: any, value: number): Promise<void> {
    await db.execute(sql`
      INSERT INTO q_values (state_hash, action_hash, q_value, state, action, updated_at)
      VALUES (
        ${this.hashState(state)},
        ${this.hashAction(action)},
        ${value},
        ${JSON.stringify(state)},
        ${JSON.stringify(action)},
        NOW()
      )
      ON CONFLICT (state_hash, action_hash)
      DO UPDATE SET 
        q_value = ${value},
        updated_at = NOW()
    `);
  }

  // Get maximum future Q-value
  private async getMaxFutureQValue(state: any): Promise<number> {
    const result = await db.execute(sql`
      SELECT MAX(q_value) as max_q
      FROM q_values
      WHERE state_hash LIKE ${this.hashState(state).substring(0, 8) + '%'}
    `);
    
    return result.rows[0]?.max_q || 0;
  }

  // Extract state features
  private extractStateFeatures(interaction: any): any {
    return {
      queryLength: interaction.messages[0]?.content.length || 0,
      queryType: this.classifyQuery(interaction.messages[0]?.content),
      userHistory: interaction.metadata?.userHistory || [],
      context: interaction.metadata?.context || {},
      timestamp: interaction.createdAt
    };
  }

  // Extract action features
  private extractActionFeatures(interaction: any): any {
    return {
      retrievalStrategy: interaction.metadata?.retrievalStrategy || 'hybrid',
      numSources: interaction.metadata?.sources?.length || 0,
      rerankingUsed: interaction.metadata?.reranking || false,
      generationModel: interaction.metadata?.model || 'default',
      temperature: interaction.metadata?.temperature || 0.7
    };
  }

  // Classify query type
  private classifyQuery(query: string): string {
    if (query.includes('?')) return 'question';
    if (query.toLowerCase().includes('how')) return 'howto';
    if (query.toLowerCase().includes('what')) return 'definition';
    if (query.toLowerCase().includes('why')) return 'explanation';
    return 'general';
  }

  // Hash state for storage
  private hashState(state: any): string {
    return require('crypto')
      .createHash('md5')
      .update(JSON.stringify(state))
      .digest('hex');
  }

  // Hash action for storage
  private hashAction(action: any): string {
    return require('crypto')
      .createHash('md5')
      .update(JSON.stringify(action))
      .digest('hex');
  }

  // Update metrics
  private async updateMetrics(feedback: FeedbackData): Promise<void> {
    const category = await this.getQueryCategory(feedback.queryId);
    const currentMetrics = this.metrics.get(category) || {
      accuracy: 0.5,
      helpfulness: 0.5,
      completeness: 0.5,
      userSatisfaction: 0.5,
      responseTime: 1000,
      sourcesQuality: 0.5
    };
    
    // Exponential moving average update
    const alpha = 0.1;
    currentMetrics.accuracy = (1 - alpha) * currentMetrics.accuracy + alpha * (feedback.accurate ? 1 : 0);
    currentMetrics.helpfulness = (1 - alpha) * currentMetrics.helpfulness + alpha * (feedback.helpful ? 1 : 0);
    currentMetrics.completeness = (1 - alpha) * currentMetrics.completeness + alpha * (feedback.complete ? 1 : 0);
    currentMetrics.userSatisfaction = (1 - alpha) * currentMetrics.userSatisfaction + alpha * (feedback.rating / 5);
    
    this.metrics.set(category, currentMetrics);
  }

  // Get query category
  private async getQueryCategory(queryId: string): Promise<string> {
    const result = await db.execute(sql`
      SELECT category
      FROM queries
      WHERE id = ${queryId}
    `);
    
    return result.rows[0]?.category || 'general';
  }

  // Analyze response structure
  private analyzeStructure(response: string): any {
    return {
      paragraphs: response.split('\n\n').length,
      sentences: response.split(/[.!?]/).length,
      hasIntro: response.startsWith('Based on') || response.startsWith('According to'),
      hasConclusion: response.includes('In summary') || response.includes('Therefore'),
      hasList: response.includes('•') || response.includes('-') || response.includes('1.')
    };
  }

  // Analyze source usage
  private analyzeSourceUsage(sources: any[]): any {
    return {
      count: sources.length,
      diversity: new Set(sources.map(s => s.source)).size,
      avgScore: sources.reduce((acc, s) => acc + s.score, 0) / sources.length,
      recency: sources.filter(s => Date.now() - new Date(s.timestamp).getTime() < 7 * 24 * 60 * 60 * 1000).length
    };
  }

  // Extract top sources
  private extractTopSources(responses: any[]): any[] {
    const sourceFrequency = new Map();
    
    for (const response of responses) {
      for (const source of response.sources || []) {
        const count = sourceFrequency.get(source.documentId) || 0;
        sourceFrequency.set(source.documentId, count + 1);
      }
    }
    
    return Array.from(sourceFrequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([id, count]) => ({ documentId: id, frequency: count }));
  }

  // Calculate confidence
  private calculateConfidence(patterns: any[]): number {
    if (patterns.length === 0) return 0;
    
    const avgRating = patterns.reduce((acc, p) => acc + p.rating, 0) / patterns.length;
    const consistency = 1 - this.calculateVariance(patterns.map(p => p.rating)) / 5;
    
    return (avgRating / 5) * consistency;
  }

  // Calculate variance
  private calculateVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
  }

  // Increase strategy weight
  private async increaseStrategyWeight(strategy: string): Promise<void> {
    await db.execute(sql`
      UPDATE retrieval_strategies
      SET weight = LEAST(weight * 1.1, 1.0)
      WHERE name = ${strategy}
    `);
  }

  // Decrease strategy weight
  private async decreaseStrategyWeight(strategy: string): Promise<void> {
    await db.execute(sql`
      UPDATE retrieval_strategies
      SET weight = GREATEST(weight * 0.9, 0.1)
      WHERE name = ${strategy}
    `);
  }

  // Get current performance metrics
  async getPerformanceMetrics(): Promise<any> {
    return {
      metrics: Object.fromEntries(this.metrics),
      rewardHistory: this.rewardHistory.slice(-100),
      avgReward: this.rewardHistory.reduce((a, b) => a + b, 0) / this.rewardHistory.length,
      policyVersion: this.policyUpdates,
      lastUpdate: new Date().toISOString()
    };
  }
}