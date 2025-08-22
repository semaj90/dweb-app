
/**
 * Advanced User Analytics & History Microservice
 * Tracks user behavior, AI interactions, and generates insights for recommendation engine
 */

import { EventEmitter } from 'events';
import { CacheManager } from '../cache/loki-cache';
import { Queue } from 'bull';
import Redis from 'ioredis';

interface UserInteraction {
  id: string;
  userId: string;
  sessionId: string;
  timestamp: number;
  interactionType: 
    | 'document-upload'
    | 'ai-analysis'
    | 'search-query'
    | 'recommendation-click'
    | 'custom-prompt'
    | 'document-comparison'
    | 'export-results'
    | 'session-start'
    | 'session-end';
  data: {
    documentId?: string;
    analysisType?: string;
    queryText?: string;
    confidence?: number;
    processingTime?: number;
    modelUsed?: string;
    resultSatisfaction?: number; // 1-5 rating
    tags?: string[];
    metadata?: unknown;
  };
  context: {
    userAgent?: string;
    ip?: string;
    referrer?: string;
    screenResolution?: string;
    language?: string;
  };
}

interface UserProfile {
  userId: string;
  demographics: {
    role: 'prosecutor' | 'detective' | 'legal-assistant' | 'admin' | 'user';
    department?: string;
    experienceLevel: 'beginner' | 'intermediate' | 'expert';
    specializations: string[];
  };
  preferences: {
    analysisDepth: 'quick' | 'standard' | 'detailed';
    outputFormat: 'structured' | 'narrative' | 'bullet-points';
    modelPreference: 'gemma3-legal' | 'legal-bert' | 'auto';
    notificationSettings: {
      email: boolean;
      inApp: boolean;
      frequency: 'immediate' | 'hourly' | 'daily';
    };
  };
  usage: {
    totalSessions: number;
    totalInteractions: number;
    averageSessionDuration: number;
    mostUsedFeatures: string[];
    totalProcessingTime: number;
    totalDocumentsProcessed: number;
    favoriteAnalysisTypes: string[];
  };
  behavioral: {
    peakUsageHours: number[];
    typicalSessionLength: number;
    queryComplexity: 'simple' | 'moderate' | 'complex';
    satisfactionScore: number;
    retentionRate: number;
    learningCurve: number; // How quickly they adopt new features
  };
  lastUpdated: number;
}

interface UserSession {
  sessionId: string;
  userId: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  interactions: UserInteraction[];
  documents: string[];
  aiCalls: {
    model: string;
    count: number;
    totalTime: number;
    averageConfidence: number;
  }[];
  outcomes: {
    successfulAnalyses: number;
    failedAnalyses: number;
    exportedResults: number;
    satisfactionRating?: number;
  };
  deviceInfo: {
    userAgent: string;
    screenSize: string;
    platform: string;
    browser: string;
  };
}

export class AnalyticsService extends EventEmitter {
  private cache: CacheManager;
  private redis: Redis;
  private analyticsQueue: Queue;
  private profiles: Map<string, UserProfile> = new Map();
  private activeSessions: Map<string, UserSession> = new Map();
  
  constructor(options: {
    redisUrl?: string;
    queueName?: string;
  } = {}) {
    super();
    
    this.cache = new CacheManager();
    this.redis = new Redis(options.redisUrl || 'redis://localhost:6379');
    this.analyticsQueue = new Queue('analytics processing', {
      redis: options.redisUrl || 'redis://localhost:6379'
    });
    
    this.setupQueueProcessors();
    this.setupPeriodicTasks();
    
    console.log('📊 Analytics Service initialized');
  }

  private setupQueueProcessors() {
    // Process interaction data in background
    this.analyticsQueue.process('process-interaction', async (job) => {
      const interaction: UserInteraction = job.data;
      await this.processUserInteraction(interaction);
      return { processed: true, interactionId: interaction.id };
    });

    // Generate user insights
    this.analyticsQueue.process('generate-insights', async (job) => {
      const { userId } = job.data;
      return await this.generateUserInsights(userId);
    });

    // Update recommendation models
    this.analyticsQueue.process('update-recommendations', async (job) => {
      const { userId, sessionData } = job.data;
      return await this.updateRecommendationModel(userId, sessionData);
    });
  }

  private setupPeriodicTasks() {
    // Update user profiles every 5 minutes
    setInterval(async () => {
      await this.updateAllUserProfiles();
    }, 5 * 60 * 1000);

    // Generate daily analytics reports
    setInterval(async () => {
      await this.generateDailyReports();
    }, 24 * 60 * 60 * 1000);

    // Cleanup old data weekly
    setInterval(async () => {
      await this.cleanupOldData();
    }, 7 * 24 * 60 * 60 * 1000);
  }

  // Core interaction tracking
  public async trackInteraction(interaction: Omit<UserInteraction, 'id' | 'timestamp'>): Promise<void> {
    const fullInteraction: UserInteraction = {
      id: this.generateId(),
      timestamp: Date.now(),
      ...interaction
    };

    // Queue for background processing
    await this.analyticsQueue.add('process-interaction', fullInteraction);
    
    // Immediately update active session
    await this.updateActiveSession(fullInteraction);
    
    console.log(`📝 Tracked interaction: ${fullInteraction.interactionType} for user ${fullInteraction.userId}`);
  }

  public async trackStreamingSession(
    userId: string, 
    sessionData: {
      sessionId: string;
      documentId?: string;
      analysisType: string;
      startTime: number;
    }
  ): Promise<void> {
    const session: UserSession = {
      sessionId: sessionData.sessionId,
      userId,
      startTime: sessionData.startTime,
      interactions: [],
      documents: sessionData.documentId ? [sessionData.documentId] : [],
      aiCalls: [],
      outcomes: {
        successfulAnalyses: 0,
        failedAnalyses: 0,
        exportedResults: 0
      },
      deviceInfo: {
        userAgent: 'Unknown',
        screenSize: 'Unknown',
        platform: 'Unknown',
        browser: 'Unknown'
      }
    };

    this.activeSessions.set(sessionData.sessionId, session);
    
    // Track session start
    await this.trackInteraction({
      userId,
      sessionId: sessionData.sessionId,
      interactionType: 'session-start',
      data: {
        analysisType: sessionData.analysisType,
        documentId: sessionData.documentId
      },
      context: {}
    });
  }

  private async processUserInteraction(interaction: UserInteraction): Promise<void> {
    // Store in cache
    await this.cache.logAnalytics(
      interaction.userId,
      interaction.interactionType,
      interaction
    );

    // Update user profile
    await this.updateUserProfile(interaction);

    // Trigger recommendation updates if significant interaction
    if (this.isSignificantInteraction(interaction)) {
      await this.analyticsQueue.add('update-recommendations', {
        userId: interaction.userId,
        sessionData: this.activeSessions.get(interaction.sessionId)
      });
    }

    this.emit('interaction-processed', interaction);
  }

  private async updateActiveSession(interaction: UserInteraction): Promise<void> {
    const session = this.activeSessions.get(interaction.sessionId);
    if (!session) return;

    session.interactions.push(interaction);

    // Update session statistics based on interaction type
    switch (interaction.interactionType) {
      case 'ai-analysis':
        session.aiCalls.push({
          model: interaction.data.modelUsed || 'unknown',
          count: 1,
          totalTime: interaction.data.processingTime || 0,
          averageConfidence: interaction.data.confidence || 0
        });
        
        if (interaction.data.confidence && interaction.data.confidence > 0.7) {
          session.outcomes.successfulAnalyses++;
        } else {
          session.outcomes.failedAnalyses++;
        }
        break;

      case 'export-results':
        session.outcomes.exportedResults++;
        break;

      case 'session-end':
        session.endTime = Date.now();
        session.duration = session.endTime - session.startTime;
        await this.finalizeSession(session);
        this.activeSessions.delete(interaction.sessionId);
        break;
    }
  }

  private async updateUserProfile(interaction: UserInteraction): Promise<void> {
    let profile = this.profiles.get(interaction.userId) || await this.getUserProfile(interaction.userId);
    
    if (!profile) {
      profile = this.createDefaultProfile(interaction.userId);
    }

    // Update usage statistics
    profile.usage.totalInteractions++;
    profile.usage.totalProcessingTime += interaction.data.processingTime || 0;
    
    if (interaction.data.documentId) {
      profile.usage.totalDocumentsProcessed++;
    }

    // Update behavioral patterns
    const hour = new Date(interaction.timestamp).getHours();
    if (!profile.behavioral.peakUsageHours.includes(hour)) {
      profile.behavioral.peakUsageHours.push(hour);
    }

    // Update preferences based on interaction patterns
    if (interaction.data.analysisType) {
      const currentFavorites = profile.usage.favoriteAnalysisTypes;
      const index = currentFavorites.indexOf(interaction.data.analysisType);
      
      if (index === -1) {
        currentFavorites.push(interaction.data.analysisType);
      } else {
        // Move to front (most recent)
        currentFavorites.splice(index, 1);
        currentFavorites.unshift(interaction.data.analysisType);
      }
      
      // Keep only top 5
      profile.usage.favoriteAnalysisTypes = currentFavorites.slice(0, 5);
    }

    // Update satisfaction score if available
    if (interaction.data.resultSatisfaction) {
      const currentScore = profile.behavioral.satisfactionScore || 0;
      profile.behavioral.satisfactionScore = (currentScore + interaction.data.resultSatisfaction) / 2;
    }

    profile.lastUpdated = Date.now();
    this.profiles.set(interaction.userId, profile);
    
    // Persist to cache
    await this.cache.setUserSession(interaction.userId, profile);
  }

  // User profile management
  public async getUserProfile(userId: string): Promise<UserProfile | null> {
    // Check memory first
    if (this.profiles.has(userId)) {
      return this.profiles.get(userId)!;
    }

    // Check cache
    const cached = await this.cache.getUserSession(userId);
    if (cached && cached.userId) {
      this.profiles.set(userId, cached);
      return cached;
    }

    return null;
  }

  private createDefaultProfile(userId: string): UserProfile {
    return {
      userId,
      demographics: {
        role: 'user',
        experienceLevel: 'beginner',
        specializations: []
      },
      preferences: {
        analysisDepth: 'standard',
        outputFormat: 'structured',
        modelPreference: 'auto',
        notificationSettings: {
          email: true,
          inApp: true,
          frequency: 'immediate'
        }
      },
      usage: {
        totalSessions: 0,
        totalInteractions: 0,
        averageSessionDuration: 0,
        mostUsedFeatures: [],
        totalProcessingTime: 0,
        totalDocumentsProcessed: 0,
        favoriteAnalysisTypes: []
      },
      behavioral: {
        peakUsageHours: [],
        typicalSessionLength: 0,
        queryComplexity: 'simple',
        satisfactionScore: 3.0,
        retentionRate: 0,
        learningCurve: 0
      },
      lastUpdated: Date.now()
    };
  }

  // Advanced analytics
  public async generateUserInsights(userId: string): Promise<{
    profile: UserProfile;
    trends: unknown;
    recommendations: unknown;
    predictions: unknown;
  }> {
    const profile = await this.getUserProfile(userId);
    if (!profile) throw new Error('User profile not found');

    // Get interaction history
    const interactions = await this.cache.getAnalytics(userId);
    
    // Analyze trends
    const trends = await this.analyzeTrends(interactions);
    
    // Generate recommendations
    const recommendations = await this.generatePersonalizedRecommendations(profile, interactions);
    
    // Make predictions
    const predictions = await this.generatePredictions(profile, interactions);

    return {
      profile,
      trends,
      recommendations,
      predictions
    };
  }

  private async analyzeTrends(interactions: unknown[]): Promise<any> {
    const last30Days = Date.now() - (30 * 24 * 60 * 60 * 1000);
    const recentInteractions = interactions.filter(i => i.timestamp > last30Days);

    return {
      dailyUsage: this.calculateDailyUsage(recentInteractions),
      featureAdoption: this.calculateFeatureAdoption(recentInteractions),
      satisfactionTrend: this.calculateSatisfactionTrend(recentInteractions),
      performanceTrend: this.calculatePerformanceTrend(recentInteractions)
    };
  }

  private async generatePersonalizedRecommendations(
    profile: UserProfile, 
    interactions: unknown[]
  ): Promise<any> {
    const recommendations = [];

    // Feature recommendations based on usage patterns
    if (profile.usage.totalInteractions > 50 && !profile.usage.mostUsedFeatures.includes('custom-analysis')) {
      recommendations.push({
        type: 'feature',
        title: 'Try Custom Analysis',
        description: 'Based on your usage pattern, you might benefit from custom analysis prompts',
        priority: 'medium'
      });
    }

    // Model recommendations
    if (profile.behavioral.satisfactionScore < 3.5) {
      recommendations.push({
        type: 'model',
        title: 'Try Different AI Model',
        description: 'Consider switching to a different AI model for better results',
        priority: 'high'
      });
    }

    // Workflow recommendations
    if (profile.usage.averageSessionDuration > 30 * 60 * 1000) { // 30 minutes
      recommendations.push({
        type: 'workflow',
        title: 'Optimize Your Workflow',
        description: 'Your sessions are quite long. Consider using batch processing for multiple documents',
        priority: 'medium'
      });
    }

    return recommendations;
  }

  private async generatePredictions(profile: UserProfile, interactions: unknown[]): Promise<any> {
    return {
      likelyNextAction: this.predictNextAction(interactions),
      churRisk: this.calculateChurnRisk(profile, interactions),
      valueScore: this.calculateUserValue(profile, interactions),
      growthPotential: this.calculateGrowthPotential(profile, interactions)
    };
  }

  // Utility methods for calculations
  private calculateDailyUsage(interactions: unknown[]): unknown {
    const usage = new Map<string, number>();
    
    interactions.forEach(interaction => {
      const date = new Date(interaction.timestamp).toISOString().split('T')[0];
      usage.set(date, (usage.get(date) || 0) + 1);
    });

    return Array.from(usage.entries()).map(([date, count]) => ({ date, count }));
  }

  private calculateFeatureAdoption(interactions: unknown[]): unknown {
    const features = new Map<string, number>();
    
    interactions.forEach(interaction => {
      const feature = interaction.data.interactionType;
      features.set(feature, (features.get(feature) || 0) + 1);
    });

    return Array.from(features.entries()).map(([feature, count]) => ({ feature, count }));
  }

  private calculateSatisfactionTrend(interactions: unknown[]): unknown {
    return interactions
      .filter(i => i.data.data?.resultSatisfaction)
      .map(i => ({
        timestamp: i.timestamp,
        satisfaction: i.data.data.resultSatisfaction
      }));
  }

  private calculatePerformanceTrend(interactions: unknown[]): unknown {
    return interactions
      .filter(i => i.data.data?.processingTime)
      .map(i => ({
        timestamp: i.timestamp,
        processingTime: i.data.data.processingTime
      }));
  }

  private predictNextAction(interactions: unknown[]): string {
    // Simple prediction based on recent patterns
    const recentActions = interactions
      .slice(-10)
      .map(i => i.data.interactionType);
    
    const actionCounts = new Map<string, number>();
    recentActions.forEach(action => {
      actionCounts.set(action, (actionCounts.get(action) || 0) + 1);
    });

    const mostCommon = Array.from(actionCounts.entries())
      .sort(([,a], [,b]) => b - a)[0];

    return mostCommon ? mostCommon[0] : 'ai-analysis';
  }

  private calculateChurnRisk(profile: UserProfile, interactions: unknown[]): number {
    const daysSinceLastInteraction = (Date.now() - profile.lastUpdated) / (24 * 60 * 60 * 1000);
    const satisfactionScore = profile.behavioral.satisfactionScore;
    const usageFrequency = interactions.length / Math.max(1, daysSinceLastInteraction);

    // Simple churn risk calculation (0-1 scale)
    let risk = 0;
    
    if (daysSinceLastInteraction > 7) risk += 0.3;
    if (satisfactionScore < 3.0) risk += 0.3;
    if (usageFrequency < 0.5) risk += 0.4;

    return Math.min(risk, 1.0);
  }

  private calculateUserValue(profile: UserProfile, interactions: unknown[]): number {
    // Value score based on usage, satisfaction, and engagement
    const usageScore = Math.min(profile.usage.totalInteractions / 100, 1.0);
    const satisfactionScore = profile.behavioral.satisfactionScore / 5.0;
    const engagementScore = Math.min(interactions.length / 50, 1.0);

    return (usageScore + satisfactionScore + engagementScore) / 3;
  }

  private calculateGrowthPotential(profile: UserProfile, interactions: unknown[]): number {
    // Growth potential based on learning curve and feature adoption
    const featureAdoption = profile.usage.mostUsedFeatures.length / 10; // Assume 10 total features
    const learningCurve = profile.behavioral.learningCurve;
    const sessionGrowth = profile.usage.totalSessions > 10 ? 1.0 : profile.usage.totalSessions / 10;

    return (featureAdoption + learningCurve + sessionGrowth) / 3;
  }

  // Session management
  private async finalizeSession(session: UserSession): Promise<void> {
    // Update user profile with session data
    const profile = await this.getUserProfile(session.userId) || this.createDefaultProfile(session.userId);
    
    profile.usage.totalSessions++;
    profile.usage.averageSessionDuration = 
      (profile.usage.averageSessionDuration + (session.duration || 0)) / 2;

    // Store session data
    await this.cache.logAnalytics(session.userId, 'session-complete', session);
    
    console.log(`✅ Session finalized: ${session.sessionId} (${session.duration}ms)`);
  }

  // Data management
  public async getUserHistory(userId: string, limit: number = 100): Promise<unknown[]> {
    return await this.cache.getAnalytics(userId, undefined, undefined);
  }

  private async updateAllUserProfiles(): Promise<void> {
    console.log('🔄 Updating all user profiles...');
    // Implementation for bulk profile updates
  }

  private async generateDailyReports(): Promise<void> {
    console.log('📊 Generating daily analytics reports...');
    // Implementation for daily reporting
  }

  private async cleanupOldData(): Promise<void> {
    console.log('🧹 Cleaning up old analytics data...');
    // Implementation for data cleanup
  }

  // Helper methods
  private isSignificantInteraction(interaction: UserInteraction): boolean {
    return [
      'ai-analysis',
      'document-comparison',
      'export-results',
      'session-end'
    ].includes(interaction.interactionType);
  }

  private generateId(): string {
    return `analytics_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Public API methods
  public async getSystemStats(): Promise<any> {
    const totalUsers = this.profiles.size;
    const activeSessions = this.activeSessions.size;
    const queueStats = await this.analyticsQueue.getWaiting();

    return {
      totalUsers,
      activeSessions,
      queueSize: queueStats.length,
      cacheStats: this.cache.getStats()
    };
  }

  public async shutdown(): Promise<void> {
    console.log('🔄 Shutting down Analytics Service...');
    
    // Finalize all active sessions
    for (const [sessionId, session] of this.activeSessions) {
      session.endTime = Date.now();
      session.duration = session.endTime! - session.startTime;
      await this.finalizeSession(session);
    }
    
    // Close queue and redis connections
    await this.analyticsQueue.close();
    this.redis.disconnect();
    
    console.log('✅ Analytics Service shutdown complete');
  }
}