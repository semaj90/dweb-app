/**
 * Recommendations Store - AI-Powered User Analytics & Suggestions
 * Integrates with NVIDIA go-llama and multi-core Ollama cluster
 */

import { writable, derived, readable } from "svelte/store";
import { productionServiceClient } from "$lib/services/production-service-client";

export interface Recommendation {
  id: string;
  type: 'case_action' | 'document_analysis' | 'evidence_review' | 'legal_precedent' | 'workflow_optimization';
  title: string;
  description: string;
  confidence: number; // 0-1
  priority: 'low' | 'medium' | 'high' | 'urgent';
  category: string;
  actionUrl?: string;
  metadata: {
    caseId?: string;
    documentId?: string;
    evidenceId?: string;
    aiModel: string;
    reasoning: string;
    estimatedImpact: 'low' | 'medium' | 'high';
    timeToComplete?: string;
  };
  createdAt: number;
  expiresAt?: number;
  dismissed?: boolean;
  accepted?: boolean;
}

export interface UserAnalytics {
  userId: string;
  profile: {
    userType: 'attorney' | 'paralegal' | 'investigator' | 'administrator';
    experienceLevel: 'junior' | 'mid' | 'senior' | 'expert';
    specializations: string[];
    workPatterns: {
      mostActiveHours: number[];
      averageSessionLength: number;
      documentsPerWeek: number;
      casesHandled: number;
    };
  };
  behavior: {
    searchPatterns: string[];
    documentTypes: string[];
    commonQueries: string[];
    toolUsage: Record<string, number>;
    navigationPaths: string[];
  };
  performance: {
    averageTaskTime: Record<string, number>;
    accuracyScores: Record<string, number>;
    productivityTrends: Array<{ date: string; score: number }>;
  };
  preferences: {
    aiAssistanceLevel: 'minimal' | 'moderate' | 'extensive';
    notificationFrequency: 'real-time' | 'hourly' | 'daily';
    recommendationTypes: string[];
  };
}

export interface RecommendationState {
  // Recommendations
  recommendations: Recommendation[];
  activeRecommendations: Recommendation[];
  dismissedRecommendations: Recommendation[];

  // User Analytics
  userAnalytics: UserAnalytics | null;
  behaviorInsights: {
    patterns: string[];
    suggestions: string[];
    trends: Array<{ metric: string; trend: 'up' | 'down' | 'stable'; change: number }>;
  };

  // AI Models
  isAnalyzing: boolean;
  lastAnalysisTime: number | null;
  aiModelsStatus: {
    nvidia_llama: boolean;
    gemma3_legal: boolean;
    recommendation_engine: boolean;
  };

  // Performance
  analyticsLatency: number;
  recommendationAccuracy: number; // User feedback based

  // Settings
  enableRealTimeAnalysis: boolean;
  privacyLevel: 'minimal' | 'standard' | 'enhanced';

  error: string | null;
}

const initialState: RecommendationState = {
  recommendations: [],
  activeRecommendations: [],
  dismissedRecommendations: [],
  userAnalytics: null,
  behaviorInsights: {
    patterns: [],
    suggestions: [],
    trends: []
  },
  isAnalyzing: false,
  lastAnalysisTime: null,
  aiModelsStatus: {
    nvidia_llama: false,
    gemma3_legal: false,
    recommendation_engine: false
  },
  analyticsLatency: 0,
  recommendationAccuracy: 0,
  enableRealTimeAnalysis: true,
  privacyLevel: 'standard',
  error: null
};

// Core store
export const recommendationStore = writable<RecommendationState>(initialState);

// Derived stores (repaired syntax)
export const highPriorityRecommendations = derived(recommendationStore, ($store) =>
  $store.activeRecommendations
    .filter(r => r.priority === 'high' || r.priority === 'urgent')
    .sort((a, b) => b.confidence - a.confidence)
);

export const recommendationsByType = derived(recommendationStore, ($store) => {
  const grouped: Record<string, Recommendation[]> = {};
  $store.activeRecommendations.forEach(rec => {
    if (!grouped[rec.type]) grouped[rec.type] = [];
    grouped[rec.type].push(rec);
  });
  return grouped;
});

export const userProductivityScore = derived(recommendationStore, ($store) => {
  if (!$store.userAnalytics?.performance) return 0;
  const trends = $store.userAnalytics.performance.productivityTrends;
  if (trends.length === 0) return 0;
  return trends[trends.length - 1].score;
});

// Actions
export const recommendationActions = {
  /**
   * Generate AI-powered recommendations based on user behavior
   */
  async generateRecommendations(userId: string, context?: {
    caseId?: string;
    currentTask?: string;
    recentActivity?: string[];
  }): Promise<void> {
    recommendationStore.update(state => ({
      ...state,
      isAnalyzing: true,
      error: null
    }));

    const startTime = Date.now();

    try {
      const response = await productionServiceClient.execute('ai.recommendations', {
        userId,
        context,
        options: {
          model: 'nvidia-llama',
          analysisDepth: 'comprehensive',
          includeUserAnalytics: true,
          maxRecommendations: 10
        }
      });

      const latency = Date.now() - startTime;

      recommendationStore.update(state => ({
        ...state,
        recommendations: response.recommendations || [],
        activeRecommendations: (response.recommendations || []).filter((r: Recommendation) => !r.dismissed),
        behaviorInsights: response.insights || state.behaviorInsights,
        analyticsLatency: latency,
        lastAnalysisTime: Date.now(),
        isAnalyzing: false
      }));

    } catch (error: any) {
      console.error('Recommendation generation failed:', error);
      recommendationStore.update(state => ({
        ...state,
        isAnalyzing: false,
        error: error instanceof Error ? error.message : 'Failed to generate recommendations'
      }));
    }
  },

  /**
   * Analyze user behavior and update analytics
   */
  async analyzeUserBehavior(userId: string, activityData: {
    action: string;
    context: any;
    timestamp: number;
    duration?: number;
  }): Promise<void> {
    if (!initialState.enableRealTimeAnalysis) return;

    try {
      const response = await productionServiceClient.execute('analytics.behavior', {
        userId,
        activity: activityData,
        options: {
          updateProfile: true,
          generateInsights: true
        }
      });

      recommendationStore.update(state => ({
        ...state,
        userAnalytics: response.userAnalytics || state.userAnalytics,
        behaviorInsights: response.insights || state.behaviorInsights
      }));

    } catch (error: any) {
      console.error('Behavior analysis failed:', error);
    }
  },

  /**
   * Accept a recommendation and provide feedback
   */
  async acceptRecommendation(recommendationId: string, feedback?: {
    helpful: boolean;
    implemented: boolean;
    notes?: string;
  }): Promise<void> {
    try {
      await productionServiceClient.execute('recommendations.feedback', {
        recommendationId,
        action: 'accept',
        feedback
      });

      recommendationStore.update(state => ({
        ...state,
        recommendations: state.recommendations.map(r =>
          r.id === recommendationId ? { ...r, accepted: true } : r
        ),
        activeRecommendations: state.activeRecommendations.filter(r => r.id !== recommendationId)
      }));

    } catch (error: any) {
      console.error('Failed to accept recommendation:', error);
    }
  },

  /**
   * Dismiss a recommendation
   */
  async dismissRecommendation(recommendationId: string, reason?: string): Promise<void> {
    try {
      await productionServiceClient.execute('recommendations.feedback', {
        recommendationId,
        action: 'dismiss',
        reason
      });

      recommendationStore.update(state => {
        const dismissedRec = state.activeRecommendations.find(r => r.id === recommendationId);
        return {
          ...state,
          recommendations: state.recommendations.map(r =>
            r.id === recommendationId ? { ...r, dismissed: true } : r
          ),
          activeRecommendations: state.activeRecommendations.filter(r => r.id !== recommendationId),
          dismissedRecommendations: dismissedRec ? [...state.dismissedRecommendations, dismissedRec] : state.dismissedRecommendations
        };
      });

    } catch (error: any) {
      console.error('Failed to dismiss recommendation:', error);
    }
  },

  /**
   * Get user analytics and performance insights
   */
  async loadUserAnalytics(userId: string): Promise<void> {
    try {
      const response = await productionServiceClient.execute('analytics.user', {
        userId,
        includePerformance: true,
        includeBehavior: true,
        timeRange: '30d'
      });

      recommendationStore.update(state => ({
        ...state,
        userAnalytics: response.analytics,
        behaviorInsights: response.insights || state.behaviorInsights
      }));

    } catch (error: any) {
      console.error('Failed to load user analytics:', error);
      recommendationStore.update(state => ({
        ...state,
        error: error instanceof Error ? error.message : 'Failed to load analytics'
      }));
    }
  },

  /**
   * Track recommendation accuracy based on user feedback
   */
  updateAccuracyMetrics(feedback: Array<{ helpful: boolean; confidence: number }>): void {
    if (feedback.length === 0) return;

    const accuracy = feedback.reduce((sum, f) =>
      sum + (f.helpful ? f.confidence : (1 - f.confidence)), 0
    ) / feedback.length;

    recommendationStore.update(state => ({
      ...state,
      recommendationAccuracy: accuracy
    }));
  },

  /**
   * Update recommendation settings
   */
  updateSettings(settings: Partial<{
    enableRealTimeAnalysis: boolean;
    privacyLevel: 'minimal' | 'standard' | 'enhanced';
  }>): void {
    recommendationStore.update(state => ({
      ...state,
      ...settings
    }));
  },

  /**
   * Check AI models status
   */
  async checkModelsStatus(): Promise<void> {
    try {
      const response = await productionServiceClient.execute('ai.models.status', {});

      recommendationStore.update(state => ({
        ...state,
        aiModelsStatus: {
          nvidia_llama: response.nvidia_llama || false,
          gemma3_legal: response.gemma3_legal || false,
          recommendation_engine: response.recommendation_engine || false
        }
      }));

    } catch (error: any) {
      console.error('Failed to check models status:', error);
    }
  },

  /**
   * Clear all recommendations
   */
  clearRecommendations(): void {
    recommendationStore.update(state => ({
      ...state,
      recommendations: [],
      activeRecommendations: [],
      error: null
    }));
  }
};

// Auto-initialize
if (typeof window !== 'undefined') {
  recommendationActions.checkModelsStatus();
}