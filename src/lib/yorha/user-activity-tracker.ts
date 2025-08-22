/**
 * YoRHa Interface - User Activity Tracker
 * Gaming-style UI with comprehensive user behavior analytics
 * Integrated with legal AI workflow patterns
 */

import { writable, derived, type Readable } from 'svelte/store';
import { browser } from '$app/environment';

export interface UserProfile {
  id: string;
  username: string;
  role: 'prosecutor' | 'detective' | 'paralegal' | 'attorney' | 'admin';
  experience: 'junior' | 'mid' | 'senior' | 'expert';
  jurisdiction: string;
  specialization: string[];
  avatar: {
    type: 'android' | 'human' | 'ai';
    appearance: string;
    level: number;
    achievements: string[];
  };
  preferences: {
    theme: 'dark' | 'light' | 'yorha';
    aiAssistance: 'minimal' | 'standard' | 'enhanced';
    notifications: boolean;
    autoSave: boolean;
    keyboardShortcuts: boolean;
  };
  stats: UserStats;
  activity: UserActivity[];
  sessionData: SessionData;
}

export interface UserStats {
  totalSessions: number;
  totalTimeSpent: number; // minutes
  documentsProcessed: number;
  casesWorked: number;
  aiInteractions: number;
  searchQueries: number;
  uploadsCompleted: number;
  accuracy: number; // 0-100
  efficiency: number; // 0-100
  streak: number; // consecutive days
  level: number;
  experience: number;
  nextLevelXP: number;
}

export interface UserActivity {
  id: string;
  type: 'login' | 'logout' | 'document_upload' | 'case_created' | 'ai_query' | 'search' | 'analysis_completed';
  timestamp: Date;
  metadata: Record<string, any>;
  entityId?: string;
  entityType?: string;
  duration?: number; // milliseconds
  outcome: 'success' | 'error' | 'pending';
  context: {
    sessionId: string;
    userAgent: string;
    ipAddress?: string;
    location?: string;
  };
}

export interface SessionData {
  id: string;
  startTime: Date;
  lastActivity: Date;
  isActive: boolean;
  totalActions: number;
  currentWorkspace: string;
  openCases: string[];
  recentDocuments: string[];
  aiChatHistory: Array<{
    query: string;
    response: string;
    timestamp: Date;
    intent: string;
  }>;
  performance: {
    responseTimes: number[];
    errorCount: number;
    taskCompletion: number;
  };
}

// Stores for user profile and activity
export const userProfile = writable<UserProfile | null>(null);
export const currentSession = writable<SessionData | null>(null);
export const activityFeed = writable<UserActivity[]>([]);
export const userMetrics = writable({
  dailyGoals: {
    documentsProcessed: 0,
    targetDocuments: 10,
    casesReviewed: 0,
    targetCases: 3,
    aiInteractions: 0,
    targetInteractions: 20
  },
  weeklyProgress: {
    efficiency: 0,
    accuracy: 0,
    productivity: 0
  },
  achievements: [] as string[]
});

// Derived stores for dashboard widgets
export const userDashboard = derived(
  [userProfile, currentSession, activityFeed],
  ([$profile, $session, $activities]) => {
    if (!$profile || !$session) return null;

    const todayActivities = $activities.filter(
      activity => activity.timestamp >= getStartOfDay()
    );

    return {
      profile: $profile,
      session: $session,
      todayStats: {
        activitiesCount: todayActivities.length,
        documentsProcessed: todayActivities.filter(a => a.type === 'document_upload').length,
        aiQueries: todayActivities.filter(a => a.type === 'ai_query').length,
        timeSpent: $session.lastActivity.getTime() - $session.startTime.getTime(),
        efficiency: calculateEfficiency(todayActivities)
      },
      recentActivity: $activities.slice(0, 10),
      performanceMetrics: calculatePerformanceMetrics($activities, $session)
    };
  }
);

export class YoRHaUserTracker {
  private profileStorage = 'yorha_user_profile';
  private sessionStorage = 'yorha_current_session';
  private activityStorage = 'yorha_activity_feed';
  
  private initialized = false;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private performanceObserver: PerformanceObserver | null = null;

  async initialize(userId: string): Promise<void> {
    if (this.initialized) return;

    try {
      console.log('🤖 Initializing YoRHa User Tracker...');

      // Load or create user profile
      await this.loadOrCreateProfile(userId);
      
      // Start new session
      await this.startSession();
      
      // Load activity history
      this.loadActivityHistory();
      
      // Set up performance monitoring
      this.setupPerformanceMonitoring();
      
      // Start heartbeat for session tracking
      this.startHeartbeat();
      
      this.initialized = true;
      console.log('✅ YoRHa User Tracker initialized');

    } catch (error) {
      console.error('❌ YoRHa User Tracker initialization failed:', error);
      throw error;
    }
  }

  private async loadOrCreateProfile(userId: string): Promise<void> {
    let profile: UserProfile;
    
    if (browser) {
      const stored = localStorage.getItem(this.profileStorage);
      if (stored) {
        profile = JSON.parse(stored);
      } else {
        profile = this.createDefaultProfile(userId);
      }
    } else {
      profile = this.createDefaultProfile(userId);
    }

    userProfile.set(profile);
  }

  private createDefaultProfile(userId: string): UserProfile {
    return {
      id: userId,
      username: `User_${userId.slice(0, 8)}`,
      role: 'attorney',
      experience: 'mid',
      jurisdiction: 'federal',
      specialization: ['general_practice'],
      avatar: {
        type: 'android',
        appearance: 'default',
        level: 1,
        achievements: ['new_unit']
      },
      preferences: {
        theme: 'yorha',
        aiAssistance: 'standard',
        notifications: true,
        autoSave: true,
        keyboardShortcuts: true
      },
      stats: {
        totalSessions: 0,
        totalTimeSpent: 0,
        documentsProcessed: 0,
        casesWorked: 0,
        aiInteractions: 0,
        searchQueries: 0,
        uploadsCompleted: 0,
        accuracy: 75,
        efficiency: 60,
        streak: 0,
        level: 1,
        experience: 0,
        nextLevelXP: 1000
      },
      activity: [],
      sessionData: this.createSessionData()
    };
  }

  private createSessionData(): SessionData {
    return {
      id: crypto.randomUUID(),
      startTime: new Date(),
      lastActivity: new Date(),
      isActive: true,
      totalActions: 0,
      currentWorkspace: 'main',
      openCases: [],
      recentDocuments: [],
      aiChatHistory: [],
      performance: {
        responseTimes: [],
        errorCount: 0,
        taskCompletion: 0
      }
    };
  }

  private async startSession(): Promise<void> {
    const sessionData = this.createSessionData();
    currentSession.set(sessionData);

    // Track session start
    await this.trackActivity({
      type: 'login',
      metadata: {
        sessionStart: sessionData.startTime.toISOString(),
        browser: navigator?.userAgent || 'unknown'
      },
      outcome: 'success'
    });

    // Update profile stats
    userProfile.update(profile => {
      if (!profile) return profile;
      return {
        ...profile,
        stats: {
          ...profile.stats,
          totalSessions: profile.stats.totalSessions + 1
        },
        sessionData
      };
    });
  }

  private loadActivityHistory(): void {
    if (!browser) return;

    try {
      const stored = localStorage.getItem(this.activityStorage);
      if (stored) {
        const activities = JSON.parse(stored);
        activityFeed.set(activities);
      }
    } catch (error) {
      console.warn('Failed to load activity history:', error);
    }
  }

  private setupPerformanceMonitoring(): void {
    if (!browser || !('PerformanceObserver' in window)) return;

    try {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        
        entries.forEach(entry => {
          if (entry.entryType === 'navigation') {
            this.updateSessionPerformance('navigation', entry.duration);
          } else if (entry.entryType === 'resource') {
            this.updateSessionPerformance('resource', entry.duration);
          }
        });
      });

      this.performanceObserver.observe({ 
        entryTypes: ['navigation', 'resource', 'measure'] 
      });

    } catch (error) {
      console.warn('Performance monitoring setup failed:', error);
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      currentSession.update(session => {
        if (!session) return session;
        
        return {
          ...session,
          lastActivity: new Date()
        };
      });
      
      this.persistSession();
    }, 30000); // Every 30 seconds
  }

  // Public methods for tracking user activities
  async trackActivity(activity: Partial<UserActivity>): Promise<void> {
    const fullActivity: UserActivity = {
      id: crypto.randomUUID(),
      type: activity.type || 'search',
      timestamp: new Date(),
      metadata: activity.metadata || {},
      entityId: activity.entityId,
      entityType: activity.entityType,
      duration: activity.duration,
      outcome: activity.outcome || 'success',
      context: {
        sessionId: this.getCurrentSessionId(),
        userAgent: navigator?.userAgent || 'unknown',
        ipAddress: activity.context?.ipAddress,
        location: activity.context?.location
      }
    };

    // Add to activity feed
    activityFeed.update(activities => [fullActivity, ...activities.slice(0, 999)]);

    // Update session stats
    currentSession.update(session => {
      if (!session) return session;
      
      return {
        ...session,
        totalActions: session.totalActions + 1,
        lastActivity: new Date()
      };
    });

    // Check for achievements
    this.checkAchievements(fullActivity);

    // Persist data
    this.persistActivity();
    this.persistSession();

    // Send to backend if available
    await this.syncToBackend(fullActivity);
  }

  async trackDocumentUpload(documentId: string, metadata: unknown): Promise<void> {
    await this.trackActivity({
      type: 'document_upload',
      entityId: documentId,
      entityType: 'document',
      metadata: {
        fileName: metadata.fileName,
        fileSize: metadata.fileSize,
        documentType: metadata.documentType,
        caseId: metadata.caseId
      },
      outcome: 'success'
    });

    // Update profile stats
    userProfile.update(profile => {
      if (!profile) return profile;
      
      const newStats = {
        ...profile.stats,
        documentsProcessed: profile.stats.documentsProcessed + 1,
        uploadsCompleted: profile.stats.uploadsCompleted + 1,
        experience: profile.stats.experience + 10
      };

      return {
        ...profile,
        stats: this.levelUpCheck(newStats)
      };
    });
  }

  async trackAIInteraction(query: string, response: string, intent: string): Promise<void> {
    await this.trackActivity({
      type: 'ai_query',
      metadata: {
        queryLength: query.length,
        responseLength: response.length,
        intent,
        timestamp: new Date().toISOString()
      },
      outcome: 'success'
    });

    // Add to chat history
    currentSession.update(session => {
      if (!session) return session;
      
      const newChatEntry = {
        query,
        response,
        timestamp: new Date(),
        intent
      };

      return {
        ...session,
        aiChatHistory: [newChatEntry, ...session.aiChatHistory.slice(0, 49)]
      };
    });

    // Update stats
    userProfile.update(profile => {
      if (!profile) return profile;
      
      const newStats = {
        ...profile.stats,
        aiInteractions: profile.stats.aiInteractions + 1,
        experience: profile.stats.experience + 5
      };

      return {
        ...profile,
        stats: this.levelUpCheck(newStats)
      };
    });
  }

  async trackCaseActivity(caseId: string, activityType: string, metadata: unknown): Promise<void> {
    await this.trackActivity({
      type: 'case_created',
      entityId: caseId,
      entityType: 'case',
      metadata: {
        activityType,
        ...metadata
      },
      outcome: 'success'
    });

    // Add to open cases if not already there
    currentSession.update(session => {
      if (!session) return session;
      
      const openCases = session.openCases.includes(caseId) 
        ? session.openCases 
        : [caseId, ...session.openCases.slice(0, 9)];

      return {
        ...session,
        openCases
      };
    });
  }

  // Performance tracking
  private updateSessionPerformance(type: string, duration: number): void {
    currentSession.update(session => {
      if (!session) return session;
      
      return {
        ...session,
        performance: {
          ...session.performance,
          responseTimes: [...session.performance.responseTimes, duration].slice(-100)
        }
      };
    });
  }

  // Achievement system
  private checkAchievements(activity: UserActivity): void {
    // Implementation would check for various achievements
    // based on user activity patterns and milestones
  }

  // Level up system
  private levelUpCheck(stats: UserStats): UserStats {
    if (stats.experience >= stats.nextLevelXP) {
      return {
        ...stats,
        level: stats.level + 1,
        experience: stats.experience - stats.nextLevelXP,
        nextLevelXP: Math.floor(stats.nextLevelXP * 1.5)
      };
    }
    return stats;
  }

  // Data persistence
  private persistActivity(): void {
    if (!browser) return;
    
    activityFeed.subscribe(activities => {
      localStorage.setItem(this.activityStorage, JSON.stringify(activities));
    });
  }

  private persistSession(): void {
    if (!browser) return;
    
    currentSession.subscribe(session => {
      if (session) {
        localStorage.setItem(this.sessionStorage, JSON.stringify(session));
      }
    });
  }

  // Backend synchronization
  private async syncToBackend(activity: UserActivity): Promise<void> {
    try {
      await fetch('/api/user-activity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(activity)
      });
    } catch (error) {
      console.warn('Failed to sync activity to backend:', error);
    }
  }

  // Utility methods
  private getCurrentSessionId(): string {
    let sessionId = '';
    currentSession.subscribe(session => {
      sessionId = session?.id || '';
    })();
    return sessionId;
  }

  // Cleanup
  async cleanup(): Promise<void> {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    // Track session end
    await this.trackActivity({
      type: 'logout',
      metadata: {
        sessionEnd: new Date().toISOString()
      },
      outcome: 'success'
    });

    this.initialized = false;
  }
}

// Utility functions
function getStartOfDay(): Date {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), now.getDate());
}

function calculateEfficiency(activities: UserActivity[]): number {
  if (activities.length === 0) return 0;
  
  const successfulActivities = activities.filter(a => a.outcome === 'success');
  return Math.round((successfulActivities.length / activities.length) * 100);
}

function calculatePerformanceMetrics(
  activities: UserActivity[], 
  session: SessionData | null
): unknown {
  if (!session) return null;

  const avgResponseTime = session.performance.responseTimes.length > 0
    ? session.performance.responseTimes.reduce((a, b) => a + b, 0) / session.performance.responseTimes.length
    : 0;

  return {
    averageResponseTime: Math.round(avgResponseTime),
    errorRate: session.performance.errorCount / session.totalActions,
    actionsPerMinute: session.totalActions / ((Date.now() - session.startTime.getTime()) / 60000),
    taskCompletion: session.performance.taskCompletion
  };
}

// Singleton instance
export const yorhaTracker = new YoRHaUserTracker();