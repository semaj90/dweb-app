/**
 * Recommendation Worker for Legal AI Platform
 * Handles background processing of user behavior patterns and AI recommendations
 */

// Worker state
let userPatterns = new Map();
let recommendationCache = new Map();
let processingQueue = [];

// Message handling
self.onmessage = function(event) {
  const { action, data } = event.data;
  
  switch (action) {
    case 'process_enhanced_recommendations':
      processEnhancedRecommendations(data);
      break;
      
    case 'analyze_user_patterns':
      analyzeUserPatterns(data);
      break;
      
    case 'update_user_behavior':
      updateUserBehavior(data);
      break;
      
    case 'get_recommendations':
      getRecommendations(data);
      break;
      
    case 'clear_cache':
      clearCache();
      break;
      
    default:
      console.warn('Unknown action:', action);
  }
};

/**
 * Process enhanced recommendations with all available data
 */
function processEnhancedRecommendations(data) {
  const { userContext, recommendations, query, legalDomain } = data;
  
  try {
    // Update user patterns
    updateUserPatterns(userContext.userId, {
      query,
      legalDomain,
      timestamp: Date.now(),
      recommendations: recommendations.length
    });
    
    // Enhance recommendations with behavioral data
    const enhancedRecommendations = recommendations.map(rec => ({
      ...rec,
      personalizedScore: calculatePersonalizedScore(rec, userContext),
      behavioralContext: getUserBehavioralContext(userContext.userId),
      priority: calculateRecommendationPriority(rec, userContext)
    }));
    
    // Store in cache
    const cacheKey = `${userContext.userId}_${legalDomain}_${hashQuery(query)}`;
    recommendationCache.set(cacheKey, {
      recommendations: enhancedRecommendations,
      timestamp: Date.now(),
      userContext,
      legalDomain
    });
    
    // Send enhanced recommendations back
    self.postMessage({
      type: 'enhanced_recommendations_ready',
      data: {
        recommendations: enhancedRecommendations,
        userContext,
        cacheKey
      }
    });
    
    // Analyze patterns for future improvements
    analyzeRecommendationPatterns(userContext.userId, enhancedRecommendations);
    
  } catch (error) {
    console.error('❌ Worker: Failed to process enhanced recommendations:', error);
    self.postMessage({
      type: 'error',
      data: { message: error.message, action: 'process_enhanced_recommendations' }
    });
  }
}

/**
 * Update user behavioral patterns
 */
function updateUserPatterns(userId, interaction) {
  if (!userPatterns.has(userId)) {
    userPatterns.set(userId, {
      queries: [],
      domains: new Map(),
      preferences: new Map(),
      timeline: [],
      lastActive: Date.now()
    });
  }
  
  const patterns = userPatterns.get(userId);
  
  // Add to query history
  patterns.queries.push({
    query: interaction.query,
    timestamp: interaction.timestamp,
    domain: interaction.legalDomain
  });
  
  // Keep only last 100 queries
  if (patterns.queries.length > 100) {
    patterns.queries = patterns.queries.slice(-100);
  }
  
  // Update domain preferences
  const domainCount = patterns.domains.get(interaction.legalDomain) || 0;
  patterns.domains.set(interaction.legalDomain, domainCount + 1);
  
  // Add to timeline
  patterns.timeline.push({
    type: 'recommendation_request',
    timestamp: interaction.timestamp,
    data: interaction
  });
  
  // Keep only last 200 timeline entries
  if (patterns.timeline.length > 200) {
    patterns.timeline = patterns.timeline.slice(-200);
  }
  
  patterns.lastActive = Date.now();
}

/**
 * Calculate personalized score for recommendations
 */
function calculatePersonalizedScore(recommendation, userContext) {
  let score = recommendation.relevance || 0.5;
  
  const userId = userContext.userId;
  if (!userPatterns.has(userId)) return score;
  
  const patterns = userPatterns.get(userId);
  
  // Boost score based on domain preference
  const domainUsage = patterns.domains.get(recommendation.category) || 0;
  const totalDomainUsage = Array.from(patterns.domains.values()).reduce((a, b) => a + b, 0);
  
  if (totalDomainUsage > 0) {
    const domainPreference = domainUsage / totalDomainUsage;
    score += domainPreference * 0.3;
  }
  
  // Boost based on user type
  if (userContext.userType === 'attorney' && recommendation.category === 'litigation') {
    score += 0.2;
  } else if (userContext.userType === 'paralegal' && recommendation.type === 'tip') {
    score += 0.15;
  }
  
  // Penalize if recently shown
  const recentRecommendations = patterns.timeline
    .filter(t => t.type === 'recommendation_shown')
    .slice(-20);
  
  const wasRecentlyShown = recentRecommendations.some(r => 
    r.data.recommendationId === recommendation.id
  );
  
  if (wasRecentlyShown) {
    score -= 0.2;
  }
  
  return Math.max(0, Math.min(1, score));
}

/**
 * Calculate recommendation priority
 */
function calculateRecommendationPriority(recommendation, userContext) {
  let priority = 'medium';
  
  // High priority for critical legal domains
  if (recommendation.category === 'litigation' || recommendation.category === 'compliance') {
    priority = 'high';
  }
  
  // Higher priority for new users
  const userId = userContext.userId;
  if (userPatterns.has(userId)) {
    const patterns = userPatterns.get(userId);
    if (patterns.queries.length < 5) {
      priority = 'high';
    }
  }
  
  // Higher priority based on relevance
  if (recommendation.relevance > 0.8) {
    priority = 'high';
  } else if (recommendation.relevance < 0.3) {
    priority = 'low';
  }
  
  return priority;
}

/**
 * Get behavioral context for user
 */
function getUserBehavioralContext(userId) {
  if (!userPatterns.has(userId)) {
    return { experience: 'new', preferences: [], recentActivity: [] };
  }
  
  const patterns = userPatterns.get(userId);
  
  // Determine experience level
  let experience = 'new';
  if (patterns.queries.length > 50) {
    experience = 'expert';
  } else if (patterns.queries.length > 20) {
    experience = 'experienced';
  } else if (patterns.queries.length > 5) {
    experience = 'intermediate';
  }
  
  // Get top preferences
  const preferences = Array.from(patterns.domains.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([domain, count]) => ({ domain, count }));
  
  // Recent activity
  const recentActivity = patterns.timeline
    .slice(-10)
    .map(t => ({ type: t.type, timestamp: t.timestamp }));
  
  return { experience, preferences, recentActivity };
}

/**
 * Analyze recommendation patterns for insights
 */
function analyzeRecommendationPatterns(userId, recommendations) {
  // This is where we would run background analytics
  // For now, just log patterns
  
  const patterns = userPatterns.get(userId);
  if (!patterns) return;
  
  // Calculate recommendation distribution
  const categoryDistribution = recommendations.reduce((dist, rec) => {
    dist[rec.category] = (dist[rec.category] || 0) + 1;
    return dist;
  }, {});
  
  // Look for trends
  const recentQueries = patterns.queries.slice(-10);
  const queryDomains = recentQueries.map(q => q.domain);
  const dominantDomain = [...new Set(queryDomains)]
    .map(domain => ({ 
      domain, 
      count: queryDomains.filter(d => d === domain).length 
    }))
    .sort((a, b) => b.count - a.count)[0];
  
  // Store insights (in real implementation, this would go to analytics service)
  const insights = {
    userId,
    timestamp: Date.now(),
    categoryDistribution,
    dominantDomain: dominantDomain?.domain,
    queryFrequency: patterns.queries.length,
    averageRelevance: recommendations.reduce((sum, r) => sum + r.relevance, 0) / recommendations.length
  };
  
  console.log('📊 Recommendation insights:', insights);
}

/**
 * Analyze user patterns for behavioral insights
 */
function analyzeUserPatterns(data) {
  const { userId } = data;
  
  if (!userPatterns.has(userId)) {
    self.postMessage({
      type: 'user_patterns_analyzed',
      data: { userId, patterns: null, insights: [] }
    });
    return;
  }
  
  const patterns = userPatterns.get(userId);
  
  // Generate insights
  const insights = [];
  
  // Query frequency insights
  const recentQueries = patterns.queries.filter(
    q => Date.now() - q.timestamp < 24 * 60 * 60 * 1000 // Last 24 hours
  );
  
  if (recentQueries.length > 10) {
    insights.push({
      type: 'high_activity',
      message: 'User is highly active today',
      recommendation: 'Consider showing advanced features'
    });
  }
  
  // Domain expertise insights
  const topDomain = Array.from(patterns.domains.entries())
    .sort((a, b) => b[1] - a[1])[0];
  
  if (topDomain && topDomain[1] > 20) {
    insights.push({
      type: 'domain_expert',
      message: `User appears to specialize in ${topDomain[0]}`,
      recommendation: `Show advanced ${topDomain[0]} features`
    });
  }
  
  self.postMessage({
    type: 'user_patterns_analyzed',
    data: { userId, patterns, insights }
  });
}

/**
 * Update user behavior data
 */
function updateUserBehavior(data) {
  const { userId, behavior } = data;
  
  if (!userPatterns.has(userId)) {
    updateUserPatterns(userId, { query: '', legalDomain: 'general', timestamp: Date.now() });
  }
  
  const patterns = userPatterns.get(userId);
  
  // Add behavior data
  patterns.timeline.push({
    type: 'behavior_update',
    timestamp: Date.now(),
    data: behavior
  });
  
  // Update preferences based on behavior
  if (behavior.preference) {
    const currentPref = patterns.preferences.get(behavior.preference.key) || 0;
    patterns.preferences.set(behavior.preference.key, currentPref + behavior.preference.weight);
  }
  
  self.postMessage({
    type: 'user_behavior_updated',
    data: { userId, success: true }
  });
}

/**
 * Get recommendations from cache
 */
function getRecommendations(data) {
  const { userId, legalDomain, query } = data;
  const cacheKey = `${userId}_${legalDomain}_${hashQuery(query)}`;
  
  if (recommendationCache.has(cacheKey)) {
    const cached = recommendationCache.get(cacheKey);
    
    // Check if cache is still fresh (5 minutes)
    if (Date.now() - cached.timestamp < 5 * 60 * 1000) {
      self.postMessage({
        type: 'recommendations_found',
        data: cached
      });
      return;
    }
  }
  
  self.postMessage({
    type: 'recommendations_not_found',
    data: { cacheKey, userId, legalDomain, query }
  });
}

/**
 * Clear all caches
 */
function clearCache() {
  userPatterns.clear();
  recommendationCache.clear();
  processingQueue = [];
  
  self.postMessage({
    type: 'cache_cleared',
    data: { success: true }
  });
}

/**
 * Hash query for cache key generation
 */
function hashQuery(query) {
  let hash = 0;
  if (query.length === 0) return hash;
  for (let i = 0; i < query.length; i++) {
    const char = query.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36);
}

// Periodic cleanup (every 30 minutes)
setInterval(() => {
  const now = Date.now();
  const thirtyMinutes = 30 * 60 * 1000;
  
  // Clean old cache entries
  for (const [key, value] of recommendationCache.entries()) {
    if (now - value.timestamp > thirtyMinutes) {
      recommendationCache.delete(key);
    }
  }
  
  // Clean inactive user patterns (older than 24 hours)
  const twentyFourHours = 24 * 60 * 60 * 1000;
  for (const [userId, patterns] of userPatterns.entries()) {
    if (now - patterns.lastActive > twentyFourHours) {
      userPatterns.delete(userId);
    }
  }
  
  console.log('🧹 Worker: Cleaned up old data');
}, 30 * 60 * 1000);

console.log('✅ Legal AI Recommendation Worker initialized');