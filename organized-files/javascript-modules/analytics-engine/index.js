const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const { Pool } = require('pg');
const Redis = require('redis');
const cron = require('node-cron');
const _ = require('lodash');
const moment = require('moment');
const math = require('mathjs');
const { Matrix } = require('ml-matrix');
const stats = require('simple-statistics');
const natural = require('natural');
const Sentiment = require('sentiment');

const app = express();
const PORT = process.env.PORT || 8082;

// Initialize services
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_v3'
});

const redis = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

const sentiment = new Sentiment();

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:4173'],
  credentials: true
}));
app.use(express.json());

// Connect to Redis
redis.connect().catch(console.error);

class LegalAnalytics {
  constructor() {
    this.cachePrefix = 'analytics:';
    this.cacheTTL = 3600; // 1 hour
  }

  async getCachedOrCompute(key, computeFunction, ttl = this.cacheTTL) {
    const cacheKey = `${this.cachePrefix}${key}`;
    try {
      const cached = await redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
    } catch (error) {
      console.warn('Cache read error:', error);
    }

    const result = await computeFunction();
    
    try {
      await redis.setEx(cacheKey, ttl, JSON.stringify(result));
    } catch (error) {
      console.warn('Cache write error:', error);
    }

    return result;
  }

  async getCaseAnalytics(timeframe = '30d') {
    return this.getCachedOrCompute(`case_analytics_${timeframe}`, async () => {
      const client = await pool.connect();
      try {
        const timeCondition = this.getTimeCondition(timeframe);
        
        // Basic case statistics
        const caseStatsQuery = `
          SELECT 
            status,
            COUNT(*) as count,
            AVG(EXTRACT(EPOCH FROM (updated_at - created_at))/86400) as avg_duration_days,
            MIN(created_at) as earliest_case,
            MAX(created_at) as latest_case
          FROM cases 
          WHERE created_at >= NOW() - INTERVAL '${timeCondition}'
          GROUP BY status
          ORDER BY count DESC
        `;
        
        const caseStats = await client.query(caseStatsQuery);
        
        // Case volume trends
        const trendQuery = `
          SELECT 
            DATE_TRUNC('day', created_at) as date,
            COUNT(*) as cases_created,
            COUNT(CASE WHEN status = 'closed' THEN 1 END) as cases_closed
          FROM cases 
          WHERE created_at >= NOW() - INTERVAL '${timeCondition}'
          GROUP BY DATE_TRUNC('day', created_at)
          ORDER BY date
        `;
        
        const trends = await client.query(trendQuery);
        
        // Performance metrics
        const performanceQuery = `
          SELECT 
            AVG(CASE WHEN status = 'closed' THEN EXTRACT(EPOCH FROM (updated_at - created_at))/86400 END) as avg_resolution_days,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (updated_at - created_at))/86400) as median_duration,
            COUNT(CASE WHEN status = 'active' AND created_at < NOW() - INTERVAL '30 days' THEN 1 END) as overdue_cases
          FROM cases 
          WHERE created_at >= NOW() - INTERVAL '${timeCondition}'
        `;
        
        const performance = await client.query(performanceQuery);
        
        return {
          summary: {
            total_cases: caseStats.rows.reduce((sum, row) => sum + parseInt(row.count), 0),
            by_status: caseStats.rows,
            performance: performance.rows[0]
          },
          trends: trends.rows,
          timeframe
        };
      } finally {
        client.release();
      }
    }, 1800); // 30 minutes cache
  }

  async getEvidenceAnalytics(caseId = null) {
    const cacheKey = caseId ? `evidence_analytics_case_${caseId}` : 'evidence_analytics_global';
    
    return this.getCachedOrCompute(cacheKey, async () => {
      const client = await pool.connect();
      try {
        const whereClause = caseId ? `WHERE case_id = $1` : '';
        const params = caseId ? [caseId] : [];
        
        // Evidence type distribution
        const typeQuery = `
          SELECT 
            type,
            COUNT(*) as count,
            AVG(
              CASE 
                WHEN metadata->>'relevance_score' IS NOT NULL 
                THEN (metadata->>'relevance_score')::float 
                ELSE 0.5 
              END
            ) as avg_relevance
          FROM evidence 
          ${whereClause}
          GROUP BY type
          ORDER BY count DESC
        `;
        
        const types = await client.query(typeQuery, params);
        
        // Evidence timeline
        const timelineQuery = `
          SELECT 
            DATE_TRUNC('day', created_at) as date,
            COUNT(*) as evidence_added,
            type
          FROM evidence 
          ${whereClause}
          GROUP BY DATE_TRUNC('day', created_at), type
          ORDER BY date
        `;
        
        const timeline = await client.query(timelineQuery, params);
        
        // Quality metrics
        const qualityQuery = `
          SELECT 
            AVG(CASE WHEN metadata->>'quality_score' IS NOT NULL THEN (metadata->>'quality_score')::float ELSE 0.7 END) as avg_quality,
            COUNT(CASE WHEN metadata->>'has_issues' = 'true' THEN 1 END) as flagged_items,
            COUNT(*) as total_items
          FROM evidence 
          ${whereClause}
        `;
        
        const quality = await client.query(qualityQuery, params);
        
        return {
          type_distribution: types.rows,
          timeline: timeline.rows,
          quality_metrics: quality.rows[0]
        };
      } finally {
        client.release();
      }
    });
  }

  async getPredictiveAnalytics(caseId) {
    return this.getCachedOrCompute(`predictive_${caseId}`, async () => {
      const client = await pool.connect();
      try {
        // Get case details
        const caseQuery = `
          SELECT 
            *,
            EXTRACT(EPOCH FROM (NOW() - created_at))/86400 as days_active,
            (SELECT COUNT(*) FROM evidence WHERE case_id = cases.id) as evidence_count
          FROM cases 
          WHERE id = $1
        `;
        
        const caseData = await client.query(caseQuery, [caseId]);
        if (caseData.rows.length === 0) {
          throw new Error('Case not found');
        }
        
        const currentCase = caseData.rows[0];
        
        // Get similar historical cases for prediction
        const similarCasesQuery = `
          SELECT 
            id,
            status,
            EXTRACT(EPOCH FROM (updated_at - created_at))/86400 as duration_days,
            (SELECT COUNT(*) FROM evidence WHERE case_id = cases.id) as evidence_count
          FROM cases 
          WHERE 
            id != $1 
            AND status = 'closed'
            AND created_at >= NOW() - INTERVAL '2 years'
          ORDER BY created_at DESC
          LIMIT 500
        `;
        
        const similarCases = await client.query(similarCasesQuery, [caseId]);
        
        // Calculate predictions
        const predictions = this.calculateCasePredictions(currentCase, similarCases.rows);
        
        return predictions;
      } finally {
        client.release();
      }
    }, 7200); // 2 hours cache
  }

  calculateCasePredictions(currentCase, historicalCases) {
    if (historicalCases.length === 0) {
      return {
        estimated_completion_days: null,
        confidence: 0,
        similar_cases_analyzed: 0
      };
    }

    // Simple statistical analysis
    const durations = historicalCases.map(c => c.duration_days).filter(d => d > 0);
    
    const avgDuration = stats.mean(durations);
    const medianDuration = stats.median(durations);
    const stdDev = stats.standardDeviation(durations);
    
    // Weight prediction based on evidence count similarity
    const currentEvidenceCount = currentCase.evidence_count || 0;
    const similarByEvidence = historicalCases.filter(c => 
      Math.abs(c.evidence_count - currentEvidenceCount) <= 3
    );
    
    let estimatedDays = avgDuration;
    let confidence = 0.3;
    
    if (similarByEvidence.length > 5) {
      const similarDurations = similarByEvidence.map(c => c.duration_days);
      estimatedDays = stats.mean(similarDurations);
      confidence = Math.min(0.9, 0.3 + (similarByEvidence.length / 50));
    }
    
    return {
      estimated_completion_days: Math.round(estimatedDays),
      confidence: Math.round(confidence * 100),
      similar_cases_analyzed: historicalCases.length,
      statistics: {
        avg_duration: Math.round(avgDuration),
        median_duration: Math.round(medianDuration),
        std_deviation: Math.round(stdDev)
      }
    };
  }

  getTimeCondition(timeframe) {
    const timeMap = {
      '7d': '7 days',
      '30d': '30 days',
      '90d': '90 days',
      '1y': '1 year'
    };
    return timeMap[timeframe] || '30 days';
  }
}

const analytics = new LegalAnalytics();

// API Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'Legal Analytics Engine',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

app.get('/api/analytics/cases', async (req, res) => {
  try {
    const { timeframe = '30d' } = req.query;
    const result = await analytics.getCaseAnalytics(timeframe);
    res.json(result);
  } catch (error) {
    console.error('Case analytics error:', error);
    res.status(500).json({ error: 'Failed to generate case analytics' });
  }
});

app.get('/api/analytics/evidence', async (req, res) => {
  try {
    const { case_id } = req.query;
    const result = await analytics.getEvidenceAnalytics(case_id);
    res.json(result);
  } catch (error) {
    console.error('Evidence analytics error:', error);
    res.status(500).json({ error: 'Failed to generate evidence analytics' });
  }
});

app.get('/api/analytics/predictions/:caseId', async (req, res) => {
  try {
    const { caseId } = req.params;
    const result = await analytics.getPredictiveAnalytics(caseId);
    res.json(result);
  } catch (error) {
    console.error('Predictive analytics error:', error);
    res.status(500).json({ error: 'Failed to generate predictions' });
  }
});

// Real-time analytics endpoint
app.get('/api/analytics/realtime', async (req, res) => {
  try {
    const client = await pool.connect();
    
    const realtimeQuery = `
      SELECT 
        (SELECT COUNT(*) FROM cases WHERE status = 'active') as active_cases,
        (SELECT COUNT(*) FROM cases WHERE created_at >= NOW() - INTERVAL '1 hour') as cases_last_hour,
        (SELECT COUNT(*) FROM evidence WHERE created_at >= NOW() - INTERVAL '1 hour') as evidence_last_hour,
        (SELECT COUNT(*) FROM users WHERE last_login >= NOW() - INTERVAL '1 hour') as active_users_last_hour
    `;
    
    const result = await client.query(realtimeQuery);
    client.release();
    
    res.json({
      timestamp: new Date().toISOString(),
      metrics: result.rows[0]
    });
  } catch (error) {
    console.error('Real-time analytics error:', error);
    res.status(500).json({ error: 'Failed to get real-time data' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`📊 Legal Analytics Engine running on port ${PORT}`);
});

module.exports = app;