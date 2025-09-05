import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from '$lib/database/postgres-enhanced';
import { sql } from "drizzle-orm";
import { z } from 'zod';

// Schema for log data
const logEntrySchema = z.object({
  id: z.string(),
  timestamp: z.string().pipe(z.coerce.date()),
  level: z.enum(['error', 'warning', 'info', 'debug']),
  message: z.string(),
  details: z.any().optional(),
  stack: z.string().optional(),
  url: z.string().optional(),
  userAgent: z.string().optional(),
  userId: z.string().optional(),
  sessionId: z.string(),
  context: z.record(z.any()).optional(),
  resolved: z.boolean().default(false)
});

const performanceMetricSchema = z.object({
  id: z.string(),
  timestamp: z.string().pipe(z.coerce.date()),
  operation: z.string(),
  duration: z.number(),
  success: z.boolean(),
  metadata: z.record(z.any()).optional()
});

const monitoringPayloadSchema = z.object({
  sessionId: z.string(),
  userId: z.string().optional(),
  timestamp: z.string(),
  errors: z.array(logEntrySchema),
  performance: z.array(performanceMetricSchema),
  metadata: z.record(z.any()).optional()
});

/**
 * Receive monitoring data from frontend
 */
export const POST: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const body = await request.json();
    const payload = monitoringPayloadSchema.parse(body);

    // Store error logs
    if (payload.errors.length > 0) {
      await storeErrorLogs(payload.errors, payload.sessionId, payload.userId);
    }

    // Store performance metrics
    if (payload.performance.length > 0) {
      await storePerformanceMetrics(payload.performance, payload.sessionId, payload.userId);
    }

    return json({
      success: true,
      message: `Stored ${payload.errors.length} errors and ${payload.performance.length} performance metrics`,
      received: {
        errors: payload.errors.length,
        performance: payload.performance.length,
        sessionId: payload.sessionId
      }
    });

  } catch (error: any) {
    console.error("Monitoring logs error:", error);

    if (error instanceof z.ZodError) {
      return json({
        success: false,
        error: "Invalid monitoring payload",
        details: error.errors,
      }, { status: 400 });
    }

    return json({
      success: false,
      error: error?.message || "Failed to store monitoring data",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Get monitoring summary and health status
 */
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  try {
    const timeRange = url.searchParams.get("timeRange") || "24h";
    const level = url.searchParams.get("level");
    const sessionId = url.searchParams.get("sessionId");

    // Calculate time window
    const timeWindow = getTimeWindow(timeRange);

    // Get error summary
    const errorSummary = await getErrorSummary(timeWindow, level, sessionId);
    
    // Get performance summary
    const performanceSummary = await getPerformanceSummary(timeWindow, sessionId);

    // Get system health
    const systemHealth = await getSystemHealth();

    return json({
      success: true,
      timeRange,
      summary: {
        errors: errorSummary,
        performance: performanceSummary,
        health: systemHealth
      },
      generatedAt: new Date().toISOString()
    });

  } catch (error: any) {
    console.error("Monitoring summary error:", error);

    return json({
      success: false,
      error: "Failed to get monitoring summary",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Store error logs in database
 */
async function storeErrorLogs(
  errors: z.infer<typeof logEntrySchema>[],
  sessionId: string,
  userId?: string
): Promise<any> {
  try {
    // Create error logs table if it doesn't exist
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS error_logs (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        details JSONB,
        stack TEXT,
        url TEXT,
        user_agent TEXT,
        user_id TEXT,
        session_id TEXT NOT NULL,
        context JSONB,
        resolved BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);

    // Create index for performance
    await db.execute(sql`
      CREATE INDEX IF NOT EXISTS error_logs_timestamp_idx ON error_logs (timestamp);
      CREATE INDEX IF NOT EXISTS error_logs_level_idx ON error_logs (level);
      CREATE INDEX IF NOT EXISTS error_logs_session_idx ON error_logs (session_id);
      CREATE INDEX IF NOT EXISTS error_logs_user_idx ON error_logs (user_id);
    `);

    // Insert error logs
    for (const error of errors) {
      await db.execute(sql`
        INSERT INTO error_logs (
          id, timestamp, level, message, details, stack, url, 
          user_agent, user_id, session_id, context, resolved
        ) VALUES (
          ${error.id}, ${error.timestamp}, ${error.level}, ${error.message},
          ${error.details ? JSON.stringify(error.details) : null},
          ${error.stack}, ${error.url}, ${error.userAgent}, 
          ${error.userId || userId}, ${error.sessionId}, 
          ${error.context ? JSON.stringify(error.context) : null}, 
          ${error.resolved}
        )
        ON CONFLICT (id) DO NOTHING
      `);
    }

  } catch (error: any) {
    console.error('Error storing error logs:', error);
    throw error;
  }
}

/**
 * Store performance metrics in database
 */
async function storePerformanceMetrics(
  metrics: z.infer<typeof performanceMetricSchema>[],
  sessionId: string,
  userId?: string
): Promise<any> {
  try {
    // Create performance metrics table if it doesn't exist
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS performance_metrics (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        operation TEXT NOT NULL,
        duration DECIMAL NOT NULL,
        success BOOLEAN NOT NULL,
        metadata JSONB,
        session_id TEXT NOT NULL,
        user_id TEXT,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);

    // Create index for performance
    await db.execute(sql`
      CREATE INDEX IF NOT EXISTS performance_metrics_timestamp_idx ON performance_metrics (timestamp);
      CREATE INDEX IF NOT EXISTS performance_metrics_operation_idx ON performance_metrics (operation);
      CREATE INDEX IF NOT EXISTS performance_metrics_duration_idx ON performance_metrics (duration);
      CREATE INDEX IF NOT EXISTS performance_metrics_session_idx ON performance_metrics (session_id);
    `);

    // Insert performance metrics
    for (const metric of metrics) {
      await db.execute(sql`
        INSERT INTO performance_metrics (
          id, timestamp, operation, duration, success, metadata, session_id, user_id
        ) VALUES (
          ${metric.id}, ${metric.timestamp}, ${metric.operation}, ${metric.duration},
          ${metric.success}, ${metric.metadata ? JSON.stringify(metric.metadata) : null},
          ${sessionId}, ${userId}
        )
        ON CONFLICT (id) DO NOTHING
      `);
    }

  } catch (error: any) {
    console.error('Error storing performance metrics:', error);
    throw error;
  }
}

/**
 * Get error summary statistics
 */
async function getErrorSummary(
  timeWindow: string,
  level?: string | null,
  sessionId?: string | null
): Promise<any> {
  try {
    const whereConditions = [`timestamp >= NOW() - INTERVAL '${timeWindow}'`];
    
    if (level) {
      whereConditions.push(`level = '${level}'`);
    }
    
    if (sessionId) {
      whereConditions.push(`session_id = '${sessionId}'`);
    }

    const whereClause = whereConditions.join(' AND ');

    // Get total error counts by level
    const errorCounts = await db.execute(sql`
      SELECT 
        level,
        COUNT(*) as count
      FROM error_logs 
      WHERE ${sql.raw(whereClause)}
      GROUP BY level
      ORDER BY count DESC
    `);

    // Get recent errors
    const recentErrors = await db.execute(sql`
      SELECT 
        id, timestamp, level, message, url, session_id
      FROM error_logs 
      WHERE ${sql.raw(whereClause)}
      ORDER BY timestamp DESC 
      LIMIT 10
    `);

    // Get top error messages
    const topErrors = await db.execute(sql`
      SELECT 
        message,
        COUNT(*) as count
      FROM error_logs 
      WHERE ${sql.raw(whereClause)}
      GROUP BY message
      ORDER BY count DESC 
      LIMIT 10
    `);

    // Get error trends (hourly)
    const errorTrends = await db.execute(sql`
      SELECT 
        DATE_TRUNC('hour', timestamp) as hour,
        COUNT(*) as count
      FROM error_logs 
      WHERE ${sql.raw(whereClause)}
      GROUP BY DATE_TRUNC('hour', timestamp)
      ORDER BY hour DESC
      LIMIT 24
    `);

    return {
      totalErrors: errorCounts.reduce((sum, row) => sum + Number(row.count), 0),
      errorsByLevel: Object.fromEntries(errorCounts.map(row => [row.level, Number(row.count)])),
      recentErrors: recentErrors,
      topErrors: topErrors.map(row => ({
        message: row.message,
        count: Number(row.count)
      })),
      trends: errorTrends.map(row => ({
        hour: row.hour,
        count: Number(row.count)
      }))
    };

  } catch (error: any) {
    console.error('Error getting error summary:', error);
    return {
      totalErrors: 0,
      errorsByLevel: {},
      recentErrors: [],
      topErrors: [],
      trends: []
    };
  }
}

/**
 * Get performance summary statistics
 */
async function getPerformanceSummary(
  timeWindow: string,
  sessionId?: string | null
): Promise<any> {
  try {
    const whereConditions = [`timestamp >= NOW() - INTERVAL '${timeWindow}'`];
    
    if (sessionId) {
      whereConditions.push(`session_id = '${sessionId}'`);
    }

    const whereClause = whereConditions.join(' AND ');

    // Get performance statistics
    const performanceStats = await db.execute(sql`
      SELECT 
        AVG(duration) as avg_duration,
        MAX(duration) as max_duration,
        MIN(duration) as min_duration,
        COUNT(*) as total_operations,
        COUNT(CASE WHEN success = true THEN 1 END) as successful_operations,
        COUNT(CASE WHEN success = false THEN 1 END) as failed_operations
      FROM performance_metrics 
      WHERE ${sql.raw(whereClause)}
    `);

    // Get slowest operations
    const slowestOperations = await db.execute(sql`
      SELECT 
        operation,
        duration,
        timestamp,
        success,
        metadata
      FROM performance_metrics 
      WHERE ${sql.raw(whereClause)}
      ORDER BY duration DESC 
      LIMIT 10
    `);

    // Get operation counts
    const operationCounts = await db.execute(sql`
      SELECT 
        operation,
        COUNT(*) as count,
        AVG(duration) as avg_duration
      FROM performance_metrics 
      WHERE ${sql.raw(whereClause)}
      GROUP BY operation
      ORDER BY count DESC
    `);

    const stats = performanceStats[0];
    const totalOps = Number(stats?.total_operations || 0);
    const successfulOps = Number(stats?.successful_operations || 0);

    return {
      averageResponseTime: Number(stats?.avg_duration || 0),
      maxResponseTime: Number(stats?.max_duration || 0),
      minResponseTime: Number(stats?.min_duration || 0),
      totalOperations: totalOps,
      successRate: totalOps > 0 ? successfulOps / totalOps : 1,
      slowestOperations: slowestOperations.map(op => ({
        operation: op.operation,
        duration: Number(op.duration),
        timestamp: op.timestamp,
        success: op.success,
        metadata: op.metadata
      })),
      operationCounts: Object.fromEntries(
        operationCounts.map(op => [
          op.operation, 
          { 
            count: Number(op.count), 
            avgDuration: Number(op.avg_duration) 
          }
        ])
      )
    };

  } catch (error: any) {
    console.error('Error getting performance summary:', error);
    return {
      averageResponseTime: 0,
      maxResponseTime: 0,
      minResponseTime: 0,
      totalOperations: 0,
      successRate: 1,
      slowestOperations: [],
      operationCounts: {}
    };
  }
}

/**
 * Get system health status
 */
async function getSystemHealth(): Promise<any> {
  try {
    // Check database connection
    const dbHealth = await checkDatabaseHealth();
    
    // Check recent error rates
    const recentErrorRate = await getRecentErrorRate();
    
    // Check performance metrics
    const avgResponseTime = await getAverageResponseTime();

    const isHealthy = dbHealth.connected && 
                     recentErrorRate < 0.05 && // Less than 5% error rate
                     avgResponseTime < 5000; // Less than 5 seconds

    return {
      status: isHealthy ? 'healthy' : 'degraded',
      database: dbHealth,
      errorRate: recentErrorRate,
      averageResponseTime: avgResponseTime,
      lastChecked: new Date().toISOString(),
      uptime: process.uptime()
    };

  } catch (error: any) {
    console.error('Error getting system health:', error);
    return {
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
      lastChecked: new Date().toISOString()
    };
  }
}

/**
 * Check database connection health
 */
async function checkDatabaseHealth(): Promise<{ connected: boolean; responseTime?: number; error?: string }> {
  try {
    const start = Date.now();
    await db.execute(sql`SELECT 1`);
    const responseTime = Date.now() - start;

    return {
      connected: true,
      responseTime
    };
  } catch (error: any) {
    return {
      connected: false,
      error: error instanceof Error ? error.message : 'Unknown database error'
    };
  }
}

/**
 * Get recent error rate (last hour)
 */
async function getRecentErrorRate(): Promise<number> {
  try {
    const results = await db.execute(sql`
      SELECT 
        COUNT(CASE WHEN level = 'error' THEN 1 END) as errors,
        COUNT(*) as total
      FROM error_logs 
      WHERE timestamp >= NOW() - INTERVAL '1 hour'
    `);

    const stats = results[0];
    const total = Number(stats?.total || 0);
    const errors = Number(stats?.errors || 0);

    return total > 0 ? errors / total : 0;

  } catch (error: any) {
    console.error('Error getting recent error rate:', error);
    return 0;
  }
}

/**
 * Get average response time (last hour)
 */
async function getAverageResponseTime(): Promise<number> {
  try {
    const results = await db.execute(sql`
      SELECT AVG(duration) as avg_duration
      FROM performance_metrics 
      WHERE timestamp >= NOW() - INTERVAL '1 hour'
    `);

    return Number(results[0]?.avg_duration || 0);

  } catch (error: any) {
    console.error('Error getting average response time:', error);
    return 0;
  }
}

/**
 * Convert time range string to SQL interval
 */
function getTimeWindow(timeRange: string): string {
  switch (timeRange) {
    case '1h': return '1 hour';
    case '6h': return '6 hours';
    case '24h': return '24 hours';
    case '7d': return '7 days';
    case '30d': return '30 days';
    default: return '24 hours';
  }
}