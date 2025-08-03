import { test, expect } from '@playwright/test';

test.describe('System Health and Logging', () => {
  test('should verify all services are healthy', async ({ page }) => {
    const response = await page.request.get('/api/health');
    expect(response.status()).toBe(200);
    
    const health = await response.json();
    
    // Check overall status
    expect(health).toHaveProperty('status');
    expect(health.status).toBe('healthy');
    
    // Check individual services
    expect(health).toHaveProperty('services');
    
    // Database health
    expect(health.services.database).toEqual({
      status: 'healthy',
      connected: true,
      response_time_ms: expect.any(Number)
    });
    
    // Redis/Cache health
    expect(health.services.cache).toEqual({
      status: 'healthy',
      connected: true,
      response_time_ms: expect.any(Number)
    });
    
    // Ollama health
    expect(health.services.ollama).toEqual({
      status: 'healthy',
      available: true,
      models_loaded: expect.any(Array),
      gpu_enabled: expect.any(Boolean)
    });
    
    // Vector store health
    expect(health.services.vector_store).toEqual({
      status: 'healthy',
      connected: true,
      index_count: expect.any(Number)
    });
    
    // Message queue health (if using RabbitMQ)
    if (health.services.message_queue) {
      expect(health.services.message_queue).toEqual({
        status: 'healthy',
        connected: true,
        queues: expect.any(Array)
      });
    }
  });

  test('should monitor system metrics', async ({ page }) => {
    const response = await page.request.get('/api/metrics');
    expect(response.status()).toBe(200);
    
    const metrics = await response.json();
    
    // CPU metrics
    expect(metrics.cpu).toMatchObject({
      usage_percent: expect.any(Number),
      load_average: expect.any(Array),
      cores: expect.any(Number)
    });
    
    // Memory metrics
    expect(metrics.memory).toMatchObject({
      total_mb: expect.any(Number),
      used_mb: expect.any(Number),
      free_mb: expect.any(Number),
      usage_percent: expect.any(Number)
    });
    
    // Disk metrics
    expect(metrics.disk).toMatchObject({
      total_gb: expect.any(Number),
      used_gb: expect.any(Number),
      free_gb: expect.any(Number),
      usage_percent: expect.any(Number)
    });
    
    // Process metrics
    expect(metrics.process).toMatchObject({
      uptime_seconds: expect.any(Number),
      memory_usage_mb: expect.any(Number),
      cpu_usage_percent: expect.any(Number),
      active_handles: expect.any(Number),
      active_requests: expect.any(Number)
    });
  });

  test('should track application logs', async ({ page }) => {
    // Trigger some actions to generate logs
    await page.goto('/login');
    await page.fill('input[name="email"]', 'test@example.com');
    await page.fill('input[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');
    
    // Wait for error
    await page.waitForSelector('.error-message, [data-testid="error-message"]');
    
    // Check logs
    const response = await page.request.get('/api/logs/recent', {
      params: {
        level: 'error',
        limit: 10
      }
    });
    
    expect(response.status()).toBe(200);
    const logs = await response.json();
    
    expect(logs).toHaveProperty('entries');
    expect(Array.isArray(logs.entries)).toBe(true);
    
    // Find login error log
    const loginError = logs.entries.find((log: any) => 
      log.message.includes('login') || log.message.includes('authentication')
    );
    
    expect(loginError).toBeDefined();
    expect(loginError).toMatchObject({
      timestamp: expect.any(String),
      level: 'error',
      message: expect.any(String),
      context: expect.objectContaining({
        email: 'test@example.com'
      })
    });
  });

  test('should monitor API performance', async ({ page }) => {
    // Make several API calls
    const endpoints = [
      '/api/health',
      '/api/cases',
      '/api/documents',
      '/api/ai/models'
    ];
    
    for (const endpoint of endpoints) {
      await page.request.get(endpoint).catch(() => {}); // Ignore auth errors
    }
    
    // Check performance metrics
    const response = await page.request.get('/api/metrics/performance');
    expect(response.status()).toBe(200);
    
    const performance = await response.json();
    
    expect(performance).toHaveProperty('endpoints');
    
    // Check metrics for each endpoint
    endpoints.forEach(endpoint => {
      if (performance.endpoints[endpoint]) {
        expect(performance.endpoints[endpoint]).toMatchObject({
          avg_response_time_ms: expect.any(Number),
          p95_response_time_ms: expect.any(Number),
          p99_response_time_ms: expect.any(Number),
          request_count: expect.any(Number),
          error_rate: expect.any(Number)
        });
      }
    });
  });

  test('should track database query performance', async ({ page }) => {
    // Login to trigger database queries
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Load some data
    await page.goto('/dashboard/cases');
    await page.goto('/dashboard/documents');
    
    // Check query metrics
    const response = await page.request.get('/api/metrics/database');
    expect(response.status()).toBe(200);
    
    const dbMetrics = await response.json();
    
    expect(dbMetrics).toHaveProperty('queries');
    expect(dbMetrics.queries).toMatchObject({
      total_count: expect.any(Number),
      slow_query_count: expect.any(Number),
      avg_duration_ms: expect.any(Number),
      connection_pool: {
        active: expect.any(Number),
        idle: expect.any(Number),
        waiting: expect.any(Number)
      }
    });
    
    // Check for slow queries
    if (dbMetrics.slow_queries && dbMetrics.slow_queries.length > 0) {
      dbMetrics.slow_queries.forEach((query: any) => {
        expect(query).toMatchObject({
          query: expect.any(String),
          duration_ms: expect.any(Number),
          timestamp: expect.any(String)
        });
      });
    }
  });

  test('should monitor error rates and patterns', async ({ page }) => {
    const response = await page.request.get('/api/metrics/errors');
    expect(response.status()).toBe(200);
    
    const errorMetrics = await response.json();
    
    expect(errorMetrics).toHaveProperty('summary');
    expect(errorMetrics.summary).toMatchObject({
      total_errors_24h: expect.any(Number),
      error_rate: expect.any(Number),
      top_errors: expect.any(Array)
    });
    
    // Check error patterns
    if (errorMetrics.summary.top_errors.length > 0) {
      errorMetrics.summary.top_errors.forEach((error: any) => {
        expect(error).toMatchObject({
          type: expect.any(String),
          count: expect.any(Number),
          last_occurrence: expect.any(String),
          affected_endpoints: expect.any(Array)
        });
      });
    }
  });

  test('should track user activity logs', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Perform some actions
    await page.goto('/dashboard/cases');
    await page.click('[data-testid="case-item"]').first().catch(() => {});
    await page.goto('/dashboard/documents');
    
    // Check activity logs
    const response = await page.request.get('/api/logs/user-activity');
    expect(response.status()).toBe(200);
    
    const activityLogs = await response.json();
    
    expect(activityLogs).toHaveProperty('activities');
    expect(Array.isArray(activityLogs.activities)).toBe(true);
    
    // Recent activities should include our actions
    const recentActivities = activityLogs.activities.slice(0, 5);
    const activityTypes = recentActivities.map((a: any) => a.action);
    
    expect(activityTypes).toContain('login');
    expect(activityTypes).toContain('page_view');
  });

  test('should monitor WebSocket connections', async ({ page }) => {
    const response = await page.request.get('/api/metrics/websockets');
    expect(response.status()).toBe(200);
    
    const wsMetrics = await response.json();
    
    expect(wsMetrics).toMatchObject({
      active_connections: expect.any(Number),
      total_connections: expect.any(Number),
      messages_sent: expect.any(Number),
      messages_received: expect.any(Number),
      avg_latency_ms: expect.any(Number)
    });
  });

  test('should check cache performance', async ({ page }) => {
    // Trigger some cacheable requests
    for (let i = 0; i < 5; i++) {
      await page.request.get('/api/config');
      await page.request.get('/api/ai/models');
    }
    
    const response = await page.request.get('/api/metrics/cache');
    expect(response.status()).toBe(200);
    
    const cacheMetrics = await response.json();
    
    expect(cacheMetrics).toMatchObject({
      hit_rate: expect.any(Number),
      miss_rate: expect.any(Number),
      total_hits: expect.any(Number),
      total_misses: expect.any(Number),
      eviction_count: expect.any(Number),
      memory_usage_mb: expect.any(Number),
      keys_count: expect.any(Number)
    });
    
    // Hit rate should be > 0 after repeated requests
    expect(cacheMetrics.hit_rate).toBeGreaterThan(0);
  });

  test('should monitor background job status', async ({ page }) => {
    const response = await page.request.get('/api/jobs/status');
    expect(response.status()).toBe(200);
    
    const jobStatus = await response.json();
    
    expect(jobStatus).toHaveProperty('queues');
    
    // Check each queue
    Object.entries(jobStatus.queues).forEach(([queueName, stats]: [string, any]) => {
      expect(stats).toMatchObject({
        active: expect.any(Number),
        waiting: expect.any(Number),
        completed: expect.any(Number),
        failed: expect.any(Number),
        delayed: expect.any(Number)
      });
    });
    
    // Check for failed jobs
    if (jobStatus.failed_jobs && jobStatus.failed_jobs.length > 0) {
      jobStatus.failed_jobs.forEach((job: any) => {
        expect(job).toMatchObject({
          id: expect.any(String),
          name: expect.any(String),
          error: expect.any(String),
          failed_at: expect.any(String),
          attempts: expect.any(Number)
        });
      });
    }
  });

  test('should verify logging configuration', async ({ page }) => {
    const response = await page.request.get('/api/logs/config');
    expect(response.status()).toBe(200);
    
    const logConfig = await response.json();
    
    expect(logConfig).toMatchObject({
      levels: {
        app: expect.stringMatching(/debug|info|warn|error/),
        http: expect.stringMatching(/debug|info|warn|error/),
        db: expect.stringMatching(/debug|info|warn|error/)
      },
      transports: expect.arrayContaining(['console', 'file']),
      retention_days: expect.any(Number),
      max_file_size_mb: expect.any(Number)
    });
  });

  test('should export system diagnostics', async ({ page }) => {
    const response = await page.request.get('/api/diagnostics/export');
    expect(response.status()).toBe(200);
    
    const diagnostics = await response.json();
    
    expect(diagnostics).toHaveProperty('timestamp');
    expect(diagnostics).toHaveProperty('system');
    expect(diagnostics).toHaveProperty('application');
    expect(diagnostics).toHaveProperty('dependencies');
    expect(diagnostics).toHaveProperty('recent_errors');
    expect(diagnostics).toHaveProperty('performance_summary');
    
    // Should be able to use this for debugging
    console.log('System Diagnostics Summary:', {
      uptime: diagnostics.system.uptime,
      memory_usage: diagnostics.system.memory.usage_percent,
      error_count: diagnostics.recent_errors.length,
      avg_response_time: diagnostics.performance_summary.avg_response_time_ms
    });
  });
});