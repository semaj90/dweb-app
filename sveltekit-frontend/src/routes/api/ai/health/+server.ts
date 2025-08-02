import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';

export const GET: RequestHandler = async () => {
  try {
    // Check Ollama health
    const ollamaHealth = await fetch('http://localhost:11434/api/tags', {
      signal: AbortSignal.timeout(5000)
    }).then(r => r.ok ? 'healthy' : 'down').catch(() => 'down');

    // Check local services
    const checks = {
      ollama: ollamaHealth,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      version: process.version
    };

    const overallStatus = ollamaHealth === 'healthy' ? 'healthy' : 'degraded';

    return json({
      status: overallStatus,
      services: checks,
      message: overallStatus === 'healthy' ? 'All systems operational' : 'Some services degraded'
    });

  } catch (error) {
    return json({
      status: 'critical',
      error: (error as Error).message,
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
};
