import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { sql } from 'drizzle-orm';

export const GET: RequestHandler = async () => {
  try {
    const checks = {
      database: false,
      redis: false,
      ollama: false,
      minio: false,
      qdrant: false,
      rabbitmq: false,
      services: {
        enhancedRag: false,
        gpuOrchestrator: false,
        vectorProcessor: false,
        documentAnalyzer: false
      }
    };
    
    // Check PostgreSQL
    try {
      await db.execute(sql`SELECT 1`);
      checks.database = true;
    } catch (err) {
      console.error('Database check failed:', err);
    }
    
    // Check Redis
    try {
      const response = await fetch('http://localhost:6379');
      checks.redis = response.ok;
    } catch (err) {
      // Redis doesn't have HTTP endpoint, check via command
      checks.redis = false;
    }
    
    // Check Ollama
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      checks.ollama = response.ok;
    } catch (err) {
      console.error('Ollama check failed:', err);
    }
    
    // Check MinIO
    try {
      const response = await fetch('http://localhost:9000/minio/health/live');
      checks.minio = response.ok;
    } catch (err) {
      console.error('MinIO check failed:', err);
    }
    
    // Check Qdrant
    try {
      const response = await fetch('http://localhost:6333/collections');
      checks.qdrant = response.ok;
    } catch (err) {
      console.error('Qdrant check failed:', err);
    }
    
    // Check Go microservices
    const services = [
      { name: 'enhancedRag', url: 'http://localhost:8094/health' },
      { name: 'gpuOrchestrator', url: 'http://localhost:8095/health' },
      { name: 'vectorProcessor', url: 'http://localhost:8096/health' },
      { name: 'documentAnalyzer', url: 'http://localhost:8097/health' }
    ];
    
    for (const service of services) {
      try {
        const response = await fetch(service.url);
        checks.services[service.name] = response.ok;
      } catch (err) {
        console.error(`${service.name} check failed:`, err);
      }
    }
    
    // Calculate overall health
    const coreHealthy = checks.database && checks.ollama;
    const allHealthy = Object.values(checks).every(v => 
      typeof v === 'boolean' ? v : Object.values(v).every(s => s)
    );
    
    return json({
      status: allHealthy ? 'healthy' : coreHealthy ? 'degraded' : 'unhealthy',
      timestamp: new Date().toISOString(),
      checks,
      message: allHealthy 
        ? 'All systems operational' 
        : coreHealthy 
          ? 'Core systems operational, some services degraded'
          : 'Critical systems offline'
    });
  } catch (error) {
    console.error('Health check error:', error);
    return json({
      status: 'error',
      message: 'Health check failed',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};