import type { RequestHandler } from '@sveltejs/kit';
// @ts-nocheck
import type { RequestHandler }, { json } from "@sveltejs/kit";

// Mock AI health endpoint for development/testing
export const GET: RequestHandler = async () => {
  return json({
    status: 'healthy',
    models: ['mock-legal-ai', 'gemma3-legal', 'mistral-7b', 'llama3.1-8b'],
    uptime: '5d 12h 34m',
    version: '1.0.0-mock',
    mock: true,
    connections: {
      active: 1,
      total: 42
    },
    performance: {
      averageResponseTime: 750,
      requestsPerMinute: 12,
      successRate: 98.5
    },
    timestamp: new Date().toISOString()
  });
};