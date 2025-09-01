/// <reference types="vite/client" />
import { healthCheck } from "$lib/server/db/index.js";
import type { RequestHandler } from './$types';


// Environment variables for Ollama configuration
const OLLAMA_URL = import.meta.env.OLLAMA_URL || 'http://localhost:11434';
const OLLAMA_TIMEOUT = 5000; // 5 seconds

export const GET: RequestHandler = async () => {
  try {
    const systemStatus = {
      services: {
        ollama: await checkOllamaStatus(),
        database: await checkDatabaseStatus()
      },
      environment: {
        ollamaUrl: OLLAMA_URL
      },
      timestamp: new Date().toISOString()
    };

    return json(systemStatus);
  } catch (error: any) {
    console.error('System status check failed:', error);
    return json(
      {
        services: {
          ollama: { status: 'error', error: 'System check failed' },
          database: { status: 'error', error: 'System check failed' }
        },
        environment: { ollamaUrl: OLLAMA_URL },
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};

async function checkOllamaStatus(): Promise<any> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT);

    const response = await fetch(`${OLLAMA_URL}/api/version`, {
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      status: 'connected',
      version: data.version || 'unknown',
      url: OLLAMA_URL
    };
  } catch (error: any) {
    console.error('Ollama connection failed:', error);
    
    let errorMessage = 'Connection failed';
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        errorMessage = 'Connection timeout';
      } else if (error.message.includes('fetch')) {
        errorMessage = 'Service unavailable';
      } else {
        errorMessage = error.message;
      }
    }

    return {
      status: 'error',
      error: errorMessage,
      url: OLLAMA_URL
    };
  }
}

async function checkDatabaseStatus(): Promise<any> {
  try {
    const result = await healthCheck();
    
    if (result.status === 'healthy') {
      return {
        status: 'connected',
        type: 'PostgreSQL',
        tablesAccessible: (result as any).tablesAccessible
      };
    } else {
      return {
        status: 'error',
        error: result.error,
        type: 'PostgreSQL'
      };
    }
  } catch (error: any) {
    console.error('Database health check failed:', error);
    
    return {
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
      type: 'PostgreSQL'
    };
  }
}

// Also support POST for triggering system checks
export const POST = GET;