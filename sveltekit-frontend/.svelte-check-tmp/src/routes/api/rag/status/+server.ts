import type { RequestHandler } from './$types';

/**
 * RAG Status API - Returns system status for file processing pipeline
 */

async function checkServiceHealth(url: string, timeout = 5000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(url, {
      signal: controller.signal,
      method: 'GET'
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    return false;
  }
}

export const GET: RequestHandler = async () => {
  try {
    // Check PostgreSQL (native Windows service)
    const postgresHealthy = await checkServiceHealth('http://localhost:5432');

    // Check Redis (native Windows service)
    const redisHealthy = await checkServiceHealth('http://localhost:6379');

    // Check Qdrant (native Windows service)
    const qdrantHealthy = await checkServiceHealth('http://localhost:6333/collections');

    // Check embeddings service (Ollama - native Windows)
    const embeddingsHealthy = await checkServiceHealth('http://localhost:11434/api/tags');

    // Native file processing - always available
    const ocrHealthy = true;
    const storageHealthy = true; // Local file system
    const searchHealthy = true; // Built-in vector search

    const status = {
      overall: postgresHealthy && redisHealthy && qdrantHealthy && embeddingsHealthy,
      services: {
        postgresql: { healthy: postgresHealthy, url: 'localhost:5432' },
        redis: { healthy: redisHealthy, url: 'localhost:6379' },
        qdrant: { healthy: qdrantHealthy, url: 'localhost:6333' },
        embeddings: { healthy: embeddingsHealthy, url: 'http://localhost:11434' },
        ocr: { healthy: ocrHealthy, url: 'native-processing' },
        storage: { healthy: storageHealthy, url: 'file-system' },
        search: { healthy: searchHealthy, url: 'built-in' }
      },
      timestamp: new Date().toISOString()
    };

    return json(status);
  } catch (error: any) {
    return json(
      {
        error: 'Failed to check system status',
        details: error.message,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};
