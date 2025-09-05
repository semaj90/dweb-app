// workers/document-processor.worker.js
const { Worker } = require('bullmq');
const axios = require('axios');
const Redis = require('ioredis');
const path = require('path');
const fs = require('fs').promises;

// Redis connection
const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  maxRetriesPerRequest: null,
  enableReadyCheck: false
});

// Go server configuration
const GO_SERVER_URL = process.env.GO_SERVER_URL || 'http://localhost:8080';

// Document processing worker
const documentWorker = new Worker('document-processing', async (job) => {
  const { documentId, filePath, documentType, caseId, options } = job.data;
  
  console.log(`[Document Worker] Processing document: ${documentId}`);
  job.updateProgress(10);
  
  try {
    // Read document content if file path provided
    let content = job.data.content;
    if (filePath && !content) {
      content = await fs.readFile(filePath, 'utf-8');
      job.updateProgress(20);
    }
    
    // Prepare processing options
    const processingOptions = {
      extract_entities: options?.extractEntities !== false,
      generate_summary: options?.generateSummary !== false,
      assess_risk: options?.assessRisk !== false,
      generate_embedding: options?.generateEmbedding !== false,
      store_in_database: options?.storeInDatabase !== false,
      use_gemma3_legal: true
    };
    
    job.updateProgress(30);
    
    // Call Go server for document processing
    const response = await axios.post(`${GO_SERVER_URL}/process-document`, {
      document_id: documentId,
      content: content,
      document_type: documentType || 'general',
      case_id: caseId,
      options: processingOptions
    }, {
      timeout: 120000, // 2 minute timeout for large documents
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    job.updateProgress(80);
    
    // Cache results in Redis for quick access
    if (response.data.success) {
      await redis.setex(
        `doc:${documentId}:result`,
        3600, // 1 hour cache
        JSON.stringify(response.data)
      );
      
      // Store embeddings separately for vector search
      if (response.data.embedding) {
        await redis.setex(
          `doc:${documentId}:embedding`,
          86400, // 24 hour cache for embeddings
          JSON.stringify(response.data.embedding)
        );
      }
    }
    
    job.updateProgress(100);
    
    return {
      success: true,
      documentId: documentId,
      processingTime: response.data.processing_time,
      summary: response.data.summary,
      entities: response.data.entities,
      riskAssessment: response.data.risk_assessment,
      hasEmbedding: !!response.data.embedding
    };
    
  } catch (error) {
    console.error(`[Document Worker] Error processing ${documentId}:`, error.message);
    throw error;
  }
}, {
  connection: redis,
  concurrency: 3, // Process up to 3 documents simultaneously
  removeOnComplete: { count: 100 },
  removeOnFail: { count: 50 }
});

// Handle worker events
documentWorker.on('completed', (job, result) => {
  console.log(`[Document Worker] Completed job ${job.id}:`, result.documentId);
});

documentWorker.on('failed', (job, err) => {
  console.error(`[Document Worker] Job ${job.id} failed:`, err.message);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down document worker...');
  await documentWorker.close();
  await redis.quit();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('Shutting down document worker...');
  await documentWorker.close();
  await redis.quit();
  process.exit(0);
});

console.log('✅ Document Processing Worker started');
console.log(`   - Concurrency: ${documentWorker.opts.concurrency}`);
console.log(`   - Connected to Go Server: ${GO_SERVER_URL}`);
