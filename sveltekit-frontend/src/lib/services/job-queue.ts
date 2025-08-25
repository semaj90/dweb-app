import { Queue, Worker, Job, type JobsOptions } from "bullmq";
import Redis from "ioredis";

// Job types for the legal document processing pipeline
interface BaseJobData {
  uploadId: string;
  caseId: string;
  timestamp: string;
  priority: 'low' | 'normal' | 'high' | 'critical';
}

interface DocumentExtractionJob extends BaseJobData {
  filename: string;
  contentType: string;
  storageUrl: string;
  extractionType: 'pdf' | 'image' | 'video' | 'audio' | 'text';
}

interface EmbeddingJob extends BaseJobData {
  textChunks: string[];
  chunkMetadata: Array<{ page?: number; timestamp?: number; coordinates?: [number, number, number, number] }>;
  embeddingModel: 'sentence-transformers' | 'ollama' | 'openai';
}

interface TensorProcessingJob extends BaseJobData {
  tensorData: number[];
  dimensions: [number, number, number, number]; // 4D tensor
  operation: 'tricubic' | 'som_cluster' | 'attention' | 'convolution';
  tileSize?: [number, number, number, number];
  haloSize?: number;
}

interface VectorIndexJob extends BaseJobData {
  embeddings: number[][];
  metadata: Array<{ docId: string; chunkId: string; text: string; [key: string]: unknown }>;
  indexType: 'qdrant' | 'pgvector' | 'faiss';
}

type JobData = DocumentExtractionJob | EmbeddingJob | TensorProcessingJob | VectorIndexJob;

// Redis connection configuration
const redisConfig = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  maxRetriesPerRequest: 3,
  // retryDelayOnFailover: 100, // removed - deprecated in ioredis v5+
  lazyConnect: true,
};

export class LegalAIJobQueue {
  private redis: Redis;
  private queues: Map<string, Queue>;
  private workers: Map<string, Worker>;

  constructor() {
    this.redis = new Redis(redisConfig);
    this.queues = new Map();
    this.workers = new Map();
    
    this.initializeQueues();
  }

  private initializeQueues() {
    const queueConfigs = [
      { name: 'document-extraction', concurrency: 5 },
      { name: 'embedding-generation', concurrency: 3 },
      { name: 'tensor-processing', concurrency: 2 },
      { name: 'vector-indexing', concurrency: 4 },
      { name: 'notification', concurrency: 10 },
    ];

    queueConfigs.forEach(({ name, concurrency }) => {
      const queue = new Queue(name, {
        connection: this.redis.duplicate(),
        defaultJobOptions: {
          removeOnComplete: 100,
          removeOnFail: 50,
          attempts: 3,
          backoff: {
            type: 'exponential',
            delay: 2000,
          },
        },
      });

      this.queues.set(name, queue);
      
      // Create worker for each queue
      const worker = new Worker(
        name,
        this.createJobProcessor(name),
        {
          connection: this.redis.duplicate(),
          concurrency,
          limiter: {
            max: concurrency * 2,
            duration: 1000,
          },
        }
      );

      // Worker event handlers
      worker.on('completed', (job) => {
        console.log(`✅ Job ${job.id} completed in queue ${name}`);
        this.broadcastProgress(job.data.uploadId, {
          stage: name,
          status: 'completed',
          progress: 100,
          result: job.returnvalue,
        });
      });

      worker.on('failed', (job, err) => {
        console.error(`❌ Job ${job?.id} failed in queue ${name}:`, err.message);
        this.broadcastProgress(job?.data.uploadId, {
          stage: name,
          status: 'failed',
          error: err.message,
        });
      });

      worker.on('progress', (job, progress) => {
        console.log(`🔄 Job ${job.id} progress: ${progress}%`);
        this.broadcastProgress(job.data.uploadId, {
          stage: name,
          status: 'processing',
          progress,
        });
      });

      this.workers.set(name, worker);
    });
  }

  private createJobProcessor(queueName: string) {
    return async (job: Job) => {
      const { data, id, name } = job;
      
      console.log(`🚀 Processing job ${id} in ${queueName}`);
      
      switch (queueName) {
        case 'document-extraction':
          return this.processDocumentExtraction(job);
        case 'embedding-generation':
          return this.processEmbeddingGeneration(job);
        case 'tensor-processing':
          return this.processTensorProcessing(job);
        case 'vector-indexing':
          return this.processVectorIndexing(job);
        case 'notification':
          return this.processNotification(job);
        default:
          throw new Error(`Unknown queue: ${queueName}`);
      }
    };
  }

  // Document extraction processor
  private async processDocumentExtraction(job: Job<DocumentExtractionJob>) {
    const { filename, contentType, storageUrl, extractionType } = job.data;
    
    job.updateProgress(10);
    
    try {
      let extractedText = '';
      let metadata = {};

      switch (extractionType) {
        case 'pdf':
          // Call Python microservice for PDF extraction
          const pdfResponse = await fetch('http://localhost:8082/extract/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: storageUrl, filename }),
          });
          const pdfResult = await pdfResponse.json();
          extractedText = pdfResult.text;
          metadata = pdfResult.metadata;
          break;

        case 'image':
          // OCR processing
          const ocrResponse = await fetch('http://localhost:8082/extract/ocr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: storageUrl, filename }),
          });
          const ocrResult = await ocrResponse.json();
          extractedText = ocrResult.text;
          metadata = ocrResult.metadata;
          break;

        case 'video':
          // Video transcription with Whisper
          const videoResponse = await fetch('http://localhost:8082/extract/video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: storageUrl, filename }),
          });
          const videoResult = await videoResponse.json();
          extractedText = videoResult.transcript;
          metadata = videoResult.metadata;
          break;

        default:
          throw new Error(`Unsupported extraction type: ${extractionType}`);
      }

      job.updateProgress(80);

      // Chunk text for embedding
      const chunks = this.chunkText(extractedText, 512, 40); // 512 tokens, 40 overlap
      
      job.updateProgress(90);

      // Queue embedding generation
      await this.addEmbeddingJob({
        ...job.data,
        textChunks: chunks.map(c => c.text),
        chunkMetadata: chunks.map(c => c.metadata),
        embeddingModel: 'sentence-transformers',
      });

      job.updateProgress(100);

      return {
        extractedText,
        chunkCount: chunks.length,
        metadata,
        nextStage: 'embedding-generation',
      };

    } catch (error) {
      console.error('❌ Document extraction failed:', error);
      throw error;
    }
  }

  // Embedding generation processor
  private async processEmbeddingGeneration(job: Job<EmbeddingJob>) {
    const { textChunks, embeddingModel } = job.data;
    
    job.updateProgress(10);

    try {
      const embeddings: number[][] = [];
      const batchSize = 10;

      for (let i = 0; i < textChunks.length; i += batchSize) {
        const batch = textChunks.slice(i, i + batchSize);
        
        // Call embedding service
        const response = await fetch('http://localhost:8083/embed', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            texts: batch,
            model: embeddingModel,
          }),
        });

        const result = await response.json();
        embeddings.push(...result.embeddings);
        
        const progress = Math.min(90, (i / textChunks.length) * 80 + 10);
        job.updateProgress(progress);
      }

      // Queue vector indexing
      await this.addVectorIndexJob({
        ...job.data,
        embeddings,
        metadata: textChunks.map((text, idx) => ({
          docId: job.data.uploadId,
          chunkId: `${job.data.uploadId}-${idx}`,
          text,
          ...job.data.chunkMetadata[idx],
        })),
        indexType: 'pgvector',
      });

      job.updateProgress(100);

      return {
        embeddingCount: embeddings.length,
        dimensions: embeddings[0]?.length || 0,
        nextStage: 'vector-indexing',
      };

    } catch (error) {
      console.error('❌ Embedding generation failed:', error);
      throw error;
    }
  }

  // Tensor processing with QUIC
  private async processTensorProcessing(job: Job<TensorProcessingJob>) {
    const { tensorData, dimensions, operation, tileSize, haloSize } = job.data;
    
    job.updateProgress(10);

    try {
      // Send to QUIC tensor server
      const response = await fetch('https://localhost:4433/tensor/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: job.id,
          upload_id: job.data.uploadId,
          tensor_tile: {
            tile_id: `${job.data.uploadId}-${job.id}`,
            dimensions,
            halo_size: haloSize || 2,
            data: tensorData,
            metadata: {
              case_id: job.data.caseId,
              operation,
            },
          },
          operation,
          timestamp: new Date().toISOString(),
        }),
      });

      const result = await response.json();
      
      job.updateProgress(50);

      // Poll for result
      let tensorResult;
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds timeout

      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const resultResponse = await fetch(
          `https://localhost:4433/tensor/result?job_id=${job.id}`
        );
        
        if (resultResponse.ok) {
          tensorResult = await resultResponse.json();
          if (tensorResult.status === 'completed') {
            break;
          }
        }
        
        attempts++;
        job.updateProgress(50 + (attempts / maxAttempts) * 40);
      }

      if (!tensorResult || tensorResult.status !== 'completed') {
        throw new Error('Tensor processing timeout or failed');
      }

      job.updateProgress(100);

      return {
        tensorResult,
        metrics: tensorResult.metrics,
        outputSize: tensorResult.output_data?.length || 0,
      };

    } catch (error) {
      console.error('❌ Tensor processing failed:', error);
      throw error;
    }
  }

  // Vector indexing processor
  private async processVectorIndexing(job: Job<VectorIndexJob>) {
    const { embeddings, metadata, indexType } = job.data;
    
    job.updateProgress(10);

    try {
      switch (indexType) {
        case 'pgvector':
          // Store in PostgreSQL with pgvector
          const pgResponse = await fetch('http://localhost:5432/api/vectors/insert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              embeddings,
              metadata,
              table: 'legal_embeddings',
            }),
          });
          break;

        case 'qdrant':
          // Store in Qdrant
          const qdrantResponse = await fetch('http://localhost:6333/collections/legal-docs/points', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              points: embeddings.map((vector, idx) => ({
                id: metadata[idx].chunkId,
                vector,
                payload: metadata[idx],
              })),
            }),
          });
          break;

        default:
          throw new Error(`Unsupported index type: ${indexType}`);
      }

      job.updateProgress(80);

      // Update document status in database
      // await updateDocumentStatus(job.data.uploadId, 'indexed');

      // Send completion notification
      await this.addNotificationJob({
        ...job.data,
        message: `Document ${job.data.uploadId} successfully processed and indexed`,
        type: 'completion',
      });

      job.updateProgress(100);

      return {
        indexedCount: embeddings.length,
        indexType,
        status: 'completed',
      };

    } catch (error) {
      console.error('❌ Vector indexing failed:', error);
      throw error;
    }
  }

  // Notification processor
  private async processNotification(job: Job) {
    // Send real-time notification via WebSocket
    // This would integrate with your WebSocket server
    console.log(`📢 Notification: ${job.data.message}`);
    
    return { sent: true };
  }

  // Helper methods for chunking text
  private chunkText(text: string, maxTokens: number = 512, overlap: number = 40) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const chunks = [];
    let currentChunk = '';
    let currentTokens = 0;

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i].trim() + '.';
      const sentenceTokens = Math.ceil(sentence.length / 4); // Rough token estimation
      
      if (currentTokens + sentenceTokens > maxTokens && currentChunk.length > 0) {
        chunks.push({
          text: currentChunk.trim(),
          metadata: { sentenceStart: chunks.length * (maxTokens - overlap) }
        });
        
        // Start new chunk with overlap
        const overlapSentences = sentences.slice(Math.max(0, i - 2), i);
        currentChunk = overlapSentences.join('. ') + '. ' + sentence;
        currentTokens = Math.ceil(currentChunk.length / 4);
      } else {
        currentChunk += ' ' + sentence;
        currentTokens += sentenceTokens;
      }
    }

    if (currentChunk.trim().length > 0) {
      chunks.push({
        text: currentChunk.trim(),
        metadata: { sentenceStart: chunks.length * (maxTokens - overlap) }
      });
    }

    return chunks;
  }

  // Public methods to add jobs
  async addDocumentExtractionJob(data: DocumentExtractionJob, options?: JobsOptions) {
    const queue = this.queues.get('document-extraction');
    return queue?.add('extract-document', data, {
      priority: this.getPriority(data.priority),
      ...options,
    });
  }

  async addEmbeddingJob(data: EmbeddingJob, options?: JobsOptions) {
    const queue = this.queues.get('embedding-generation');
    return queue?.add('generate-embeddings', data, {
      priority: this.getPriority(data.priority),
      ...options,
    });
  }

  async addTensorProcessingJob(data: TensorProcessingJob, options?: JobsOptions) {
    const queue = this.queues.get('tensor-processing');
    return queue?.add('process-tensor', data, {
      priority: this.getPriority(data.priority),
      ...options,
    });
  }

  async addVectorIndexJob(data: VectorIndexJob, options?: JobsOptions) {
    const queue = this.queues.get('vector-indexing');
    return queue?.add('index-vectors', data, {
      priority: this.getPriority(data.priority),
      ...options,
    });
  }

  async addNotificationJob(data: any, options?: JobsOptions) {
    const queue = this.queues.get('notification');
    return queue?.add('send-notification', data, options);
  }

  private getPriority(priority: string): number {
    const priorities = { low: 1, normal: 5, high: 10, critical: 20 };
    return priorities[priority as keyof typeof priorities] || 5;
  }

  private async broadcastProgress(uploadId: string, progress: any) {
    // This would integrate with your WebSocket server
    console.log(`📊 Progress for ${uploadId}:`, progress);
    
    // Store progress in Redis for polling
    await this.redis.setex(`progress:${uploadId}`, 3600, JSON.stringify(progress));
  }

  // Cleanup
  async close() {
    await Promise.all([
      ...Array.from(this.queues.values()).map(q => q.close()),
      ...Array.from(this.workers.values()).map(w => w.close()),
      this.redis.disconnect(),
    ]);
  }
}

// Export singleton instance
export const jobQueue = new LegalAIJobQueue();