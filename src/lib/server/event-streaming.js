import amqp from "amqplib";
import { Redis } from "ioredis";
import { langchainRAGService } from "./langchain-rag.js";
import { embeddingService } from "./embedding.js";
import { db } from "./db/index.js";
import { documents, documentChunks } from "./db/schema.js";
import {
  RABBITMQ_URL,
  REDIS_URL,
  DOCUMENT_QUEUE_NAME,
  EMBEDDING_QUEUE_NAME,
  JOB_RETRY_ATTEMPTS,
  JOB_TIMEOUT,
} from "$env/static/private";

/**
 * Phase 4: RabbitMQ Event Streaming Service
 * Handles background document processing, embedding generation, and job queuing
 */

class EventStreamingService {
  constructor() {
    this.connection = null;
    this.channel = null;
    this.redis = null;
    this.isConnected = false;
    this.queues = {
      documentProcessing: DOCUMENT_QUEUE_NAME || "document_processing",
      embeddingGeneration: EMBEDDING_QUEUE_NAME || "embedding_generation",
      caseAnalysis: "case_analysis",
      reportGeneration: "report_generation",
      batchProcessing: "batch_processing",
    };
  }

  /**
   * Initialize RabbitMQ connection and Redis
   */
  async initialize() {
    if (this.isConnected) return;

    try {
      console.log("🐰 Initializing RabbitMQ Event Streaming Service...");

      // Connect to RabbitMQ
      this.connection = await amqp.connect(
        RABBITMQ_URL || "amqp://localhost:5672"
      );
      this.channel = await this.connection.createChannel();

      // Connect to Redis for job tracking
      this.redis = new Redis(REDIS_URL || "redis://localhost:6379");

      // Declare all queues
      for (const [name, queueName] of Object.entries(this.queues)) {
        await this.channel.assertQueue(queueName, {
          durable: true,
          arguments: {
            "x-message-ttl": parseInt(JOB_TIMEOUT) || 300000, // 5 minutes
            "x-max-retries": parseInt(JOB_RETRY_ATTEMPTS) || 3,
          },
        });
        console.log(`✅ Queue declared: ${queueName}`);
      }

      // Set up error handling
      this.connection.on("error", (err) => {
        console.error("❌ RabbitMQ connection error:", err);
        this.isConnected = false;
      });

      this.connection.on("close", () => {
        console.log("🔌 RabbitMQ connection closed");
        this.isConnected = false;
      });

      this.isConnected = true;
      console.log("🎉 Event Streaming Service initialized");

      // Start consuming jobs
      await this.startConsumers();
    } catch (error) {
      console.error("❌ Failed to initialize Event Streaming Service:", error);
      throw error;
    }
  }

  /**
   * Start all job consumers
   */
  async startConsumers() {
    // Document Processing Consumer
    await this.channel.consume(
      this.queues.documentProcessing,
      async (msg) => {
        if (msg) {
          await this.processDocumentJob(msg);
        }
      },
      { noAck: false }
    );

    // Embedding Generation Consumer
    await this.channel.consume(
      this.queues.embeddingGeneration,
      async (msg) => {
        if (msg) {
          await this.processEmbeddingJob(msg);
        }
      },
      { noAck: false }
    );

    // Case Analysis Consumer
    await this.channel.consume(
      this.queues.caseAnalysis,
      async (msg) => {
        if (msg) {
          await this.processCaseAnalysisJob(msg);
        }
      },
      { noAck: false }
    );

    // Report Generation Consumer
    await this.channel.consume(
      this.queues.reportGeneration,
      async (msg) => {
        if (msg) {
          await this.processReportJob(msg);
        }
      },
      { noAck: false }
    );

    // Batch Processing Consumer
    await this.channel.consume(
      this.queues.batchProcessing,
      async (msg) => {
        if (msg) {
          await this.processBatchJob(msg);
        }
      },
      { noAck: false }
    );

    console.log("👂 All job consumers started");
  }

  /**
   * Queue a document for processing
   */
  async queueDocumentProcessing(documentData) {
    await this.initialize();

    const job = {
      id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: "document_processing",
      data: documentData,
      createdAt: new Date().toISOString(),
      attempts: 0,
      maxAttempts: parseInt(JOB_RETRY_ATTEMPTS) || 3,
    };

    // Store job in Redis for tracking
    await this.redis.setex(`job:${job.id}`, 3600, JSON.stringify(job));

    // Send to queue
    await this.channel.sendToQueue(
      this.queues.documentProcessing,
      Buffer.from(JSON.stringify(job)),
      { persistent: true }
    );

    console.log(`📤 Queued document processing job: ${job.id}`);
    return job.id;
  }

  /**
   * Queue embedding generation
   */
  async queueEmbeddingGeneration(embeddingData) {
    await this.initialize();

    const job = {
      id: `emb_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: "embedding_generation",
      data: embeddingData,
      createdAt: new Date().toISOString(),
      attempts: 0,
      maxAttempts: parseInt(JOB_RETRY_ATTEMPTS) || 3,
    };

    await this.redis.setex(`job:${job.id}`, 3600, JSON.stringify(job));

    await this.channel.sendToQueue(
      this.queues.embeddingGeneration,
      Buffer.from(JSON.stringify(job)),
      { persistent: true }
    );

    console.log(`📤 Queued embedding generation job: ${job.id}`);
    return job.id;
  }

  /**
   * Queue case analysis
   */
  async queueCaseAnalysis(caseData) {
    await this.initialize();

    const job = {
      id: `case_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: "case_analysis",
      data: caseData,
      createdAt: new Date().toISOString(),
      attempts: 0,
      maxAttempts: parseInt(JOB_RETRY_ATTEMPTS) || 3,
    };

    await this.redis.setex(`job:${job.id}`, 7200, JSON.stringify(job)); // 2 hours for case analysis

    await this.channel.sendToQueue(
      this.queues.caseAnalysis,
      Buffer.from(JSON.stringify(job)),
      { persistent: true }
    );

    console.log(`📤 Queued case analysis job: ${job.id}`);
    return job.id;
  }

  /**
   * Queue report generation
   */
  async queueReportGeneration(reportData) {
    await this.initialize();

    const job = {
      id: `rep_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: "report_generation",
      data: reportData,
      createdAt: new Date().toISOString(),
      attempts: 0,
      maxAttempts: parseInt(JOB_RETRY_ATTEMPTS) || 3,
    };

    await this.redis.setex(`job:${job.id}`, 3600, JSON.stringify(job));

    await this.channel.sendToQueue(
      this.queues.reportGeneration,
      Buffer.from(JSON.stringify(job)),
      { persistent: true }
    );

    console.log(`📤 Queued report generation job: ${job.id}`);
    return job.id;
  }

  /**
   * Process document processing job
   */
  async processDocumentJob(msg) {
    const job = JSON.parse(msg.content.toString());

    try {
      console.log(`📄 Processing document job: ${job.id}`);

      const { documentId, content, metadata } = job.data;

      // Update job status
      await this.updateJobStatus(job.id, "processing");

      // Process with LangChain
      const result = await langchainRAGService.addDocuments([
        {
          content,
          ...metadata,
        },
      ]);

      // Update database
      await db
        .update(documents)
        .set({
          status: "processed",
          processedAt: new Date(),
          metadata: { ...metadata, processingResult: result },
        })
        .where(eq(documents.id, documentId));

      // Update job status
      await this.updateJobStatus(job.id, "completed", result);

      this.channel.ack(msg);
      console.log(`✅ Document job completed: ${job.id}`);
    } catch (error) {
      console.error(`❌ Document job failed: ${job.id}`, error);
      await this.handleJobError(job, msg, error);
    }
  }

  /**
   * Process embedding generation job
   */
  async processEmbeddingJob(msg) {
    const job = JSON.parse(msg.content.toString());

    try {
      console.log(`🧮 Processing embedding job: ${job.id}`);

      const { text, documentId, chunkIndex } = job.data;

      await this.updateJobStatus(job.id, "processing");

      // Generate embedding
      const embedding = await embeddingService.generateEmbedding(text);

      // Store in database
      await db
        .update(documentChunks)
        .set({ embedding })
        .where(
          and(
            eq(documentChunks.documentId, documentId),
            eq(documentChunks.chunkIndex, chunkIndex)
          )
        );

      await this.updateJobStatus(job.id, "completed", {
        embedding: "generated",
      });

      this.channel.ack(msg);
      console.log(`✅ Embedding job completed: ${job.id}`);
    } catch (error) {
      console.error(`❌ Embedding job failed: ${job.id}`, error);
      await this.handleJobError(job, msg, error);
    }
  }

  /**
   * Process case analysis job
   */
  async processCaseAnalysisJob(msg) {
    const job = JSON.parse(msg.content.toString());

    try {
      console.log(`⚖️ Processing case analysis job: ${job.id}`);

      const { caseId, analysisType, parameters } = job.data;

      await this.updateJobStatus(job.id, "processing");

      // Perform analysis using RAG
      const query = this.buildAnalysisQuery(analysisType, parameters);
      const analysis = await langchainRAGService.processQuery(query, {
        caseId,
      });

      // Store analysis results
      const analysisResult = {
        caseId,
        analysisType,
        results: analysis,
        generatedAt: new Date().toISOString(),
        jobId: job.id,
      };

      await this.updateJobStatus(job.id, "completed", analysisResult);

      this.channel.ack(msg);
      console.log(`✅ Case analysis completed: ${job.id}`);
    } catch (error) {
      console.error(`❌ Case analysis failed: ${job.id}`, error);
      await this.handleJobError(job, msg, error);
    }
  }

  /**
   * Process report generation job
   */
  async processReportJob(msg) {
    const job = JSON.parse(msg.content.toString());

    try {
      console.log(`📊 Processing report job: ${job.id}`);

      const { reportType, caseId, parameters } = job.data;

      await this.updateJobStatus(job.id, "processing");

      // Generate report using RAG
      const reportQuery = this.buildReportQuery(reportType, parameters);
      const reportContent = await langchainRAGService.processQuery(
        reportQuery,
        { caseId }
      );

      const report = {
        type: reportType,
        caseId,
        content: reportContent,
        generatedAt: new Date().toISOString(),
        parameters,
        jobId: job.id,
      };

      await this.updateJobStatus(job.id, "completed", report);

      this.channel.ack(msg);
      console.log(`✅ Report generation completed: ${job.id}`);
    } catch (error) {
      console.error(`❌ Report generation failed: ${job.id}`, error);
      await this.handleJobError(job, msg, error);
    }
  }

  /**
   * Process batch job
   */
  async processBatchJob(msg) {
    const job = JSON.parse(msg.content.toString());

    try {
      console.log(`🗂️ Processing batch job: ${job.id}`);

      const { operation, items } = job.data;

      await this.updateJobStatus(job.id, "processing");

      const results = [];
      for (const item of items) {
        try {
          let result;
          switch (operation) {
            case "bulk_embedding":
              result = await embeddingService.generateEmbedding(item.text);
              break;
            case "bulk_analysis":
              result = await langchainRAGService.processQuery(item.query);
              break;
            default:
              throw new Error(`Unknown batch operation: ${operation}`);
          }
          results.push({ item, result, status: "success" });
        } catch (error) {
          results.push({ item, error: error.message, status: "error" });
        }
      }

      await this.updateJobStatus(job.id, "completed", { results });

      this.channel.ack(msg);
      console.log(`✅ Batch job completed: ${job.id}`);
    } catch (error) {
      console.error(`❌ Batch job failed: ${job.id}`, error);
      await this.handleJobError(job, msg, error);
    }
  }

  /**
   * Update job status in Redis
   */
  async updateJobStatus(jobId, status, result = null) {
    const jobKey = `job:${jobId}`;
    const job = await this.redis.get(jobKey);

    if (job) {
      const jobData = JSON.parse(job);
      jobData.status = status;
      jobData.updatedAt = new Date().toISOString();

      if (result) {
        jobData.result = result;
      }

      await this.redis.setex(jobKey, 3600, JSON.stringify(jobData));
    }

    // Also store in status-specific key
    await this.redis.setex(`job:${jobId}:status`, 300, status);
  }

  /**
   * Handle job errors and retries
   */
  async handleJobError(job, msg, error) {
    job.attempts = (job.attempts || 0) + 1;
    job.lastError = error.message;
    job.lastAttemptAt = new Date().toISOString();

    if (job.attempts < job.maxAttempts) {
      // Retry the job
      console.log(
        `🔄 Retrying job ${job.id} (attempt ${job.attempts}/${job.maxAttempts})`
      );

      // Wait before retry (exponential backoff)
      const delay = Math.pow(2, job.attempts) * 1000;
      setTimeout(async () => {
        await this.channel.sendToQueue(
          msg.fields.routingKey,
          Buffer.from(JSON.stringify(job)),
          { persistent: true }
        );
      }, delay);

      await this.updateJobStatus(job.id, "retrying");
    } else {
      // Max retries reached, mark as failed
      console.error(
        `💀 Job ${job.id} failed permanently after ${job.attempts} attempts`
      );
      await this.updateJobStatus(job.id, "failed", { error: error.message });
    }

    this.channel.ack(msg);
  }

  /**
   * Build analysis query based on type
   */
  buildAnalysisQuery(analysisType, parameters) {
    switch (analysisType) {
      case "probable_cause":
        return `Analyze the evidence for probable cause determination. Consider: ${JSON.stringify(
          parameters
        )}`;
      case "evidence_strength":
        return `Evaluate the strength of evidence in this case. Focus on: ${JSON.stringify(
          parameters
        )}`;
      case "case_timeline":
        return `Construct a comprehensive timeline of events based on available evidence.`;
      case "witness_credibility":
        return `Assess witness credibility based on statements and corroborating evidence.`;
      default:
        return `Perform legal analysis of type: ${analysisType}`;
    }
  }

  /**
   * Build report query based on type
   */
  buildReportQuery(reportType, parameters) {
    switch (reportType) {
      case "prosecution_memo":
        return `Generate a comprehensive prosecution memorandum including case summary, evidence analysis, and recommended charges.`;
      case "evidence_summary":
        return `Create a detailed evidence summary report with chain of custody and relevance analysis.`;
      case "case_strategy":
        return `Develop a prosecution strategy report with strengths, weaknesses, and recommendations.`;
      default:
        return `Generate ${reportType} report based on case evidence and analysis.`;
    }
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId) {
    const job = await this.redis.get(`job:${jobId}`);
    return job ? JSON.parse(job) : null;
  }

  /**
   * Get queue statistics
   */
  async getQueueStats() {
    const stats = {};

    for (const [name, queueName] of Object.entries(this.queues)) {
      try {
        const queueInfo = await this.channel.checkQueue(queueName);
        stats[name] = {
          messageCount: queueInfo.messageCount,
          consumerCount: queueInfo.consumerCount,
        };
      } catch (error) {
        stats[name] = { error: error.message };
      }
    }

    return stats;
  }

  /**
   * Health check
   */
  async healthCheck() {
    try {
      const queueStats = await this.getQueueStats();
      const redisInfo = await this.redis.info();

      return {
        status: "healthy",
        connection: this.isConnected,
        queues: queueStats,
        redis: "connected",
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    if (this.channel) {
      await this.channel.close();
    }
    if (this.connection) {
      await this.connection.close();
    }
    if (this.redis) {
      await this.redis.quit();
    }
    console.log("🧹 Event Streaming Service cleaned up");
  }
}

// Export singleton instance
export const eventStreamingService = new EventStreamingService();
export default eventStreamingService;
