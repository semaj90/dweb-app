/**
 * Job Scheduler Service
 * Automatically queues background jobs for document processing, embedding generation, etc.
 */

import { rabbitMQService } from '$lib/services/rabbitmq-connection';
import { QUEUES, ROUTING_KEYS } from '$lib/config/rabbitmq-config';
import type { EmbeddingJobPayload } from '$lib/workers/rabbitmq-embedding-worker';

export interface ScheduleOptions {
  priority?: number;
  delay?: number; // milliseconds
  correlationId?: string;
  maxRetries?: number;
  createdBy?: string;
}

class JobSchedulerService {
  private isInitialized: boolean = false;

  /**
   * Initialize the job scheduler
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      // Ensure RabbitMQ connection is established
      await rabbitMQService.connect();
      this.isInitialized = true;
      console.log('✅ Job scheduler initialized');
    } catch (error) {
      console.error('❌ Failed to initialize job scheduler:', error);
      throw error;
    }
  }

  /**
   * Schedule embedding generation for a document
   */
  async scheduleDocumentEmbedding(
    documentId: string,
    embeddingTypes: ('content' | 'title' | 'summary')[] = ['content'],
    textContent?: string,
    options: ScheduleOptions = {}
  ): Promise<string[]> {
    await this.ensureInitialized();

    const jobIds: string[] = [];

    for (const embeddingType of embeddingTypes) {
      const payload: EmbeddingJobPayload = {
        entity_type: 'document',
        entity_id: documentId,
        embedding_type: embeddingType,
        text_content: textContent,
        update_vector_store: true
      };

      const jobType = `generate_document_${embeddingType}_embedding`;

      try {
        const jobId = await this.scheduleJob(
          QUEUES.DOCUMENT_EMBEDDING,
          jobType,
          payload,
          options
        );
        jobIds.push(jobId);
        console.log(`📅 Scheduled ${embeddingType} embedding for document ${documentId}: ${jobId}`);
      } catch (error) {
        console.error(`❌ Failed to schedule ${embeddingType} embedding for document ${documentId}:`, error);
        throw error;
      }
    }

    return jobIds;
  }

  /**
   * Schedule embedding generation for a case
   */
  async scheduleCaseEmbedding(
    caseId: string,
    textContent?: string,
    options: ScheduleOptions = {}
  ): Promise<string> {
    await this.ensureInitialized();

    const payload: EmbeddingJobPayload = {
      entity_type: 'case',
      entity_id: caseId,
      text_content: textContent,
      update_vector_store: true
    };

    const jobType = 'generate_case_embedding';

    try {
      const jobId = await this.scheduleJob(
        QUEUES.CASE_EMBEDDING,
        jobType,
        payload,
        options
      );
      console.log(`📅 Scheduled case embedding for ${caseId}: ${jobId}`);
      return jobId;
    } catch (error) {
      console.error(`❌ Failed to schedule case embedding for ${caseId}:`, error);
      throw error;
    }
  }

  /**
   * Schedule embedding generation for document chunks
   */
  async scheduleChunkEmbedding(
    chunkId: string,
    textContent?: string,
    options: ScheduleOptions = {}
  ): Promise<string> {
    await this.ensureInitialized();

    const payload: EmbeddingJobPayload = {
      entity_type: 'chunk',
      entity_id: chunkId,
      text_content: textContent,
      update_vector_store: true
    };

    const jobType = 'generate_chunk_embedding';

    try {
      const jobId = await this.scheduleJob(
        QUEUES.DOCUMENT_EMBEDDING, // Chunks use same queue as documents
        jobType,
        payload,
        options
      );
      console.log(`📅 Scheduled chunk embedding for ${chunkId}: ${jobId}`);
      return jobId;
    } catch (error) {
      console.error(`❌ Failed to schedule chunk embedding for ${chunkId}:`, error);
      throw error;
    }
  }

  /**
   * Schedule document analysis job (AI summarization, entity extraction, etc.)
   */
  async scheduleDocumentAnalysis(
    documentId: string,
    analysisTypes: ('summarization' | 'entity_extraction' | 'classification')[] = ['summarization'],
    options: ScheduleOptions = {}
  ): Promise<string[]> {
    await this.ensureInitialized();

    const jobIds: string[] = [];

    for (const analysisType of analysisTypes) {
      let queueName: string;
      let routingKey: string;
      let jobType: string;

      switch (analysisType) {
        case 'summarization':
          queueName = QUEUES.AI_SUMMARIZATION;
          routingKey = ROUTING_KEYS.SUMMARIZE_CONTENT;
          jobType = 'summarize_document';
          break;
        case 'entity_extraction':
          queueName = QUEUES.AI_ENTITY_EXTRACTION;
          routingKey = ROUTING_KEYS.EXTRACT_ENTITIES;
          jobType = 'extract_document_entities';
          break;
        case 'classification':
          queueName = QUEUES.AI_CLASSIFICATION;
          routingKey = ROUTING_KEYS.CLASSIFY_DOCUMENT;
          jobType = 'classify_document';
          break;
        default:
          throw new Error(`Unsupported analysis type: ${analysisType}`);
      }

      const payload = {
        entity_type: 'document',
        entity_id: documentId,
        analysis_type: analysisType
      };

      try {
        const jobId = await this.scheduleJob(queueName, jobType, payload, options);
        jobIds.push(jobId);
        console.log(`📅 Scheduled ${analysisType} for document ${documentId}: ${jobId}`);
      } catch (error) {
        console.error(`❌ Failed to schedule ${analysisType} for document ${documentId}:`, error);
        throw error;
      }
    }

    return jobIds;
  }

  /**
   * Schedule case similarity analysis
   */
  async scheduleCaseSimilarityAnalysis(
    caseId: string,
    options: ScheduleOptions = {}
  ): Promise<string> {
    await this.ensureInitialized();

    const payload = {
      entity_type: 'case',
      entity_id: caseId,
      analysis_type: 'similarity'
    };

    const jobType = 'analyze_case_similarity';

    try {
      const jobId = await this.scheduleJob(
        QUEUES.CASE_SIMILARITY,
        jobType,
        payload,
        options
      );
      console.log(`📅 Scheduled case similarity analysis for ${caseId}: ${jobId}`);
      return jobId;
    } catch (error) {
      console.error(`❌ Failed to schedule case similarity analysis for ${caseId}:`, error);
      throw error;
    }
  }

  /**
   * Schedule vector index update
   */
  async scheduleVectorIndexUpdate(
    entityType: 'document' | 'case' | 'chunk',
    entityId: string,
    options: ScheduleOptions = {}
  ): Promise<string> {
    await this.ensureInitialized();

    const payload = {
      entity_type: entityType,
      entity_id: entityId,
      operation: 'index_update'
    };

    const jobType = `update_${entityType}_vector_index`;

    try {
      const jobId = await this.scheduleJob(
        QUEUES.VECTOR_SEARCH_UPDATE,
        jobType,
        payload,
        {
          ...options,
          priority: options.priority || 7 // Higher priority for index updates
        }
      );
      console.log(`📅 Scheduled vector index update for ${entityType} ${entityId}: ${jobId}`);
      return jobId;
    } catch (error) {
      console.error(`❌ Failed to schedule vector index update for ${entityType} ${entityId}:`, error);
      throw error;
    }
  }

  /**
   * Schedule complete document processing pipeline
   * This will queue multiple jobs in the correct order with dependencies
   */
  async scheduleCompleteDocumentProcessing(
    documentId: string,
    processingOptions: {
      generateEmbeddings?: boolean;
      embeddingTypes?: ('content' | 'title' | 'summary')[];
      performAnalysis?: boolean;
      analysisTypes?: ('summarization' | 'entity_extraction' | 'classification')[];
      updateVectorIndex?: boolean;
      textContent?: string;
    } = {},
    options: ScheduleOptions = {}
  ): Promise<{
    embeddingJobs: string[];
    analysisJobs: string[];
    indexUpdateJob?: string;
  }> {
    await this.ensureInitialized();

    const {
      generateEmbeddings = true,
      embeddingTypes = ['content', 'title'],
      performAnalysis = true,
      analysisTypes = ['summarization', 'entity_extraction'],
      updateVectorIndex = true,
      textContent
    } = processingOptions;

    const results = {
      embeddingJobs: [] as string[],
      analysisJobs: [] as string[],
      indexUpdateJob: undefined as string | undefined
    };

    try {
      // Step 1: Generate embeddings (highest priority)
      if (generateEmbeddings) {
        results.embeddingJobs = await this.scheduleDocumentEmbedding(
          documentId,
          embeddingTypes,
          textContent,
          { ...options, priority: 8 }
        );
      }

      // Step 2: Perform AI analysis (medium priority, after embeddings)
      if (performAnalysis) {
        // Add small delay to ensure embeddings are processed first
        results.analysisJobs = await this.scheduleDocumentAnalysis(
          documentId,
          analysisTypes,
          { ...options, priority: 6, delay: 2000 }
        );
      }

      // Step 3: Update vector index (lower priority, after everything else)
      if (updateVectorIndex) {
        results.indexUpdateJob = await this.scheduleVectorIndexUpdate(
          'document',
          documentId,
          { ...options, priority: 5, delay: 5000 }
        );
      }

      console.log(`📋 Scheduled complete processing pipeline for document ${documentId}`);
      console.log(`   - Embeddings: ${results.embeddingJobs.length} jobs`);
      console.log(`   - Analysis: ${results.analysisJobs.length} jobs`);
      console.log(`   - Index update: ${results.indexUpdateJob ? '1 job' : 'skipped'}`);

      return results;

    } catch (error) {
      console.error(`❌ Failed to schedule complete document processing for ${documentId}:`, error);
      throw error;
    }
  }

  /**
   * Generic job scheduling method
   */
  private async scheduleJob(
    queueName: string,
    jobType: string,
    payload: any,
    options: ScheduleOptions = {}
  ): Promise<string> {
    const {
      priority = 5,
      delay = 0,
      correlationId,
      maxRetries = 3,
      createdBy
    } = options;

    // Add delay if specified
    if (delay > 0) {
      await new Promise(resolve => setTimeout(resolve, delay));
    }

    // Publish job to RabbitMQ
    const jobId = await rabbitMQService.publishJob(
      queueName,
      jobType,
      payload,
      {
        priority,
        correlationId,
        maxRetries,
        createdBy,
        metadata: {
          scheduled_at: new Date().toISOString(),
          queue: queueName
        }
      }
    );

    return jobId;
  }

  /**
   * Ensure the scheduler is initialized
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }
  }

  /**
   * Get scheduler statistics
   */
  getStats(): {
    initialized: boolean;
    rabbitmq_connected: boolean;
  } {
    return {
      initialized: this.isInitialized,
      rabbitmq_connected: rabbitMQService['isConnected'] || false
    };
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    details: {
      scheduler_initialized: boolean;
      rabbitmq_available: boolean;
    };
  }> {
    const rabbitHealth = await rabbitMQService.healthCheck();
    const isHealthy = this.isInitialized && rabbitHealth.connected;

    return {
      status: isHealthy ? 'healthy' : 'unhealthy',
      details: {
        scheduler_initialized: this.isInitialized,
        rabbitmq_available: rabbitHealth.connected
      }
    };
  }
}

// Export singleton instance
export const jobScheduler = new JobSchedulerService();
export default jobScheduler;