// Node.js BullMQ Producer Service
import { Queue, Worker } from 'bullmq';
import Redis from 'ioredis';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Redis connection for Windows
const redisConnection = new Redis({
    host: 'localhost',
    port: 6379,
    maxRetriesPerRequest: 3,
    retryDelayOnFailover: 100,
    enableOfflineQueue: false,
    lazyConnect: true,
});

// Queue configuration
const legalProcessorQueue = new Queue('legal-processor', {
    connection: redisConnection,
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

// Job types for legal processing
export class LegalJobProducer {
    constructor() {
        this.queue = legalProcessorQueue;
    }

    async enqueueDocumentAnalysis(data) {
        return await this.queue.add('document_analysis', {
            ...data,
            priority: data.priority || 5,
            timestamp: Date.now(),
        }, {
            priority: data.priority || 5,
            delay: data.delay || 0,
        });
    }

    async enqueueRAGSearch(data) {
        return await this.queue.add('rag_search', {
            ...data,
            useGPU: data.useGPU ?? true,
            timestamp: Date.now(),
        }, {
            priority: data.priority || 3,
        });
    }

    async enqueueBulkEmbedding(data) {
        return await this.queue.add('bulk_embedding', {
            ...data,
            batchSize: data.batchSize || 512,
            timestamp: Date.now(),
        }, {
            priority: data.priority || 7,
        });
    }

    async enqueueRecommendationTraining(data) {
        return await this.queue.add('recommendation_training', {
            ...data,
            somParams: data.somParams || { width: 20, height: 20 },
            timestamp: Date.now(),
        }, {
            priority: data.priority || 8,
        });
    }

    async getQueueStats() {
        const waiting = await this.queue.getWaiting();
        const active = await this.queue.getActive();
        const completed = await this.queue.getCompleted();
        const failed = await this.queue.getFailed();

        return {
            waiting: waiting.length,
            active: active.length,
            completed: completed.length,
            failed: failed.length,
            total: waiting.length + active.length,
        };
    }

    async getJob(jobId) {
        return await this.queue.getJob(jobId);
    }

    async removeJob(jobId) {
        const job = await this.getJob(jobId);
        if (job) {
            await job.remove();
            return true;
        }
        return false;
    }

    async pauseQueue() {
        await this.queue.pause();
    }

    async resumeQueue() {
        await this.queue.resume();
    }

    async cleanQueue(grace = 5000) {
        await this.queue.clean(grace, 100);
    }
}

// Results processor
export class LegalResultsProcessor {
    constructor() {
        this.resultsQueue = new Queue('legal-results', {
            connection: redisConnection,
        });
    }

    async processResults(data) {
        return await this.resultsQueue.add('process_results', data);
    }
}

// Queue monitoring and metrics
export class QueueMonitor {
    constructor() {
        this.metrics = {
            processedJobs: 0,
            failedJobs: 0,
            averageProcessingTime: 0,
            queueSizes: {},
        };
    }

    startMonitoring() {
        setInterval(async () => {
            try {
                const stats = await legalProcessorQueue.getJobCounts();
                this.metrics.queueSizes = stats;
                
                // Store metrics in Redis for Go service to access
                await redisConnection.setex(
                    'queue:metrics',
                    60,
                    JSON.stringify(this.metrics)
                );
            } catch (error) {
                console.error('Queue monitoring error:', error);
            }
        }, 10000); // Update every 10 seconds
    }

    getMetrics() {
        return this.metrics;
    }
}

// Export instances
export const jobProducer = new LegalJobProducer();
export const resultsProcessor = new LegalResultsProcessor();
export const queueMonitor = new QueueMonitor();

// Start monitoring
queueMonitor.startMonitoring();

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('Shutting down BullMQ producer...');
    await legalProcessorQueue.close();
    await redisConnection.disconnect();
    process.exit(0);
});
