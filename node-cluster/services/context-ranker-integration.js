#!/usr/bin/env node

/**
 * Context Ranker Integration for Node.js Cluster
 * 
 * Provides context ranking services to the cluster workers
 * Integrates with the main context-ranker service
 */

import { contextRanker, formatContextForAI, getContextStatistics } from '../../src/lib/services/context-ranker.js';
import { Worker } from 'worker_threads';
import { performance } from 'perf_hooks';

class ContextRankerWorkerService {
    constructor() {
        this.workers = new Map();
        this.requestQueue = [];
        this.activeRequests = 0;
        this.maxConcurrentRequests = 4;
        this.metrics = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageProcessingTime: 0,
            cacheHitRate: 0
        };
    }

    /**
     * Initialize the service
     */
    async initialize() {
        console.log('🚀 Initializing Context Ranker Worker Service...');
        
        // Test connection to dependencies
        await this.testConnections();
        
        console.log('✅ Context Ranker Worker Service initialized');
        return true;
    }

    /**
     * Test connections to required services
     */
    async testConnections() {
        try {
            // Test Ollama connection
            const response = await fetch('http://localhost:11434/api/tags');
            if (!response.ok) {
                throw new Error('Ollama not accessible');
            }
            console.log('✅ Ollama connection verified');

            // Test database connection (assume it's available)
            console.log('✅ Database connection assumed available');

        } catch (error) {
            console.error('❌ Connection test failed:', error.message);
            throw error;
        }
    }

    /**
     * Process context ranking request
     */
    async rankContext(query, options = {}) {
        const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const startTime = performance.now();

        console.log(`📝 Processing context ranking request ${requestId}: "${query.substring(0, 50)}..."`);

        try {
            this.metrics.totalRequests++;
            this.activeRequests++;

            // Use the imported context ranker
            const result = await contextRanker.context_ranker(query, options);
            
            // Format for AI consumption
            const formattedContext = formatContextForAI(result);
            const stats = getContextStatistics(result);

            const processingTime = performance.now() - startTime;
            
            // Update metrics
            this.metrics.successfulRequests++;
            this.updateAverageProcessingTime(processingTime);

            console.log(`✅ Context ranking completed for ${requestId}: ${result.contexts.length} contexts, ${processingTime.toFixed(2)}ms`);

            return {
                success: true,
                requestId,
                query,
                result,
                formattedContext,
                stats,
                processingTime,
                metadata: {
                    topK: options.topK || 5,
                    model: options.embeddingModel || 'nomic-embed-text',
                    timestamp: new Date().toISOString()
                }
            };

        } catch (error) {
            this.metrics.failedRequests++;
            console.error(`❌ Context ranking failed for ${requestId}:`, error);

            return {
                success: false,
                requestId,
                query,
                error: error.message,
                processingTime: performance.now() - startTime
            };
        } finally {
            this.activeRequests--;
            this.processQueue();
        }
    }

    /**
     * Batch process multiple queries
     */
    async batchRankContext(queries, options = {}) {
        console.log(`📋 Processing batch context ranking: ${queries.length} queries`);
        
        const batchId = `batch_${Date.now()}`;
        const startTime = performance.now();

        try {
            const promises = queries.map((query, index) => 
                this.rankContext(query, {
                    ...options,
                    batchId,
                    batchIndex: index
                })
            );

            const results = await Promise.all(promises);
            const processingTime = performance.now() - startTime;

            const successful = results.filter(r => r.success).length;
            const failed = results.length - successful;

            console.log(`✅ Batch context ranking completed: ${successful} successful, ${failed} failed, ${processingTime.toFixed(2)}ms total`);

            return {
                batchId,
                totalQueries: queries.length,
                successful,
                failed,
                results,
                processingTime,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error(`❌ Batch context ranking failed:`, error);
            throw error;
        }
    }

    /**
     * Get service metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            activeRequests: this.activeRequests,
            queueLength: this.requestQueue.length,
            successRate: this.metrics.totalRequests > 0 
                ? (this.metrics.successfulRequests / this.metrics.totalRequests * 100).toFixed(2) + '%'
                : '0%',
            uptime: process.uptime(),
            memoryUsage: process.memoryUsage(),
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Update average processing time
     */
    updateAverageProcessingTime(newTime) {
        const totalRequests = this.metrics.totalRequests;
        this.metrics.averageProcessingTime = 
            (this.metrics.averageProcessingTime * (totalRequests - 1) + newTime) / totalRequests;
    }

    /**
     * Process queued requests
     */
    processQueue() {
        while (this.requestQueue.length > 0 && this.activeRequests < this.maxConcurrentRequests) {
            const { resolve, reject, query, options } = this.requestQueue.shift();
            
            this.rankContext(query, options)
                .then(resolve)
                .catch(reject);
        }
    }

    /**
     * Queue a request if at capacity
     */
    queueRequest(query, options = {}) {
        return new Promise((resolve, reject) => {
            if (this.activeRequests < this.maxConcurrentRequests) {
                this.rankContext(query, options)
                    .then(resolve)
                    .catch(reject);
            } else {
                this.requestQueue.push({ resolve, reject, query, options });
                console.log(`📤 Request queued. Queue length: ${this.requestQueue.length}`);
            }
        });
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const testQuery = "What is embezzlement?";
            const result = await this.rankContext(testQuery, { topK: 1 });
            
            return {
                status: result.success ? 'healthy' : 'degraded',
                timestamp: new Date().toISOString(),
                testResult: result.success,
                metrics: this.getMetrics(),
                dependencies: {
                    ollama: await this.checkOllamaHealth(),
                    database: 'assumed-healthy' // Would check actual DB in production
                }
            };
        } catch (error) {
            return {
                status: 'unhealthy',
                timestamp: new Date().toISOString(),
                error: error.message,
                metrics: this.getMetrics()
            };
        }
    }

    /**
     * Check Ollama health
     */
    async checkOllamaHealth() {
        try {
            const response = await fetch('http://localhost:11434/api/tags', {
                timeout: 5000
            });
            return response.ok ? 'healthy' : 'degraded';
        } catch {
            return 'unhealthy';
        }
    }

    /**
     * Cleanup resources
     */
    async shutdown() {
        console.log('🛑 Shutting down Context Ranker Worker Service...');
        
        // Wait for active requests to complete
        while (this.activeRequests > 0) {
            console.log(`⏳ Waiting for ${this.activeRequests} active requests to complete...`);
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Clear cache
        contextRanker.clearCache();
        
        console.log('✅ Context Ranker Worker Service shutdown complete');
    }
}

// Export for use in cluster workers
export default ContextRankerWorkerService;
export { ContextRankerWorkerService };

// If run directly, start the service
if (import.meta.url === `file://${process.argv[1]}`) {
    const service = new ContextRankerWorkerService();
    
    // Handle process signals
    process.on('SIGTERM', async () => {
        await service.shutdown();
        process.exit(0);
    });

    process.on('SIGINT', async () => {
        await service.shutdown();
        process.exit(0);
    });

    // Initialize service
    service.initialize()
        .then(() => {
            console.log('🚀 Context Ranker Worker Service ready');
            
            // Set up periodic health checks
            setInterval(async () => {
                const health = await service.healthCheck();
                console.log(`💓 Health: ${health.status}`);
            }, 30000);
        })
        .catch(error => {
            console.error('💥 Failed to initialize Context Ranker Worker Service:', error);
            process.exit(1);
        });
}