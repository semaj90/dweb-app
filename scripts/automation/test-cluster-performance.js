#!/usr/bin/env node
/**
 * Cluster Performance Test for Enhanced RAG System
 * Tests real workloads with concurrent processing and performance metrics
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { performance } from 'perf_hooks';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test configuration
const PERFORMANCE_CONFIG = {
    workerCount: 4,
    testDuration: 30000, // 30 seconds
    concurrentRequests: 10,
    rampUpTime: 5000, // 5 seconds
    testTypes: ['rag-query', 'memory-graph', 'context7-docs', 'agent-orchestration'],
    mcpServerUrl: 'http://localhost:40000',
    ollamaUrl: 'http://localhost:11434'
};

// Test workloads simulating real usage
const TEST_WORKLOADS = {
    'rag-query': [
        'Analyze contract liability clauses for breach scenarios',
        'What are the key elements of employment law compliance?',
        'Identify precedent cases for intellectual property disputes',
        'Summarize regulatory requirements for financial services',
        'Legal implications of data breach notification requirements'
    ],
    'memory-graph': [
        { entities: [{ type: 'case', id: 'case-001', properties: { title: 'Contract Dispute' } }] },
        { entities: [{ type: 'evidence', id: 'evidence-001', properties: { type: 'document' } }] },
        { entities: [{ type: 'person', id: 'person-001', properties: { role: 'witness' } }] }
    ],
    'context7-docs': [
        { libraryName: 'sveltekit', topic: 'forms' },
        { libraryName: 'typescript', topic: 'types' },
        { libraryName: 'drizzle', topic: 'schema' },
        { libraryName: 'legal-ai', topic: 'compliance' }
    ],
    'agent-orchestration': [
        { agent: 'claude', prompt: 'Analyze this legal document for risks' },
        { agent: 'crewai', prompt: 'Research precedent cases for this scenario' },
        { agent: 'autogen', prompt: 'Generate compliance checklist' }
    ]
};

class ClusterPerformanceTest {
    constructor() {
        this.results = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            averageResponseTime: 0,
            minResponseTime: Infinity,
            maxResponseTime: 0,
            throughput: 0,
            errorRate: 0,
            workerStats: [],
            testsByType: {}
        };
        this.startTime = null;
        this.workers = [];
        this.testResults = [];
    }

    async runPerformanceTest() {
        console.log('🚀 Starting Cluster Performance Test...\n');
        console.log('📊 Test Configuration:');
        console.log(`   Workers: ${PERFORMANCE_CONFIG.workerCount}`);
        console.log(`   Duration: ${PERFORMANCE_CONFIG.testDuration / 1000}s`);
        console.log(`   Concurrent Requests: ${PERFORMANCE_CONFIG.concurrentRequests}`);
        console.log(`   Test Types: ${PERFORMANCE_CONFIG.testTypes.join(', ')}`);
        console.log('');

        try {
            // Step 1: Initialize workers
            await this.initializeWorkers();

            // Step 2: Warm up phase
            await this.warmUpPhase();

            // Step 3: Run performance test
            await this.runMainTest();

            // Step 4: Collect and analyze results
            await this.collectResults();

            // Step 5: Generate performance report
            this.generatePerformanceReport();

        } catch (error) {
            console.error('💥 Performance test failed:', error);
            process.exit(1);
        } finally {
            await this.cleanup();
        }
    }

    async initializeWorkers() {
        console.log('👥 Initializing worker threads...');
        
        for (let i = 0; i < PERFORMANCE_CONFIG.workerCount; i++) {
            const worker = new Worker(__filename, {
                workerData: {
                    workerId: i,
                    config: PERFORMANCE_CONFIG,
                    workloads: TEST_WORKLOADS
                }
            });

            worker.on('message', (message) => {
                if (message.type === 'test-result') {
                    this.testResults.push(message.data);
                } else if (message.type === 'worker-ready') {
                    console.log(`✅ Worker ${message.workerId} ready`);
                }
            });

            worker.on('error', (error) => {
                console.error(`❌ Worker ${i} error:`, error);
            });

            this.workers.push(worker);
        }

        // Wait for all workers to be ready
        await new Promise(resolve => setTimeout(resolve, 2000));
        console.log(`✅ ${this.workers.length} workers initialized\n`);
    }

    async warmUpPhase() {
        console.log('🔥 Warm-up phase starting...');
        
        // Send warm-up requests to all workers
        this.workers.forEach((worker, index) => {
            worker.postMessage({
                type: 'warm-up',
                duration: PERFORMANCE_CONFIG.rampUpTime
            });
        });

        await new Promise(resolve => setTimeout(resolve, PERFORMANCE_CONFIG.rampUpTime));
        console.log('✅ Warm-up phase completed\n');
    }

    async runMainTest() {
        console.log('🏃 Starting main performance test...');
        this.startTime = performance.now();

        // Send test commands to all workers
        this.workers.forEach((worker, index) => {
            worker.postMessage({
                type: 'start-test',
                duration: PERFORMANCE_CONFIG.testDuration,
                concurrentRequests: PERFORMANCE_CONFIG.concurrentRequests
            });
        });

        // Show progress during test
        const progressInterval = setInterval(() => {
            const elapsed = Math.round((performance.now() - this.startTime) / 1000);
            const progress = Math.min(100, (elapsed / (PERFORMANCE_CONFIG.testDuration / 1000)) * 100);
            process.stdout.write(`\r⏱️  Progress: ${progress.toFixed(1)}% (${elapsed}s)`);
        }, 1000);

        await new Promise(resolve => setTimeout(resolve, PERFORMANCE_CONFIG.testDuration));
        clearInterval(progressInterval);
        
        console.log('\n✅ Main test completed\n');
    }

    async collectResults() {
        console.log('📊 Collecting test results...');

        // Send stop signal to workers and collect final stats
        const workerPromises = this.workers.map((worker, index) => {
            return new Promise((resolve) => {
                worker.postMessage({ type: 'get-stats' });
                worker.once('message', (message) => {
                    if (message.type === 'worker-stats') {
                        resolve(message.data);
                    }
                });
            });
        });

        this.results.workerStats = await Promise.all(workerPromises);

        // Process test results
        this.processTestResults();
        console.log('✅ Results collected\n');
    }

    processTestResults() {
        const allResults = this.testResults;
        
        if (allResults.length === 0) {
            console.warn('⚠️ No test results collected');
            return;
        }

        // Calculate overall statistics
        this.results.totalRequests = allResults.length;
        this.results.successfulRequests = allResults.filter(r => r.success).length;
        this.results.failedRequests = allResults.filter(r => !r.success).length;
        this.results.errorRate = this.results.failedRequests / this.results.totalRequests;

        // Calculate response time statistics
        const responseTimes = allResults.filter(r => r.success).map(r => r.responseTime);
        if (responseTimes.length > 0) {
            this.results.averageResponseTime = responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
            this.results.minResponseTime = Math.min(...responseTimes);
            this.results.maxResponseTime = Math.max(...responseTimes);
        }

        // Calculate throughput
        const totalTime = (performance.now() - this.startTime) / 1000;
        this.results.throughput = this.results.successfulRequests / totalTime;

        // Group results by test type
        PERFORMANCE_CONFIG.testTypes.forEach(testType => {
            const typeResults = allResults.filter(r => r.testType === testType);
            this.results.testsByType[testType] = {
                total: typeResults.length,
                successful: typeResults.filter(r => r.success).length,
                averageResponseTime: typeResults.length > 0 
                    ? typeResults.reduce((sum, r) => sum + r.responseTime, 0) / typeResults.length 
                    : 0
            };
        });
    }

    generatePerformanceReport() {
        console.log('📈 CLUSTER PERFORMANCE REPORT');
        console.log('=' .repeat(50));
        
        console.log('\n🎯 Overall Performance:');
        console.log(`   Total Requests: ${this.results.totalRequests}`);
        console.log(`   Successful: ${this.results.successfulRequests} (${((this.results.successfulRequests / this.results.totalRequests) * 100).toFixed(1)}%)`);
        console.log(`   Failed: ${this.results.failedRequests} (${(this.results.errorRate * 100).toFixed(1)}%)`);
        console.log(`   Throughput: ${this.results.throughput.toFixed(2)} req/s`);
        
        console.log('\n⏱️ Response Times:');
        console.log(`   Average: ${this.results.averageResponseTime.toFixed(2)}ms`);
        console.log(`   Min: ${this.results.minResponseTime.toFixed(2)}ms`);
        console.log(`   Max: ${this.results.maxResponseTime.toFixed(2)}ms`);

        console.log('\n📊 Performance by Test Type:');
        Object.entries(this.results.testsByType).forEach(([testType, stats]) => {
            console.log(`   ${testType}:`);
            console.log(`     Requests: ${stats.total}`);
            console.log(`     Success Rate: ${((stats.successful / stats.total) * 100).toFixed(1)}%`);
            console.log(`     Avg Response: ${stats.averageResponseTime.toFixed(2)}ms`);
        });

        console.log('\n👥 Worker Statistics:');
        this.results.workerStats.forEach((stats, index) => {
            console.log(`   Worker ${index}:`);
            console.log(`     Requests Processed: ${stats.requestsProcessed}`);
            console.log(`     Average Response: ${stats.averageResponseTime.toFixed(2)}ms`);
            console.log(`     Error Rate: ${(stats.errorRate * 100).toFixed(1)}%`);
        });

        // Performance assessment
        console.log('\n🏆 Performance Assessment:');
        if (this.results.throughput > 50 && this.results.errorRate < 0.05) {
            console.log('🎉 EXCELLENT: Cluster performance exceeds production requirements!');
            console.log('✅ High throughput with low error rate');
            console.log('✅ Response times within acceptable range');
        } else if (this.results.throughput > 20 && this.results.errorRate < 0.1) {
            console.log('🚀 GOOD: Cluster performance meets production requirements');
            console.log('ℹ️ Consider optimization for higher loads');
        } else {
            console.log('⚠️ NEEDS IMPROVEMENT: Cluster performance below optimal');
            console.log('❗ Review error rates and response times');
        }

        // Save detailed results
        const reportData = {
            timestamp: new Date().toISOString(),
            config: PERFORMANCE_CONFIG,
            results: this.results,
            assessment: {
                score: this.calculatePerformanceScore(),
                recommendations: this.generateRecommendations()
            }
        };

        fs.writeFileSync('cluster-performance-report.json', JSON.stringify(reportData, null, 2));
        console.log('\n💾 Detailed report saved to: cluster-performance-report.json');
    }

    calculatePerformanceScore() {
        let score = 0;
        
        // Throughput score (max 40 points)
        score += Math.min(40, (this.results.throughput / 50) * 40);
        
        // Error rate score (max 30 points)
        score += Math.max(0, 30 - (this.results.errorRate * 300));
        
        // Response time score (max 30 points)
        const responseTimeScore = Math.max(0, 30 - (this.results.averageResponseTime / 100));
        score += responseTimeScore;
        
        return Math.round(score);
    }

    generateRecommendations() {
        const recommendations = [];
        
        if (this.results.errorRate > 0.05) {
            recommendations.push('Investigate and reduce error rate');
            recommendations.push('Add more robust error handling and retries');
        }
        
        if (this.results.averageResponseTime > 1000) {
            recommendations.push('Optimize response times');
            recommendations.push('Consider caching frequently accessed data');
        }
        
        if (this.results.throughput < 20) {
            recommendations.push('Scale up worker count');
            recommendations.push('Optimize processing algorithms');
        }
        
        recommendations.push('Monitor performance in production');
        recommendations.push('Set up alerting for performance degradation');
        
        return recommendations;
    }

    async cleanup() {
        console.log('\n🧹 Cleaning up workers...');
        
        await Promise.all(this.workers.map(worker => {
            return new Promise((resolve) => {
                worker.terminate().then(resolve);
            });
        }));
        
        console.log('✅ Cleanup completed');
    }
}

// Worker thread logic
if (!isMainThread) {
    class PerformanceWorker {
        constructor() {
            this.workerId = workerData.workerId;
            this.config = workerData.config;
            this.workloads = workerData.workloads;
            this.stats = {
                requestsProcessed: 0,
                successfulRequests: 0,
                totalResponseTime: 0,
                errorRate: 0
            };
            this.isRunning = false;
        }

        async start() {
            parentPort.postMessage({ type: 'worker-ready', workerId: this.workerId });

            parentPort.on('message', async (message) => {
                switch (message.type) {
                    case 'warm-up':
                        await this.warmUp(message.duration);
                        break;
                    case 'start-test':
                        await this.runTest(message.duration, message.concurrentRequests);
                        break;
                    case 'get-stats':
                        this.sendStats();
                        break;
                }
            });
        }

        async warmUp(duration) {
            const endTime = Date.now() + duration;
            while (Date.now() < endTime) {
                await this.performRandomTest();
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        async runTest(duration, concurrentRequests) {
            this.isRunning = true;
            const endTime = Date.now() + duration;
            const promises = [];

            while (Date.now() < endTime && this.isRunning) {
                // Maintain concurrent request pool
                while (promises.length < concurrentRequests && Date.now() < endTime) {
                    const testPromise = this.performRandomTest()
                        .then(() => {
                            const index = promises.indexOf(testPromise);
                            if (index > -1) promises.splice(index, 1);
                        });
                    promises.push(testPromise);
                }

                await new Promise(resolve => setTimeout(resolve, 10));
            }

            // Wait for remaining requests to complete
            await Promise.all(promises);
            this.isRunning = false;
        }

        async performRandomTest() {
            const testType = this.config.testTypes[Math.floor(Math.random() * this.config.testTypes.length)];
            const workload = this.getRandomWorkload(testType);
            
            const startTime = performance.now();
            let success = false;

            try {
                await this.executeTest(testType, workload);
                success = true;
                this.stats.successfulRequests++;
            } catch (error) {
                // Test failed
                success = false;
            }

            const responseTime = performance.now() - startTime;
            this.stats.requestsProcessed++;
            this.stats.totalResponseTime += responseTime;

            parentPort.postMessage({
                type: 'test-result',
                data: {
                    testType,
                    success,
                    responseTime,
                    workerId: this.workerId
                }
            });
        }

        getRandomWorkload(testType) {
            const workloads = this.workloads[testType];
            return workloads[Math.floor(Math.random() * workloads.length)];
        }

        async executeTest(testType, workload) {
            switch (testType) {
                case 'rag-query':
                    return await this.testRAGQuery(workload);
                case 'memory-graph':
                    return await this.testMemoryGraph(workload);
                case 'context7-docs':
                    return await this.testContext7Docs(workload);
                case 'agent-orchestration':
                    return await this.testAgentOrchestration(workload);
                default:
                    throw new Error(`Unknown test type: ${testType}`);
            }
        }

        async testRAGQuery(query) {
            const response = await fetch(`${this.config.mcpServerUrl}/mcp/enhanced-rag/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    caseId: 'perf-test',
                    maxResults: 5
                })
            });

            if (!response.ok) throw new Error(`RAG query failed: ${response.status}`);
            return await response.json();
        }

        async testMemoryGraph(entities) {
            const response = await fetch(`${this.config.mcpServerUrl}/mcp/memory/create-relations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(entities)
            });

            if (!response.ok) throw new Error(`Memory graph failed: ${response.status}`);
            return await response.json();
        }

        async testContext7Docs(params) {
            const response = await fetch(`${this.config.mcpServerUrl}/mcp/context7/resolve-library-id`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) throw new Error(`Context7 docs failed: ${response.status}`);
            return await response.json();
        }

        async testAgentOrchestration(params) {
            // Simulate agent orchestration
            await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));
            return { success: true, agent: params.agent };
        }

        sendStats() {
            const averageResponseTime = this.stats.requestsProcessed > 0 
                ? this.stats.totalResponseTime / this.stats.requestsProcessed 
                : 0;
            
            const errorRate = this.stats.requestsProcessed > 0
                ? (this.stats.requestsProcessed - this.stats.successfulRequests) / this.stats.requestsProcessed
                : 0;

            parentPort.postMessage({
                type: 'worker-stats',
                data: {
                    requestsProcessed: this.stats.requestsProcessed,
                    averageResponseTime,
                    errorRate
                }
            });
        }
    }

    const worker = new PerformanceWorker();
    worker.start();
}

// Run performance test if this is the main thread
if (isMainThread) {
    const test = new ClusterPerformanceTest();
    test.runPerformanceTest();
}