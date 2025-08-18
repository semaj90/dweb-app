// orchestrator/health_check.js
// Health check utility for the GPU orchestrator system

import Redis from 'ioredis';
import amqp from 'amqplib';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const RABBIT_URL = process.env.RABBIT_URL || 'amqp://localhost';
const CUDA_WORKER_PATH = process.env.CUDA_WORKER_PATH || path.join(__dirname, '..', 'cuda-worker', 'cuda-worker.exe');

class HealthChecker {
    constructor() {
        this.results = {
            overall: 'unknown',
            timestamp: new Date().toISOString(),
            components: {}
        };
    }

    async checkRedis() {
        console.log('🔍 Checking Redis connection...');
        const redis = new Redis(REDIS_URL, {
            retryDelayOnFailover: 100,
            enableReadyCheck: false,
            maxRetriesPerRequest: 1,
            lazyConnect: true
        });

        try {
            const start = Date.now();
            const pong = await redis.ping();
            const latency = Date.now() - start;
            
            // Test basic operations
            await redis.set('health_check', 'ok', 'EX', 10);
            const value = await redis.get('health_check');
            
            this.results.components.redis = {
                status: 'healthy',
                latency: `${latency}ms`,
                operations: value === 'ok' ? 'working' : 'failed'
            };
            
            await redis.disconnect();
            return true;
        } catch (error) {
            this.results.components.redis = {
                status: 'unhealthy',
                error: error.message
            };
            await redis.disconnect();
            return false;
        }
    }

    async checkRabbitMQ() {
        console.log('🔍 Checking RabbitMQ connection...');
        try {
            const start = Date.now();
            const conn = await amqp.connect(RABBIT_URL);
            const ch = await conn.createChannel();
            const latency = Date.now() - start;
            
            // Test queue operations
            const testQueue = 'health_check_queue';
            await ch.assertQueue(testQueue, { durable: false });
            await ch.deleteQueue(testQueue);
            
            this.results.components.rabbitmq = {
                status: 'healthy',
                latency: `${latency}ms`,
                operations: 'working'
            };
            
            await ch.close();
            await conn.close();
            return true;
        } catch (error) {
            this.results.components.rabbitmq = {
                status: 'unhealthy',
                error: error.message
            };
            return false;
        }
    }

    async checkCudaWorker() {
        console.log('🔍 Checking CUDA worker...');
        
        if (!fs.existsSync(CUDA_WORKER_PATH)) {
            this.results.components.cuda_worker = {
                status: 'unhealthy',
                error: `CUDA worker not found at: ${CUDA_WORKER_PATH}`
            };
            return false;
        }

        return new Promise((resolve) => {
            const start = Date.now();
            const child = spawn(CUDA_WORKER_PATH, [], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: 10000
            });

            const testData = JSON.stringify({
                jobId: 'health-check',
                type: 'embedding',
                data: [1, 2, 3, 4, 5]
            });

            child.stdin.write(testData);
            child.stdin.end();

            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (data) => stdout += data.toString());
            child.stderr.on('data', (data) => stderr += data.toString());

            child.on('close', (code) => {
                const latency = Date.now() - start;
                
                try {
                    if (code === 0 && stdout.trim()) {
                        const result = JSON.parse(stdout.trim());
                        
                        if (result.jobId === 'health-check' && result.vector && result.vector.length > 0) {
                            this.results.components.cuda_worker = {
                                status: 'healthy',
                                latency: `${latency}ms`,
                                vector_length: result.vector.length,
                                operations: 'working'
                            };
                            resolve(true);
                        } else {
                            this.results.components.cuda_worker = {
                                status: 'unhealthy',
                                error: 'Invalid response format',
                                latency: `${latency}ms`
                            };
                            resolve(false);
                        }
                    } else {
                        this.results.components.cuda_worker = {
                            status: 'unhealthy',
                            error: `Process failed (code: ${code})`,
                            stderr: stderr.trim(),
                            latency: `${latency}ms`
                        };
                        resolve(false);
                    }
                } catch (parseError) {
                    this.results.components.cuda_worker = {
                        status: 'unhealthy',
                        error: `JSON parse error: ${parseError.message}`,
                        stdout: stdout.trim(),
                        latency: `${latency}ms`
                    };
                    resolve(false);
                }
            });

            child.on('error', (error) => {
                this.results.components.cuda_worker = {
                    status: 'unhealthy',
                    error: `Spawn error: ${error.message}`
                };
                resolve(false);
            });
        });
    }

    async checkOrchestratorMaster() {
        console.log('🔍 Checking orchestrator master...');
        
        try {
            const response = await fetch('http://localhost:8099/health');
            
            if (response.ok) {
                const data = await response.json();
                this.results.components.orchestrator_master = {
                    status: 'healthy',
                    total_workers: data.totalWorkers,
                    total_jobs: data.totalJobsProcessed,
                    uptime: `${Math.round(data.uptime)}s`
                };
                return true;
            } else {
                this.results.components.orchestrator_master = {
                    status: 'unhealthy',
                    error: `HTTP ${response.status}: ${response.statusText}`
                };
                return false;
            }
        } catch (error) {
            this.results.components.orchestrator_master = {
                status: 'unhealthy',
                error: error.message,
                note: 'Master might not be running or health endpoint disabled'
            };
            return false;
        }
    }

    async checkRedisGoService() {
        console.log('🔍 Checking Redis Go service...');
        
        try {
            const response = await fetch('http://localhost:8081/health');
            
            if (response.ok) {
                const data = await response.json();
                this.results.components.redis_go_service = {
                    status: data.status === 'healthy' ? 'healthy' : 'unhealthy',
                    connected: data.connected,
                    uptime: `${data.uptime}s`
                };
                return data.status === 'healthy';
            } else {
                this.results.components.redis_go_service = {
                    status: 'unhealthy',
                    error: `HTTP ${response.status}: ${response.statusText}`
                };
                return false;
            }
        } catch (error) {
            this.results.components.redis_go_service = {
                status: 'unhealthy',
                error: error.message,
                note: 'Redis Go service might not be running'
            };
            return false;
        }
    }

    async runAllChecks() {
        console.log('🏥 Starting comprehensive health check...\n');
        
        const checks = [
            this.checkRedis(),
            this.checkRabbitMQ(),
            this.checkCudaWorker(),
            this.checkOrchestratorMaster(),
            this.checkRedisGoService()
        ];

        const results = await Promise.all(checks);
        const healthyCount = results.filter(Boolean).length;
        const totalCount = results.length;

        this.results.overall = healthyCount === totalCount ? 'healthy' : 
                              healthyCount === 0 ? 'critical' : 'degraded';
        
        this.results.summary = {
            healthy_components: healthyCount,
            total_components: totalCount,
            health_percentage: Math.round((healthyCount / totalCount) * 100)
        };

        return this.results;
    }

    printResults() {
        console.log('\n📊 Health Check Results');
        console.log('========================');
        console.log(`Overall Status: ${this.getStatusEmoji()} ${this.results.overall.toUpperCase()}`);
        console.log(`Health Score: ${this.results.summary.health_percentage}% (${this.results.summary.healthy_components}/${this.results.summary.total_components})`);
        console.log(`Timestamp: ${this.results.timestamp}\n`);

        for (const [component, status] of Object.entries(this.results.components)) {
            const emoji = status.status === 'healthy' ? '✅' : '❌';
            console.log(`${emoji} ${component.toUpperCase().replace(/_/g, ' ')}`);
            
            if (status.status === 'healthy') {
                if (status.latency) console.log(`   └─ Latency: ${status.latency}`);
                if (status.operations) console.log(`   └─ Operations: ${status.operations}`);
                if (status.total_workers) console.log(`   └─ Workers: ${status.total_workers}`);
                if (status.uptime) console.log(`   └─ Uptime: ${status.uptime}`);
            } else {
                console.log(`   └─ Error: ${status.error}`);
                if (status.note) console.log(`   └─ Note: ${status.note}`);
            }
            console.log();
        }

        console.log('🔧 Recommended Actions:');
        this.printRecommendations();
    }

    getStatusEmoji() {
        switch (this.results.overall) {
            case 'healthy': return '🟢';
            case 'degraded': return '🟡';
            case 'critical': return '🔴';
            default: return '⚪';
        }
    }

    printRecommendations() {
        const unhealthy = Object.entries(this.results.components)
            .filter(([, status]) => status.status === 'unhealthy')
            .map(([name]) => name);

        if (unhealthy.length === 0) {
            console.log('   ✅ All components are healthy - no action required');
            return;
        }

        unhealthy.forEach(component => {
            switch (component) {
                case 'redis':
                    console.log('   🔧 Redis: Ensure Redis server is running on localhost:6379');
                    break;
                case 'rabbitmq':
                    console.log('   🔧 RabbitMQ: Start RabbitMQ service or check connection settings');
                    break;
                case 'cuda_worker':
                    console.log('   🔧 CUDA Worker: Run "npm run orchestrator:build-cuda" to compile');
                    break;
                case 'orchestrator_master':
                    console.log('   🔧 Orchestrator: Start with "npm run orchestrator:start"');
                    break;
                case 'redis_go_service':
                    console.log('   🔧 Redis Go Service: Start with "npm run orchestrator:redis-service"');
                    break;
            }
        });
    }
}

// CLI interface
if (process.argv[1] === __filename) {
    const checker = new HealthChecker();
    
    checker.runAllChecks()
        .then((results) => {
            checker.printResults();
            
            // Exit with appropriate code
            if (results.overall === 'healthy') {
                process.exit(0);
            } else if (results.overall === 'degraded') {
                process.exit(1);
            } else {
                process.exit(2);
            }
        })
        .catch((error) => {
            console.error('💥 Health check failed:', error.message);
            process.exit(3);
        });
}

export { HealthChecker };