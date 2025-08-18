// orchestrator/worker_process.js
// RabbitMQ consumer + spawn cuda-worker + xstate idle machine

import amqp from 'amqplib';
import { spawn } from 'child_process';
import Redis from 'ioredis';
import { createMachine, interpret } from 'xstate';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const RABBIT_URL = process.env.RABBIT_URL || 'amqp://localhost';
const QUEUE_NAME = process.env.QUEUE_NAME || 'gpu_jobs';
const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const CUDA_WORKER_PATH = process.env.CUDA_WORKER_PATH || path.join(__dirname, '..', 'cuda-worker', 'cuda-worker.exe');
const MOCK_CUDA_WORKER_PATH = path.join(__dirname, '..', 'cuda-worker', 'mock-cuda-worker.cjs');
const WORKER_MODE = process.env.WORKER_MODE || 'production';
const AUTO_SOLVE_ENABLED = process.env.AUTO_SOLVE_ENABLED === 'true';
const FIX_QUEUE = process.env.FIX_QUEUE || 'fix_jobs';

const workerId = process.pid;
const redis = new Redis(REDIS_URL, {
    retryDelayOnFailover: 100,
    enableReadyCheck: false,
    maxRetriesPerRequest: 3,
    lazyConnect: true
});

// XState idle detection machine
const idleStateMachine = createMachine({
    id: 'idleDetection',
    initial: 'active',
    context: {
        idleStart: null,
        jobCount: 0,
        lastActivity: Date.now()
    },
    states: {
        active: {
            entry: ['updateActivity'],
            on: {
                JOB_RECEIVED: {
                    actions: ['incrementJobCount', 'updateActivity']
                },
                CHECK_IDLE: {
                    target: 'checkingIdle'
                },
                FORCE_IDLE: 'idleWait'
            }
        },
        checkingIdle: {
            always: [
                {
                    target: 'idleWait',
                    cond: 'hasBeenIdleLongEnough'
                },
                {
                    target: 'active'
                }
            ]
        },
        idleWait: {
            entry: ['startIdleTimer'],
            after: {
                60000: 'triggerAutoIndex' // 60 seconds of idle triggers auto-index
            },
            on: {
                JOB_RECEIVED: {
                    target: 'active',
                    actions: ['incrementJobCount', 'updateActivity']
                }
            }
        },
        triggerAutoIndex: {
            entry: ['executeAutoIndex'],
            always: 'active'
        }
    }
}, {
    actions: {
        updateActivity: (context) => {
            context.lastActivity = Date.now();
        },
        incrementJobCount: (context) => {
            context.jobCount++;
        },
        startIdleTimer: (context) => {
            context.idleStart = Date.now();
            console.log(`🏃‍♂️ Worker ${workerId} entering idle state`);
        },
        executeAutoIndex: async () => {
            if (AUTO_SOLVE_ENABLED) {
                console.log(`🤖 Worker ${workerId} triggering auto-index job`);
                await publishAutoIndexJob();
            }
        }
    },
    guards: {
        hasBeenIdleLongEnough: (context) => {
            const now = Date.now();
            return (now - context.lastActivity) > 30000; // 30 seconds
        }
    }
});

const idleService = interpret(idleStateMachine).start();

// Track worker health and performance
let workerStats = {
    startTime: Date.now(),
    jobsProcessed: 0,
    jobsSuccessful: 0,
    jobsFailed: 0,
    averageProcessingTime: 0,
    lastJobTime: null,
    cudaAvailable: false,
    redisConnected: false
};

// Check CUDA worker availability with fallback to mock
function checkCudaWorker() {
    return new Promise((resolve) => {
        // First try the real CUDA worker
        if (fs.existsSync(CUDA_WORKER_PATH)) {
            const testChild = spawn(CUDA_WORKER_PATH, [], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: 5000
            });

            const testData = JSON.stringify({
                jobId: 'health-check',
                type: 'embedding',
                data: [1, 2, 3, 4]
            });

            testChild.stdin.write(testData);
            testChild.stdin.end();

            let output = '';
            testChild.stdout.on('data', (data) => output += data.toString());

            testChild.on('close', (code) => {
                try {
                    const result = JSON.parse(output);
                    const success = result.jobId === 'health-check' && result.vector && result.vector.length > 0;
                    if (success) {
                        console.log(`✅ CUDA worker health check: PASSED (Real GPU)`);
                        resolve(true);
                        return;
                    }
                } catch (e) {
                    // Fall through to mock worker
                }
                testMockWorker(resolve);
            });

            testChild.on('error', (err) => {
                console.log(`❌ CUDA worker spawn error: ${err.message}`);
                testMockWorker(resolve);
            });
        } else {
            console.log(`⚠️ CUDA worker not found at: ${CUDA_WORKER_PATH}`);
            testMockWorker(resolve);
        }
    });
}

function testMockWorker(resolve) {
    console.log(`🧪 Testing mock CUDA worker fallback...`);
    
    const testChild = spawn('node', [MOCK_CUDA_WORKER_PATH], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 5000
    });

    const testData = JSON.stringify({
        jobId: 'health-check',
        type: 'embedding',
        data: [1, 2, 3, 4]
    });

    testChild.stdin.write(testData);
    testChild.stdin.end();

    let output = '';
    testChild.stdout.on('data', (data) => output += data.toString());

    testChild.on('close', (code) => {
        try {
            const result = JSON.parse(output);
            const success = result.jobId === 'health-check' && result.vector && result.vector.length > 0;
            console.log(`${success ? '✅' : '❌'} Mock CUDA worker health check: ${success ? 'PASSED (CPU Simulation)' : 'FAILED'}`);
            resolve(success);
        } catch (e) {
            console.log(`❌ Mock CUDA worker health check failed: ${e.message}`);
            resolve(false);
        }
    });

    testChild.on('error', (err) => {
        console.log(`❌ Mock CUDA worker spawn error: ${err.message}`);
        resolve(false);
    });
}

// Auto-index job publisher
async function publishAutoIndexJob() {
    try {
        const conn = await amqp.connect(RABBIT_URL);
        const ch = await conn.createChannel();
        await ch.assertQueue(QUEUE_NAME, { durable: true });

        const job = {
            jobId: `autoindex-${workerId}-${Date.now()}`,
            type: 'autoindex',
            data: Array.from({length: 8}, () => Math.random()), // Random data for indexing
            metadata: {
                triggeredBy: 'idle-detection',
                workerId: workerId,
                timestamp: new Date().toISOString()
            }
        };

        ch.sendToQueue(QUEUE_NAME, Buffer.from(JSON.stringify(job)), { persistent: true });
        console.log(`📤 Published auto-index job: ${job.jobId}`);

        await ch.close();
        await conn.close();
    } catch (error) {
        console.error(`❌ Failed to publish auto-index job:`, error.message);
    }
}

// Publish a fix job (triggered after autosolve iteration or external event)
async function publishFixJob(fixSummary) {
    try {
        const conn = await amqp.connect(RABBIT_URL);
        const ch = await conn.createChannel();
        await ch.assertQueue(FIX_QUEUE, { durable: true });
        const job = {
            jobId: `fix-${workerId}-${Date.now()}`,
            type: 'fix_job',
            data: fixSummary || { status: 'noop' },
            metadata: { created_at: new Date().toISOString(), workerId }
        };
        ch.sendToQueue(FIX_QUEUE, Buffer.from(JSON.stringify(job)), { persistent: true });
        console.log(`🧩 Published fix job: ${job.jobId}`);
        await ch.close();
        await conn.close();
    } catch(e){ console.error('Failed to publish fix job', e.message); }
}

// Process job with CUDA worker (with mock fallback)
async function processJobWithCuda(job) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();
        let workerPath = CUDA_WORKER_PATH;
        let workerArgs = [];

        // Use mock worker if real CUDA worker doesn't exist
        if (!fs.existsSync(CUDA_WORKER_PATH)) {
            workerPath = 'node';
            workerArgs = [MOCK_CUDA_WORKER_PATH];
        }

        const child = spawn(workerPath, workerArgs, {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        const requestData = {
            jobId: job.jobId,
            type: job.type || 'embedding',
            data: job.data || [1, 2, 3, 4]
        };

        child.stdin.write(JSON.stringify(requestData));
        child.stdin.end();

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => stdout += data.toString());
        child.stderr.on('data', (data) => stderr += data.toString());

        child.on('close', (code) => {
            const processingTime = Date.now() - startTime;

            if (code === 0 && stdout.trim()) {
                try {
                    const result = JSON.parse(stdout.trim());
                    result.processingTime = processingTime;
                    result.workerId = workerId;
                    resolve(result);
                } catch (parseError) {
                    // If real CUDA worker failed, try mock worker
                    if (workerPath === CUDA_WORKER_PATH) {
                        console.log(`⚠️ Real CUDA worker failed, trying mock worker...`);
                        processJobWithMock(job, startTime).then(resolve).catch(reject);
                        return;
                    }
                    reject(new Error(`Failed to parse worker output: ${parseError.message}`));
                }
            } else {
                // If real CUDA worker failed, try mock worker
                if (workerPath === CUDA_WORKER_PATH) {
                    console.log(`⚠️ Real CUDA worker failed (code: ${code}), trying mock worker...`);
                    processJobWithMock(job, startTime).then(resolve).catch(reject);
                    return;
                }
                reject(new Error(`Worker failed (code: ${code}): ${stderr}`));
            }
        });

        child.on('error', (error) => {
            // If real CUDA worker failed, try mock worker
            if (workerPath === CUDA_WORKER_PATH) {
                console.log(`⚠️ Real CUDA worker spawn error, trying mock worker...`);
                processJobWithMock(job, startTime).then(resolve).catch(reject);
                return;
            }
            reject(new Error(`Worker spawn error: ${error.message}`));
        });

        // Timeout after 30 seconds
        setTimeout(() => {
            child.kill('SIGKILL');
            reject(new Error('Worker timeout'));
        }, 30000);
    });
}

// Process job with mock worker
async function processJobWithMock(job, originalStartTime) {
    return new Promise((resolve, reject) => {
        const startTime = originalStartTime || Date.now();

        const child = spawn('node', [MOCK_CUDA_WORKER_PATH], {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        const requestData = {
            jobId: job.jobId,
            type: job.type || 'embedding',
            data: job.data || [1, 2, 3, 4]
        };

        child.stdin.write(JSON.stringify(requestData));
        child.stdin.end();

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => stdout += data.toString());
        child.stderr.on('data', (data) => stderr += data.toString());

        child.on('close', (code) => {
            const processingTime = Date.now() - startTime;

            if (code === 0 && stdout.trim()) {
                try {
                    const result = JSON.parse(stdout.trim());
                    result.processingTime = processingTime;
                    result.workerId = workerId;
                    resolve(result);
                } catch (parseError) {
                    reject(new Error(`Failed to parse mock worker output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`Mock worker failed (code: ${code}): ${stderr}`));
            }
        });

        child.on('error', (error) => {
            reject(new Error(`Mock worker spawn error: ${error.message}`));
        });

        // Timeout after 10 seconds for mock worker
        setTimeout(() => {
            child.kill('SIGKILL');
            reject(new Error('Mock worker timeout'));
        }, 10000);
    });
}

// Store result in Redis
async function storeResult(jobId, result, error = null) {
    try {
        const resultData = {
            jobId,
            result: result || null,
            error: error || null,
            workerId,
            timestamp: new Date().toISOString(),
            processingTime: result?.processingTime || null
        };

        await redis.hset(`job:${jobId}`, resultData);
        await redis.expire(`job:${jobId}`, 3600); // 1 hour TTL

        // Also store in a list for monitoring
        await redis.lpush('job_history', JSON.stringify({
            jobId,
            status: error ? 'failed' : 'completed',
            workerId,
            timestamp: resultData.timestamp
        }));
        await redis.ltrim('job_history', 0, 999); // Keep last 1000 jobs

    } catch (redisError) {
        console.error(`❌ Failed to store result in Redis:`, redisError.message);
    }
}

// Main worker processing loop
async function startWorker() {
    console.log(`🚀 Worker ${workerId} starting in ${WORKER_MODE} mode`);

    // Initial health checks
    workerStats.cudaAvailable = await checkCudaWorker();

    try {
        await redis.ping();
        workerStats.redisConnected = true;
        console.log(`✅ Redis connection established`);
    } catch (redisError) {
        console.log(`⚠️ Redis connection failed: ${redisError.message}`);
    }

    // Connect to RabbitMQ and start consuming
    try {
        const conn = await amqp.connect(RABBIT_URL);
        const ch = await conn.createChannel();
        await ch.assertQueue(QUEUE_NAME, { durable: true });
        ch.prefetch(1); // Process one job at a time

        console.log(`✅ Connected to RabbitMQ, waiting for jobs on queue: ${QUEUE_NAME}`);

        ch.consume(QUEUE_NAME, async (msg) => {
            if (msg === null) return;

            let job;
            try {
                job = JSON.parse(msg.content.toString());
                console.log(`📨 Worker ${workerId} received job: ${job.jobId} (type: ${job.type})`);

                // Notify state machine of job activity
                idleService.send('JOB_RECEIVED');

                // Process job
                const startTime = Date.now();
                const result = await processJobWithCuda(job);
                const processingTime = Date.now() - startTime;

                // Update statistics
                workerStats.jobsProcessed++;
                workerStats.jobsSuccessful++;
                workerStats.lastJobTime = Date.now();
                workerStats.averageProcessingTime =
                    (workerStats.averageProcessingTime * (workerStats.jobsProcessed - 1) + processingTime) / workerStats.jobsProcessed;

                // Store result
                await storeResult(job.jobId, result);

                // If this was a fix_job, optionally trigger follow-up (e.g., re-run check pipeline)
                if (job.type === 'fix_job'){
                    redis.lpush('fix_job_history', JSON.stringify({ jobId: job.jobId, ts: Date.now(), result }));
                    redis.ltrim('fix_job_history',0,199);
                }

                // Notify master process
                process.send?.({
                    type: 'job_completed',
                    jobId: job.jobId,
                    processingTime,
                    workerId
                });

                console.log(`✅ Job ${job.jobId} completed in ${processingTime}ms`);
                ch.ack(msg);

            } catch (error) {
                console.error(`❌ Job processing failed:`, error.message);

                // Update statistics
                workerStats.jobsFailed++;

                if (job) {
                    await storeResult(job.jobId, null, error.message);
                }

                // Reject and don't requeue to avoid infinite loops
                ch.nack(msg, false, false);
            }
        });

        // Periodic idle check
        setInterval(() => {
            idleService.send('CHECK_IDLE');
        }, 15000); // Check every 15 seconds

        // Periodically poll for autosolve summaries in Redis to emit fix jobs
        if (AUTO_SOLVE_ENABLED){
            setInterval(async ()=>{
                try {
                    const summary = await redis.lpop('autosolve_summaries');
                    if (summary){
                        await publishFixJob(JSON.parse(summary));
                    }
                } catch(e){ /* ignore */ }
            }, 20000);
        }

        // Periodic health report to master
        setInterval(() => {
            if (process.send) {
                process.send({
                    type: 'worker_health',
                    workerId,
                    data: {
                        ...workerStats,
                        currentState: idleService.state.value,
                        uptime: Date.now() - workerStats.startTime
                    }
                });
            }
        }, 30000); // Every 30 seconds

        // Handle shutdown signal
        process.on('message', (msg) => {
            if (msg.type === 'shutdown') {
                console.log(`🛑 Worker ${workerId} received shutdown signal`);
                ch.close();
                conn.close();
                redis.disconnect();
                process.exit(0);
            }
        });

    } catch (connectionError) {
        console.error(`❌ Worker ${workerId} failed to start:`, connectionError.message);
        process.exit(1);
    }
}

// Start the worker
startWorker().catch((error) => {
    console.error(`💥 Worker ${workerId} crashed:`, error.message);
    process.exit(1);
});