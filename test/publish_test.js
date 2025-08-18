// test/publish_test.js
// Test job publisher for GPU orchestrator system

import amqp from 'amqplib';
import { fileURLToPath } from 'url';

const RABBIT_URL = process.env.RABBIT_URL || 'amqp://localhost';
const QUEUE_NAME = process.env.QUEUE_NAME || 'gpu_jobs';

// Test job templates
const TEST_JOBS = {
    embedding: {
        type: 'embedding',
        data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        description: 'Basic embedding calculation'
    },
    similarity: {
        type: 'similarity',
        data: [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], // Two vectors concatenated
        description: 'Vector similarity computation'
    },
    autoindex: {
        type: 'autoindex',
        data: Array.from({length: 16}, () => Math.random()),
        description: 'Auto-indexing operation'
    },
    large_vector: {
        type: 'embedding',
        data: Array.from({length: 128}, (_, i) => Math.sin(i * 0.1)),
        description: 'Large vector embedding (128 dimensions)'
    },
    stress_test: {
        type: 'embedding',
        data: Array.from({length: 512}, (_, i) => Math.random() * 2 - 1),
        description: 'Stress test with 512-dimensional vector'
    }
};

class JobPublisher {
    constructor() {
        this.connection = null;
        this.channel = null;
    }

    async connect() {
        try {
            console.log(`🔌 Connecting to RabbitMQ at ${RABBIT_URL}...`);
            this.connection = await amqp.connect(RABBIT_URL);
            this.channel = await this.connection.createChannel();
            await this.channel.assertQueue(QUEUE_NAME, { durable: true });
            console.log(`✅ Connected to queue: ${QUEUE_NAME}`);
        } catch (error) {
            console.error(`❌ Failed to connect to RabbitMQ:`, error.message);
            throw error;
        }
    }

    async publishJob(jobType, customData = null) {
        if (!this.channel) {
            throw new Error('Not connected to RabbitMQ');
        }

        const template = TEST_JOBS[jobType];
        if (!template) {
            throw new Error(`Unknown job type: ${jobType}`);
        }

        const job = {
            jobId: `test-${jobType}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            type: template.type,
            data: customData || template.data,
            metadata: {
                description: template.description,
                publishedAt: new Date().toISOString(),
                testRun: true,
                dimensions: (customData || template.data).length
            }
        };

        const message = Buffer.from(JSON.stringify(job));
        this.channel.sendToQueue(QUEUE_NAME, message, { persistent: true });
        
        console.log(`📤 Published job: ${job.jobId}`);
        console.log(`   └─ Type: ${job.type}`);
        console.log(`   └─ Description: ${job.metadata.description}`);
        console.log(`   └─ Data dimensions: ${job.metadata.dimensions}`);
        
        return job.jobId;
    }

    async publishBatch(jobType, count = 5) {
        console.log(`📦 Publishing batch of ${count} ${jobType} jobs...`);
        const jobIds = [];
        
        for (let i = 0; i < count; i++) {
            const jobId = await this.publishJob(jobType);
            jobIds.push(jobId);
            
            // Small delay between jobs to avoid overwhelming
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        console.log(`✅ Batch complete: ${count} jobs published`);
        return jobIds;
    }

    async publishStressTest(duration = 30) {
        console.log(`🔥 Starting stress test for ${duration} seconds...`);
        const startTime = Date.now();
        const endTime = startTime + (duration * 1000);
        let jobCount = 0;
        
        const jobTypes = Object.keys(TEST_JOBS);
        
        while (Date.now() < endTime) {
            const randomJobType = jobTypes[Math.floor(Math.random() * jobTypes.length)];
            await this.publishJob(randomJobType);
            jobCount++;
            
            // Random delay between 100ms and 1s
            const delay = Math.random() * 900 + 100;
            await new Promise(resolve => setTimeout(resolve, delay));
        }
        
        const actualDuration = (Date.now() - startTime) / 1000;
        console.log(`✅ Stress test complete: ${jobCount} jobs in ${actualDuration.toFixed(1)}s (${(jobCount/actualDuration).toFixed(2)} jobs/sec)`);
        return jobCount;
    }

    async close() {
        if (this.channel) {
            await this.channel.close();
        }
        if (this.connection) {
            await this.connection.close();
        }
        console.log('🔌 Disconnected from RabbitMQ');
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'help';
    
    const publisher = new JobPublisher();
    
    try {
        await publisher.connect();
        
        switch (command) {
            case 'single':
                const jobType = args[1] || 'embedding';
                if (!TEST_JOBS[jobType]) {
                    console.error(`❌ Unknown job type: ${jobType}`);
                    console.log(`Available types: ${Object.keys(TEST_JOBS).join(', ')}`);
                    process.exit(1);
                }
                await publisher.publishJob(jobType);
                break;
                
            case 'batch':
                const batchJobType = args[1] || 'embedding';
                const batchCount = parseInt(args[2]) || 5;
                if (!TEST_JOBS[batchJobType]) {
                    console.error(`❌ Unknown job type: ${batchJobType}`);
                    console.log(`Available types: ${Object.keys(TEST_JOBS).join(', ')}`);
                    process.exit(1);
                }
                await publisher.publishBatch(batchJobType, batchCount);
                break;
                
            case 'stress':
                const duration = parseInt(args[1]) || 30;
                await publisher.publishStressTest(duration);
                break;
                
            case 'all':
                console.log('📋 Publishing one job of each type...');
                for (const jobType of Object.keys(TEST_JOBS)) {
                    await publisher.publishJob(jobType);
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
                break;
                
            case 'custom':
                const customJobType = args[1] || 'embedding';
                const dimensions = parseInt(args[2]) || 64;
                const customData = Array.from({length: dimensions}, () => Math.random() * 2 - 1);
                console.log(`🎯 Publishing custom ${customJobType} job with ${dimensions} dimensions...`);
                await publisher.publishJob(customJobType, customData);
                break;
                
            case 'help':
            default:
                console.log('📚 GPU Orchestrator Test Job Publisher');
                console.log('=====================================');
                console.log('Usage: node publish_test.js <command> [options]');
                console.log('');
                console.log('Commands:');
                console.log('  single [type]           - Publish single job (default: embedding)');
                console.log('  batch [type] [count]    - Publish batch of jobs (default: embedding, 5)');
                console.log('  stress [duration]       - Stress test for N seconds (default: 30)');
                console.log('  all                     - Publish one job of each type');
                console.log('  custom [type] [dims]    - Publish custom job with N dimensions');
                console.log('  help                    - Show this help');
                console.log('');
                console.log('Available job types:');
                for (const [type, template] of Object.entries(TEST_JOBS)) {
                    console.log(`  ${type.padEnd(15)} - ${template.description}`);
                }
                console.log('');
                console.log('Examples:');
                console.log('  node publish_test.js single embedding');
                console.log('  node publish_test.js batch similarity 10');
                console.log('  node publish_test.js stress 60');
                console.log('  node publish_test.js custom embedding 256');
                break;
        }
        
    } catch (error) {
        console.error('💥 Error:', error.message);
        process.exit(1);
    } finally {
        await publisher.close();
    }
}

// Run if called directly
const __filename = fileURLToPath(import.meta.url);

if (process.argv[1] === __filename) {
    main().catch(console.error);
}

export { JobPublisher, TEST_JOBS };