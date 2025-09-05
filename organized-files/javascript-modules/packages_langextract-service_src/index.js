import Fastify from 'fastify';
import cors from '@fastify/cors';
import websocket from '@fastify/websocket';
import { Queue, Worker } from 'bullmq';
import Redis from 'ioredis';
import { PythonShell } from 'python-shell';
import { z } from 'zod';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Schemas
const ExtractSchema = z.object({
  text: z.string(),
  schema: z.object({}).optional(),
  options: z.object({
    model: z.string().default('gpt-3.5-turbo'),
    temperature: z.number().min(0).max(2).default(0.1),
    max_tokens: z.number().default(1000),
    gpu_acceleration: z.boolean().default(true)
  }).default({})
});

class LangExtractService {
  constructor() {
    this.fastify = Fastify({
      logger: {
        transport: {
          target: 'pino-pretty',
          options: {
            colorize: true
          }
        }
      }
    });
    
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT) || 6379,
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      maxRetriesPerRequest: null
    });

    this.extractQueue = new Queue('langextract', { connection: this.redis });
    this.setupWorker();
    this.setupRoutes();
  }

  setupRoutes() {
    // Register plugins
    this.fastify.register(cors, {
      origin: true,
      credentials: true
    });
    
    this.fastify.register(websocket);

    // Health check
    this.fastify.get('/health', async () => {
      const gpuInfo = await this.checkGPUAvailability();
      return {
        status: 'healthy',
        service: 'langextract',
        gpu: gpuInfo,
        timestamp: new Date().toISOString(),
        queues: {
          waiting: await this.extractQueue.getWaiting(),
          active: await this.extractQueue.getActive(),
          completed: await this.extractQueue.getCompleted()
        }
      };
    });

    // Extract endpoint
    this.fastify.post('/extract', async (request, reply) => {
      try {
        const data = ExtractSchema.parse(request.body);
        
        const job = await this.extractQueue.add('extract', {
          text: data.text,
          schema: data.schema,
          options: data.options,
          requestId: generateId()
        }, {
          attempts: 3,
          backoff: {
            type: 'exponential',
            delay: 2000
          }
        });

        return {
          jobId: job.id,
          status: 'queued',
          message: 'Extraction job queued successfully'
        };
      } catch (error) {
        reply.code(400);
        return { error: error.message };
      }
    });

    // Job status endpoint
    this.fastify.get('/job/:jobId', async (request, reply) => {
      const { jobId } = request.params;
      const job = await this.extractQueue.getJob(jobId);
      
      if (!job) {
        reply.code(404);
        return { error: 'Job not found' };
      }

      return {
        jobId: job.id,
        status: await job.getState(),
        progress: job.progress,
        data: job.returnvalue,
        error: job.failedReason
      };
    });

    // WebSocket for real-time updates
    this.fastify.register(async function (fastify) {
      fastify.get('/ws/:jobId', { websocket: true }, (connection, req) => {
        const { jobId } = req.params;
        
        // Subscribe to job updates
        const subscriber = new Redis({
          host: process.env.REDIS_HOST || 'localhost',
          port: parseInt(process.env.REDIS_PORT) || 6379
        });

        subscriber.subscribe(`langextract:progress:${jobId}`);
        
        subscriber.on('message', (channel, message) => {
          connection.socket.send(message);
        });

        connection.socket.on('close', () => {
          subscriber.disconnect();
        });
      });
    });

    // Batch extract endpoint
    this.fastify.post('/extract/batch', async (request, reply) => {
      try {
        const { items, schema, options = {} } = request.body;
        
        if (!Array.isArray(items)) {
          reply.code(400);
          return { error: 'Items must be an array' };
        }

        const jobs = [];
        for (const [index, text] of items.entries()) {
          const job = await this.extractQueue.add('extract', {
            text,
            schema,
            options,
            requestId: generateId(),
            batchIndex: index
          });
          jobs.push({ index, jobId: job.id });
        }

        return {
          batchId: generateId(),
          jobs,
          status: 'queued',
          message: `${jobs.length} extraction jobs queued`
        };
      } catch (error) {
        reply.code(400);
        return { error: error.message };
      }
    });
  }

  setupWorker() {
    this.worker = new Worker('langextract', async (job) => {
      const { text, schema, options, requestId } = job.data;
      
      try {
        // Update progress
        await job.updateProgress(10);
        await this.publishProgress(job.id, { status: 'starting', progress: 10 });

        // Check GPU availability
        const gpuInfo = await this.checkGPUAvailability();
        await job.updateProgress(20);

        // Prepare Python environment and script
        const pythonScript = path.join(__dirname, 'python', 'langextract_processor.py');
        
        await job.updateProgress(30);
        await this.publishProgress(job.id, { status: 'processing', progress: 30 });

        // Run LangExtract via Python
        const result = await this.runLangExtract(text, schema, {
          ...options,
          gpu_available: gpuInfo.available
        });

        await job.updateProgress(80);
        await this.publishProgress(job.id, { status: 'finalizing', progress: 80 });

        // Post-process results
        const processedResult = this.postProcessResults(result);

        await job.updateProgress(100);
        await this.publishProgress(job.id, { 
          status: 'completed', 
          progress: 100, 
          result: processedResult 
        });

        return processedResult;
      } catch (error) {
        await this.publishProgress(job.id, { 
          status: 'failed', 
          error: error.message 
        });
        throw error;
      }
    }, { 
      connection: this.redis,
      concurrency: parseInt(process.env.WORKER_CONCURRENCY) || 3
    });

    this.worker.on('completed', (job, result) => {
      this.fastify.log.info(`Job ${job.id} completed successfully`);
    });

    this.worker.on('failed', (job, err) => {
      this.fastify.log.error(`Job ${job.id} failed: ${err.message}`);
    });
  }

  async runLangExtract(text, schema, options) {
    return new Promise((resolve, reject) => {
      const pythonOptions = {
        mode: 'json',
        pythonPath: process.env.PYTHON_PATH || 'python',
        pythonOptions: ['-u'],
        scriptPath: path.join(__dirname, 'python'),
        args: [
          '--text', JSON.stringify(text),
          '--schema', JSON.stringify(schema || {}),
          '--options', JSON.stringify(options)
        ]
      };

      PythonShell.run('langextract_processor.py', pythonOptions, (err, results) => {
        if (err) {
          reject(new Error(`Python script error: ${err.message}`));
          return;
        }

        if (!results || results.length === 0) {
          reject(new Error('No results from Python script'));
          return;
        }

        try {
          const result = results[results.length - 1]; // Get last result
          resolve(typeof result === 'string' ? JSON.parse(result) : result);
        } catch (parseError) {
          reject(new Error(`Failed to parse Python script output: ${parseError.message}`));
        }
      });
    });
  }

  async checkGPUAvailability() {
    try {
      const result = await new Promise((resolve, reject) => {
        PythonShell.run('gpu_check.py', {
          mode: 'json',
          scriptPath: path.join(__dirname, 'python')
        }, (err, results) => {
          if (err) reject(err);
          else resolve(results[0]);
        });
      });
      
      return result;
    } catch (error) {
      return {
        available: false,
        device: 'none',
        error: error.message
      };
    }
  }

  postProcessResults(result) {
    return {
      ...result,
      processed_at: new Date().toISOString(),
      service: 'langextract',
      version: '1.0.0'
    };
  }

  async publishProgress(jobId, data) {
    await this.redis.publish(`langextract:progress:${jobId}`, JSON.stringify(data));
  }

  async start() {
    try {
      await this.fastify.listen({ 
        port: parseInt(process.env.PORT) || 3001,
        host: '0.0.0.0'
      });
      
      this.fastify.log.info('🚀 LangExtract Service started successfully');
      this.fastify.log.info(`📊 Worker concurrency: ${this.worker.opts.concurrency}`);
      
      // Check GPU on startup
      const gpuInfo = await this.checkGPUAvailability();
      this.fastify.log.info(`🔥 GPU Acceleration: ${gpuInfo.available ? 'Enabled' : 'Disabled'} (${gpuInfo.device})`);
      
    } catch (err) {
      this.fastify.log.error(err);
      process.exit(1);
    }
  }

  async stop() {
    await this.worker.close();
    await this.redis.disconnect();
    await this.fastify.close();
  }
}

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Received SIGTERM, shutting down gracefully...');
  await service.stop();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('Received SIGINT, shutting down gracefully...');
  await service.stop();
  process.exit(0);
});

// Start the service
const service = new LangExtractService();
service.start();

export default LangExtractService;
