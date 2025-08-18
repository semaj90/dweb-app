// autosolve-to-queue.cjs - run autosolve then publish summary to Redis list for orchestrator
const { autosolve } = require('./autosolve-runner.cjs');
const Redis = require('ioredis');
const amqp = require('amqplib');

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const RABBIT_URL = process.env.RABBIT_URL || 'amqp://localhost';
const FIX_QUEUE = process.env.FIX_QUEUE || 'fix_jobs';

async function publishFixJob(summary){
  try {
    const conn = await amqp.connect(RABBIT_URL);
    const ch = await conn.createChannel();
    await ch.assertQueue(FIX_QUEUE, { durable:true });
    const job = { jobId: `autosolve-${Date.now()}`, type:'fix_job', data: summary, metadata:{ source:'autosolve', created_at: new Date().toISOString() } };
    ch.sendToQueue(FIX_QUEUE, Buffer.from(JSON.stringify(job)), { persistent:true });
    await ch.close();
    await conn.close();
    console.log('📤 Published fix_job from autosolve');
  } catch(e){ console.error('Failed to publish fix_job', e.message); }
}

(async ()=>{
  const redis = new Redis(REDIS_URL);
  const code = await autosolve();
  const summary = { exitCode: code, timestamp: new Date().toISOString() };
  await redis.rpush('autosolve_summaries', JSON.stringify(summary));
  await publishFixJob(summary);
  redis.disconnect();
  process.exit(code);
})();
