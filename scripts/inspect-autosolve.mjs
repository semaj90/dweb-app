#!/usr/bin/env node
// Inspect autosolve pipeline artifacts: Redis summaries list and RabbitMQ fix_jobs queue depth.
import Redis from 'ioredis';
import amqp from 'amqplib';

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const RABBIT_URL = process.env.RABBIT_URL || 'amqp://localhost';
const FIX_QUEUE = process.env.FIX_QUEUE || 'fix_jobs';

(async () => {
  const redis = new Redis(REDIS_URL);
  try {
    const len = await redis.llen('autosolve_summaries');
    const samples = len ? await redis.lrange('autosolve_summaries', 0, Math.min(2, len-1)) : [];
    console.log('\n📦 Redis autosolve_summaries length:', len);
    if (samples.length) {
      console.log('🔍 Sample summary[0]:');
      try { console.log(JSON.stringify(JSON.parse(samples[0]), null, 2)); } catch { console.log(samples[0]); }
    }
  } catch (e) {
    console.error('❌ Redis inspection failed:', e.message);
  } finally {
    redis.quit();
  }

  // RabbitMQ queue depth
  try {
    const conn = await amqp.connect(RABBIT_URL);
    const ch = await conn.createChannel();
    await ch.assertQueue(FIX_QUEUE, { durable: true });
    const q = await ch.checkQueue(FIX_QUEUE);
    console.log(`\n🐇 RabbitMQ queue '${FIX_QUEUE}' messages:`, q.messageCount);
    await ch.close();
    await conn.close();
  } catch (e) {
    console.error('❌ RabbitMQ inspection failed:', e.message);
  }
})();
