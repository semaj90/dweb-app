// scripts/redis-monitor.js
const redis = require('ioredis');
const client = new redis(process.env.REDIS_PORT, process.env.REDIS_HOST);

setInterval(async () => {
  const info = await client.info('stats');
  const memory = await client.info('memory');
  const connected = await client.client('list');
  
  console.log({
    time: new Date().toISOString(),
    connections: connected.split('\n').length - 1,
    memory: memory.match(/used_memory_human:(.+)/)?.[1],
    ops: info.match(/instantaneous_ops_per_sec:(\d+)/)?.[1]
  });
  
  // Alert if issues
  if (connected.split('\n').length > 100) {
    console.error('WARNING: High connection count!');
  }
}, 30000);