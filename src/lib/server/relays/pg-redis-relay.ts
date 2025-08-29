import * as postgres from 'postgres';
import { createClient } from 'redis';
import type { RedisClientType } from 'redis';

// Load environment config
const envConfig = {
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379'
};

// Connect to PostgreSQL
const pg = postgres.default(envConfig.DATABASE_URL, { 
  max: 1,
  onnotice: () => {}, // Suppress notices
});

// Connect to Redis
const redis: RedisClientType = createClient({ url: envConfig.REDIS_URL });

async function startRelay(): Promise<any> {
  try {
    // Connect to Redis
    await redis.connect();
    console.log('✅ Connected to Redis');

    // Subscribe to PostgreSQL channels
    const channels = ['evidence_inserted', 'report_inserted', 'cases_changed', 'reports_changed'];
    
    for (const channel of channels) {
      // Use direct SQL to listen to channels
      await pg.unsafe(`LISTEN ${channel}`);
      console.log(`📡 Listening to PostgreSQL channel: ${channel}`);
    }

    // Set up notification handlers
    pg.listen('evidence_inserted', async (payload): Promise<any> => {
      console.log('📨 Evidence notification:', payload);
      await redis.publish('realtime:evidence', payload);
    });

    pg.listen('report_inserted', async (payload): Promise<any> => {
      console.log('📨 Report notification:', payload);
      await redis.publish('realtime:reports', payload);
    });

    pg.listen('cases_changed', async (payload): Promise<any> => {
      console.log('📨 Cases notification:', payload);
      await redis.publish('realtime:cases', payload);
    });

    pg.listen('reports_changed', async (payload): Promise<any> => {
      console.log('📨 Reports notification:', payload);
      await redis.publish('realtime:reports', payload);
    });

    console.log('🚀 PG→Redis relay running successfully');
    console.log('📡 Monitoring channels: evidence_inserted, report_inserted, cases_changed, reports_changed');

  } catch (error: any) {
    console.error('❌ Error starting PG→Redis relay:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', async (): Promise<any> => {
  console.log('\n📤 Shutting down PG→Redis relay...');
  await redis.quit();
  await pg.end();
  process.exit(0);
});

// Start the relay
startRelay();