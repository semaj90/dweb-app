// src/lib/server/rabbitmq.ts
import * as amqp from 'amqplib';
import type { Channel } from 'amqplib';

let connection: any | null = null;
let channel: Channel | null = null;

export async function getConnection(): Promise<any> {
  if (connection) return connection;
  
  const rabbitmqUrl = import.meta.env.RABBITMQ_URL || 'amqp://localhost:5672';
  console.log('🐰 Connecting to RabbitMQ:', rabbitmqUrl);
  
  connection = await amqp.connect(rabbitmqUrl);
  
  connection.on('error', (err) => {
    console.error('❌ RabbitMQ connection error:', err);
    connection = null;
    channel = null;
  });
  
  connection.on('close', () => {
    console.log('🔌 RabbitMQ connection closed');
    connection = null;
    channel = null;
  });
  
  return connection;
}

export async function getChannel(): Promise<Channel> {
  if (channel) return channel;
  
  const conn = await getConnection();
  channel = await (conn as any).createChannel();
  
  // Set prefetch for better load balancing
  await channel.prefetch(1);
  
  channel.on('error', (err) => {
    console.error('❌ RabbitMQ channel error:', err);
    channel = null;
  });
  
  channel.on('close', () => {
    console.log('📺 RabbitMQ channel closed');
    channel = null;
  });
  
  return channel;
}

export async function publishToQueue(queueName: string, payload: any): Promise<void> {
  try {
    const ch = await getChannel();
    
    // Ensure queue exists
    await ch.assertQueue(queueName, { 
      durable: true,
      arguments: {
        'x-message-ttl': 3600000, // 1 hour TTL
        'x-max-length': 10000     // Max 10k messages
      }
    });
    
    const message = JSON.stringify(payload);
    const sent = ch.sendToQueue(queueName, Buffer.from(message), { 
      persistent: true,
      timestamp: Date.now(),
      messageId: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    });
    
    if (!sent) {
      throw new Error('Message queue is full');
    }
    
    console.log(`📤 Published to queue ${queueName}:`, { 
      messageId: payload.sessionId || 'unknown',
      queueName 
    });
    
  } catch (error: any) {
    console.error(`❌ Failed to publish to queue ${queueName}:`, error);
    throw error;
  }
}

export async function consumeFromQueue(
  queueName: string, 
  processor: (payload: any, ack: () => void, nack: () => void) => Promise<void>
): Promise<void> {
  try {
    const ch = await getChannel();
    
    await ch.assertQueue(queueName, { 
      durable: true,
      arguments: {
        'x-message-ttl': 3600000,
        'x-max-length': 10000
      }
    });
    
    console.log(`🔄 Starting consumer for queue: ${queueName}`);
    
    await ch.consume(queueName, async (msg) => {
      if (!msg) return;
      
      try {
        const payload = JSON.parse(msg.content.toString());
        
        await processor(
          payload,
          () => ch.ack(msg),
          () => ch.nack(msg, false, false)
        );
        
      } catch (error: any) {
        console.error(`❌ Error processing message from ${queueName}:`, error);
        ch.nack(msg, false, false); // Don't requeue on parse errors
      }
    });
    
  } catch (error: any) {
    console.error(`❌ Failed to consume from queue ${queueName}:`, error);
    throw error;
  }
}

export async function setupQueues(): Promise<void> {
  try {
    const ch = await getChannel();
    
    // Setup main processing queues
    const queues = [
      'evidence.process.queue',
      'evidence.process.control',
      'evidence.ocr.queue',
      'evidence.embedding.queue',
      'evidence.rag.queue'
    ];
    
    for (const queueName of queues) {
      await ch.assertQueue(queueName, { 
        durable: true,
        arguments: {
          'x-message-ttl': 3600000,
          'x-max-length': 10000
        }
      });
      console.log(`✅ Queue setup: ${queueName}`);
    }
    
    // Setup dead letter exchange for failed messages
    await ch.assertExchange('evidence.dlx', 'direct', { durable: true });
    await ch.assertQueue('evidence.failed', { 
      durable: true,
      arguments: {
        'x-message-ttl': 86400000, // 24 hours
      }
    });
    await ch.bindQueue('evidence.failed', 'evidence.dlx', 'failed');
    
    console.log('✅ RabbitMQ setup complete');
    
  } catch (error: any) {
    console.error('❌ Failed to setup RabbitMQ queues:', error);
    throw error;
  }
}

// Graceful shutdown
export async function closeRabbitMQ(): Promise<void> {
  try {
    if (channel) {
      await channel.close();
      channel = null;
    }
    if (connection) {
      await (connection as any).close();
      connection = null;
    }
    console.log('✅ RabbitMQ connections closed gracefully');
  } catch (error: any) {
    console.error('❌ Error closing RabbitMQ connections:', error);
  }
}

// Health check
export async function healthCheck(): Promise<boolean> {
  try {
    const ch = await getChannel();
    await ch.checkQueue('evidence.process.queue');
    return true;
  } catch (error: any) {
    console.error('❌ RabbitMQ health check failed:', error);
    return false;
  }
}
