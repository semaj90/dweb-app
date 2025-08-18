import amqp from 'amqplib';

let channel: amqp.Channel | null = null;
let connection: amqp.Connection | null = null;

export async function getChannel() {
  if (channel) return channel;
  
  const conn = await amqp.connect(process.env.RABBITMQ_URL || 'amqp://localhost');
  connection = conn;
  channel = await conn.createChannel();
  
  // Handle connection errors
  conn.on('error', (err) => {
    console.error('RabbitMQ connection error:', err);
    channel = null;
    connection = null;
  });
  
  conn.on('close', () => {
    console.warn('RabbitMQ connection closed');
    channel = null;
    connection = null;
  });
  
  return channel;
}

export async function publishToQueue(queue: string, payload: any) {
  const ch = await getChannel();
  await ch.assertQueue(queue, { durable: true });
  ch.sendToQueue(queue, Buffer.from(JSON.stringify(payload)), { persistent: true });
}

export async function consumeFromQueue(queue: string, handler: (msg: any) => Promise<void>) {
  const ch = await getChannel();
  await ch.assertQueue(queue, { durable: true });
  
  ch.consume(queue, async (msg) => {
    if (!msg) return;
    
    try {
      const payload = JSON.parse(msg.content.toString());
      await handler(payload);
      ch.ack(msg);
    } catch (err) {
      console.error('Queue handler error:', err);
      ch.nack(msg, false, false); // drop message on error
    }
  });
}

export async function closeConnection() {
  if (connection) {
    await connection.close();
    connection = null;
    channel = null;
  }
}