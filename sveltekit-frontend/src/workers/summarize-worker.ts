// Minimal RabbitMQ worker that calls the Go SIMD summarize endpoint and writes to Loki cache
import { connect } from 'amqplib';
// Cleaned orphaned import corruption
const RABBITMQ_URL = (import.meta as any).env?.RABBITMQ_URL || 'amqp://localhost';
const QUEUE_NAME = 'summarization_tasks';
const SIMD_URL = (import.meta as any).env?.SIMD_URL || 'http://localhost:8081';

// Minimal in-memory cache placeholder
const _cache = new Map<string, any>();
async function setCache(k: string, v: any) { _cache.set(k, { v, ts: Date.now() }); }

async function processTask(msg: { content: Buffer } | null) {
  if (!msg) return;
  try {
    const task = JSON.parse(msg.content.toString());
    const { documentId, content } = task;

    const resp = await fetch(`${SIMD_URL}/api/simd/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    const data = await resp.json();

    await setCache(documentId, { status: "complete", data });
    console.log(`✓ Summarized and cached ${documentId}`);
  } catch (e) {
    const err = e as Error;
    console.error("Worker error:", (err && err.message) || e);
  }
}

async function start() {
  const connection = await connect(RABBITMQ_URL);
  const channel = await connection.createChannel();
  await channel.assertQueue(QUEUE_NAME, { durable: true });
  console.log("[*] Waiting for tasks. To exit press CTRL+C");
  channel.consume(QUEUE_NAME, async (msg) => {
    await processTask(msg);
    channel.ack(msg!);
  });
}

start().catch((e) => {
  console.error("Failed to start worker:", e);
  process.exit(1);
});
