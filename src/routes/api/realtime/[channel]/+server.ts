import type { RequestHandler } from './$types';
import { error } from '@sveltejs/kit';
import { createClient } from 'redis';
import envConfig from '../../../../../env-config.mjs';

// GET /api/realtime/[channel] - Server-Sent Events endpoint
export const GET: RequestHandler = async ({ locals, params }) => {
  if (!locals.user) {
    return new Response('Unauthorized', { status: 401 });
  }

  const channel = params.channel;
  
  // Validate channel
  const allowedChannels = [
    'realtime:evidence',
    'realtime:reports', 
    'realtime:cases'
  ];
  
  if (!allowedChannels.includes(channel)) {
    throw error(400, 'Invalid channel');
  }

  try {
    // Create Redis client
    const redis = createClient({ url: envConfig.REDIS_URL });
    await redis.connect();

    // Create readable stream for SSE
    const stream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        
        // Send initial connection message
        controller.enqueue(encoder.encode('event: connected\ndata: {"status":"connected"}\n\n'));

        // Subscribe to Redis channel
        redis.subscribe(channel, (message) => {
          try {
            const data = JSON.parse(message);
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
          } catch (err) {
            console.error('Error parsing Redis message:', err);
            controller.enqueue(encoder.encode(`data: ${message}\n\n`));
          }
        });

        // Ping every 30 seconds to keep connection alive
        const pingInterval = setInterval(() => {
          controller.enqueue(encoder.encode('event: ping\ndata: {}\n\n'));
        }, 30000);

        // Store cleanup function
        (controller as any)._cleanup = async () => {
          clearInterval(pingInterval);
          await redis.unsubscribe(channel);
          await redis.quit();
        };
      },

      async cancel() {
        // Cleanup when client disconnects
        await (this as any)._cleanup?.();
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      }
    });

  } catch (err) {
    console.error('Error setting up SSE:', err);
    throw error(500, 'Failed to establish real-time connection');
  }
};