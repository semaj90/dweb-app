
import type { RequestHandler } from './$types';

// routes/api/ai-synthesizer/stream/[streamId]/+server.ts
// Server-Sent Events endpoint for streaming AI synthesis updates

import { streamingService } from "$lib/server/ai/streaming-service";

export const GET: RequestHandler = async ({ params, request }) => {
  const { streamId } = params;

  if (!streamId) {
    throw error(400, 'Stream ID is required');
  }

  logger.info(`[AI-Synthesizer] Opening stream connection: ${streamId}`);

  // Create SSE response
  const headers = new Headers({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no' // Disable nginx buffering
  });

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      // Subscribe to stream updates
      const unsubscribe = streamingService.subscribe(streamId, (event: any) => {
        const data = `event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`;
        controller.enqueue(encoder.encode(data));

        // Close stream on completion or error
        if (event.type === 'complete' || event.type === 'error') {
          setTimeout(() => {
            controller.close();
            unsubscribe();
          }, 100);
        }
      });

      // Send heartbeat every 30 seconds
      const heartbeatInterval = setInterval(() => {
        const heartbeat = `event: heartbeat\ndata: ${JSON.stringify({ timestamp: Date.now() })}\n\n`;
        try {
          controller.enqueue(encoder.encode(heartbeat));
        } catch (err: any) {
          clearInterval(heartbeatInterval);
        }
      }, 30000);

      // Cleanup on client disconnect
      request.signal.addEventListener('abort', () => {
        clearInterval(heartbeatInterval);
        unsubscribe();
        controller.close();
        logger.info(`[AI-Synthesizer] Stream ${streamId} closed by client`);
      });
    }
  });

  return new Response(stream, { headers });
};
