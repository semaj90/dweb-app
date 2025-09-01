import type { RequestHandler } from './$types.js';

// src/routes/api/evidence/stream/[sessionId]/+server.ts
import { registerWsConnection, getMissedMessages } from "drizzle-orm";
import WebSocket from "ws";
import { URL } from "url";

export const GET: RequestHandler = async ({ request, params, url }) => {
  const { sessionId } = params;
  
  if (!sessionId) {
    return new Response('Session ID required', { status: 400 });
  }
  
  const upgradeHeader = request.headers.get('upgrade')?.toLowerCase();
  
  if (upgradeHeader !== 'websocket') {
    // Non-WebSocket request - return missed messages via HTTP
    try {
      const since = url.searchParams.get('since');
      const messages = await getMissedMessages(sessionId, since || undefined);
      
      return new Response(JSON.stringify({
        sessionId,
        messages,
        timestamp: new Date().toISOString()
      }), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache'
        }
      });
      
    } catch (error: any) {
      console.error('❌ Error getting missed messages:', error);
      return new Response('Internal Server Error', { status: 500 });
    }
  }
  
  // WebSocket upgrade request
  try {
    // Note: This is a simplified example. In production, you'll need to handle
    // WebSocket upgrades according to your specific SvelteKit adapter
    // For Node.js adapter, you might need to use a different approach
    
    return new Response('WebSocket upgrade not supported in this route handler. Use dedicated WebSocket server.', {
      status: 426,
      headers: {
        'Upgrade': 'websocket',
        'Connection': 'Upgrade'
      }
    });
    
  } catch (error: any) {
    console.error('❌ WebSocket upgrade error:', error);
    return new Response('WebSocket upgrade failed', { status: 500 });
  }
};

// For development/testing - simple server-sent events alternative
export const POST: RequestHandler = async ({ request, params }) => {
  const { sessionId } = params;
  
  if (!sessionId) {
    return new Response('Session ID required', { status: 400 });
  }
  
  try {
    const { action, ...data } = await request.json();
    
    if (action === 'subscribe') {
      // Return server-sent events stream
      const encoder = new TextEncoder();
      
      const stream = new ReadableStream({
        start(controller) {
          // Send initial connection event
          const message = `data: ${JSON.stringify({
            type: 'connection-established',
            sessionId,
            timestamp: new Date().toISOString()
          })}\n\n`;
          
          controller.enqueue(encoder.encode(message));
          
          // TODO: Set up subscription to receive progress messages
          // and forward them to the controller
          
          // For now, just keep the connection alive
          const keepAlive = setInterval(() => {
            try {
              controller.enqueue(encoder.encode('data: {"type":"heartbeat"}\n\n'));
            } catch (error: any) {
              clearInterval(keepAlive);
            }
          }, 30000);
          
          // Cleanup after 1 hour
          setTimeout(() => {
            clearInterval(keepAlive);
            controller.close();
          }, 3600000);
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
    }
    
    return new Response('Invalid action', { status: 400 });
    
  } catch (error: any) {
    console.error('❌ SSE setup error:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
};
