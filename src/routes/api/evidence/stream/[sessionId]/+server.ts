import { registerWsConnection } from '$lib/server/wsBroker';
import type { RequestHandler } from '@sveltejs/kit';

export const GET: RequestHandler = async ({ request, params }) => {
  const { sessionId } = params;

  if (!sessionId) {
    return new Response('Session ID required', { status: 400 });
  }

  const upgradeHeader = request.headers.get('upgrade') || '';
  
  if (upgradeHeader.toLowerCase() !== 'websocket') {
    return new Response('Upgrade to WebSocket required', { status: 426, headers: {
      'Upgrade': 'websocket'
    }});
  }

  try {
    // Note: This is a simplified WebSocket upgrade for demonstration
    // In production with SvelteKit + Node adapter, you'd typically use a dedicated WebSocket server
    // or libraries like ws, uws, or socket.io
    
    // For SvelteKit Node adapter with ws library:
    if ('upgradeWebSocket' in globalThis) {
      const { socket, response } = await (globalThis as any).upgradeWebSocket(request);
      registerWsConnection(sessionId, socket);
      return response;
    }

    // Fallback response for adapters that don't support WebSocket upgrade
    return new Response(
      `WebSocket endpoint for session: ${sessionId}\nConnect via WebSocket client to receive real-time progress updates.`,
      {
        status: 200,
        headers: {
          'Content-Type': 'text/plain',
          'X-Session-Id': sessionId
        }
      }
    );

  } catch (error) {
    console.error('WebSocket upgrade error:', error);
    return new Response('WebSocket upgrade failed', { status: 500 });
  }
};