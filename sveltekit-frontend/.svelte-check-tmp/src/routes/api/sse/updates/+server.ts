
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ setHeaders }) => {
    setHeaders({
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });

    // Example: send a single event and close
    return new Response('data: Hello from SSE!\n\n');
};
