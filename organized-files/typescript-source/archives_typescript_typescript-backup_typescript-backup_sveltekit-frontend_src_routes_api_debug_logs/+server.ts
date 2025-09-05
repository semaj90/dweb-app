import { json } from '@sveltejs/kit';

export const GET = async (): Promise<any> => {
  if (process.env.NODE_ENV !== 'development') {
    return new Response('Not allowed', { status: 403 });
  }

  const logs = (globalThis as any)._devLogs || [];
  return json({
    env: process.env.NODE_ENV,
    timestamp: new Date().toISOString(),
    logs: logs.slice(0, 200),
  });
};
