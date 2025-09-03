import type { PageServerLoad } from './$types';

// Simple server load that proxies (placeholder) or returns diagnostic
export const load: PageServerLoad = async ({ fetch, url }) => {
  // Placeholder: ensure page exists to prevent 404
  return {
    message: 'Proxy route ready',
    targetExample: '/api/status',
    now: new Date().toISOString(),
    path: url.pathname
  };
};
