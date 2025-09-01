import type { Handle } from '@sveltejs/kit';

// Critical polyfill fix for SvelteKit runtime process.cwd issue
if (typeof globalThis.process === 'undefined') {
  globalThis.process = {
    cwd: () => '/',
    env: { NODE_ENV: 'production', BROWSER: 'false' },
    browser: false,
    version: 'v18.0.0',
    versions: { node: '18.0.0' }
  } as any;
} else if (typeof globalThis.process.cwd !== 'function') {
  globalThis.process.cwd = () => '/';
}

const sessionStorage = new Map<string, string>();

export const handle: Handle = async ({ event, resolve }) => {
  const session = event.cookies.get('session');
  if (session) {
    const userId = sessionStorage.get(session);
    if (userId) {
      event.locals.user = { id: userId };
    }
  }
  return await resolve(event);
};
