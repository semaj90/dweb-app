import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
  try {
    // Test basic connection without database imports
    return json({ 
      status: 'ok', 
      message: 'Database test endpoint works',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Database test error:', error);
    return json({ 
      status: 'error', 
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};