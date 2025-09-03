// Database connection using direct drizzle import to avoid circular imports
export { db } from './drizzle';
export * from './unified-schema';

// Health check function
export async function dbHealth() { 
  try { 
    const { db } = await import('./drizzle');
    await db.execute('SELECT 1'); 
    return { ok: true }; 
  } catch (e: any) { 
    return { ok: false, error: e.message }; 
  } 
}