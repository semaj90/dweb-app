import type { RequestHandler } from '@sveltejs/kit';
import type { RequestHandler } from "./$types";

import { json } from "@sveltejs/kit";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, key, data, dataType } = await request.json();
    
    switch (action) {
      case 'set':
        await cacheManager.set(key, data, dataType || 'generic', 300); // 5 min TTL
        return json({ success: true, message: `Data set for key: ${key}` });
      
      case 'get':
        const result = await cacheManager.get(key, dataType || 'generic');
        return json({ success: true, data: result, key });
      
      case 'stats':
        const stats = cacheManager.getLayerStats();
        return json({ success: true, stats });
      
      default:
        return json({ success: false, error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('CRUD test error:', error);
    return json({ success: false, error: error.message }, { status: 500 });
  }
};

export const GET: RequestHandler = async () => {
  try {
    // Test all cache layers
    const testKey = `test-${Date.now()}`;
    const testData = { 
      message: 'Hello Cache!', 
      timestamp: new Date().toISOString(),
      layers: ['memory', 'redis', 'qdrant', 'postgres', 'neo4j']
    };
    
    // Set test data
    await cacheManager.set(testKey, testData, 'test', 300);
    
    // Get test data
    const retrieved = await cacheManager.get(testKey, 'test');
    
    // Get stats
    const stats = cacheManager.getLayerStats();
    
    return json({
      success: true,
      test: {
        key: testKey,
        original: testData,
        retrieved,
        dataMatch: JSON.stringify(testData) === JSON.stringify(retrieved)
      },
      stats
    });
  } catch (error) {
    console.error('CRUD GET test error:', error);
    return json({ success: false, error: error.message }, { status: 500 });
  }
};