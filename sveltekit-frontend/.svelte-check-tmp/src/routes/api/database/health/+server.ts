import { json } from "@sveltejs/kit";
import { testDatabaseConnection, initializeDatabase } from '$lib/database/connection';
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async () => {
  try {
    console.log('🔍 Testing database connection...');
    
    // Test basic connection
    const healthCheck = await testDatabaseConnection();
    
    if (!healthCheck.success) {
      console.error('❌ Database connection failed:', healthCheck.message);
      return json({
        status: 'error',
        message: healthCheck.message,
        details: healthCheck.details,
        timestamp: new Date().toISOString()
      }, { status: 500 });
    }

    console.log('✅ Database connection successful');
    
    // Try to initialize if not already done
    let initResult = null;
    try {
      initResult = await initializeDatabase();
    } catch (error: any) {
      console.warn('⚠️ Database initialization error (may already be initialized):', error);
    }

    return json({
      status: 'healthy',
      message: 'Database connection successful',
      connection: healthCheck.details,
      initialization: initResult,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('❌ Database health check failed:', error);
    
    return json({
      status: 'error',
      message: `Database health check failed: ${(error as Error).message}`,
      error: (error as Error).stack,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action } = await request.json();

    if (action === 'initialize') {
      console.log('🔧 Initializing database...');
      const result = await initializeDatabase();
      
      return json({
        status: result.success ? 'success' : 'error',
        message: result.message,
        details: result.details || null,
        timestamp: new Date().toISOString()
      });
    }

    return json({
      status: 'error',
      message: 'Invalid action. Use "initialize"',
      timestamp: new Date().toISOString()
    }, { status: 400 });

  } catch (error: any) {
    console.error('❌ Database action failed:', error);
    
    return json({
      status: 'error',
      message: `Database action failed: ${(error as Error).message}`,
      error: (error as Error).stack,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};