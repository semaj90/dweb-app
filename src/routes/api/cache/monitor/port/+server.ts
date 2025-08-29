import { json, type RequestHandler } from '@sveltejs/kit';

/**
 * API endpoint to provide cache monitoring WebSocket port information
 * This allows the frontend to discover the dynamic WebSocket port
 */

export const GET: RequestHandler = async (): Promise<any> => {
    try {
        // Method 1: Try to get from environment variable
        const envPort = process.env.PUBLIC_CACHE_MONITOR_WS_PORT;
        if (envPort) {
            const port = parseInt(envPort, 10);
            if (!isNaN(port)) {
                return json({
                    port,
                    source: 'environment',
                    url: `ws://localhost:${port}`,
                    status: 'available'
                });
            }
        }

        // Method 2: Try to import and get from service instance
        try {
            // @ts-ignore - Dynamic import may not exist
            const { CacheMonitoringService } = await import('$lib/websockets/cache-monitoring-service');
            const port = CacheMonitoringService.getGlobalPort();
            
            if (port) {
                return json({
                    port,
                    source: 'service-instance',
                    url: `ws://localhost:${port}`,
                    status: 'available'
                });
            }
        } catch (error: any) {
            console.warn('Could not import cache monitoring service:', error);
        }

        // Method 3: Return default port as fallback
        const defaultPort = 9002;
        return json({
            port: defaultPort,
            source: 'default',
            url: `ws://localhost:${defaultPort}`,
            status: 'unknown',
            warning: 'Using default port - actual service may be on different port'
        });

    } catch (error: any) {
        console.error('Error getting cache monitoring port:', error);
        return json({
            error: 'Could not determine cache monitoring WebSocket port',
            details: (error as Error).message
        }, { status: 500 });
    }
};

// Also provide service status information
export const POST: RequestHandler = async (): Promise<any> => {
    try {
        // Try to get service statistics
        // @ts-ignore - Dynamic import may not exist
        const { cacheMonitoringService } = await import('$lib/websockets/cache-monitoring-service');
        
        if (cacheMonitoringService) {
            const stats = cacheMonitoringService.getServerStats();
            return json({
                status: 'running',
                stats,
                timestamp: new Date().toISOString()
            });
        }

        return json({
            status: 'not-running',
            message: 'Cache monitoring service not initialized',
            timestamp: new Date().toISOString()
        });

    } catch (error: any) {
        return json({
            status: 'error',
            error: (error as Error).message,
            timestamp: new Date().toISOString()
        }, { status: 500 });
    }
};