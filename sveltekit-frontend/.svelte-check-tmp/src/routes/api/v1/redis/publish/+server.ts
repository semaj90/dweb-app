import { getRedisService } from '$lib/server/redis/redis-service';
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { channel, data } = await request.json();
    
    if (!channel || !data) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: 'Missing channel or data' 
      }), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const redisService = getRedisService();
    
    if (!redisService.isConnectedToRedis()) {
      return new Response(JSON.stringify({ 
        success: false, 
        error: 'Redis not connected' 
      }), { 
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Publish event based on channel type
    switch (channel) {
      case 'evidence_update':
        await redisService.publishEvidenceCreated(
          data.evidenceId,
          { fileName: data.fileName, caseId: data.caseId },
          data.userId
        );
        break;
      case 'case_update':
        await redisService.publishCaseUpdated(
          data.caseId,
          data.changes || {},
          data.userId
        );
        break;
      case 'canvas_update':
        if (data.type === 'CANVAS_NODE_ADDED') {
          await redisService.publishCanvasNodeAdded(
            data.caseId,
            data.nodeData,
            data.userId
          );
        }
        break;
      default:
        // Generic publish for custom channels
        await redisService.trackEvent(channel, data, data.userId);
    }

    return new Response(JSON.stringify({ 
      success: true, 
      message: 'Event published successfully',
      timestamp: new Date().toISOString()
    }), { 
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Redis publish error:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};