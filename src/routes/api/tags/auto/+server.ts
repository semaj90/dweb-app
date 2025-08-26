import { json, error } from '@sveltejs/kit';
import { createClient } from 'redis';
import envConfig from '../../../../../env-config.mjs';
import type { RequestHandler } from './$types';

// POST /api/tags/auto - Enqueue auto-tagging job
export const POST: RequestHandler = async ({ locals, request }) => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  try {
    const body = await request.json();
    const { evidenceId, content, filename } = body;

    if (!evidenceId || !content) {
      throw error(400, 'Evidence ID and content are required');
    }

    // Create Redis client
    const redis = createClient({ url: envConfig.REDIS_URL });
    await redis.connect();

    // Create auto-tagging job
    const job = {
      id: `autotag_${evidenceId}_${Date.now()}`,
      evidenceId,
      content,
      filename: filename || 'unknown',
      userId: locals.user.id,
      timestamp: new Date().toISOString(),
      priority: 'normal'
    };

    // Add to Redis Stream
    await redis.xAdd('autotag:requests', '*', {
      job: JSON.stringify(job)
    });

    await redis.quit();

    return json({
      success: true,
      data: {
        jobId: job.id,
        message: 'Auto-tagging job queued successfully'
      }
    });

  } catch (err) {
    console.error('Error queuing auto-tagging job:', err);
    throw error(500, 'Failed to queue auto-tagging job');
  }
};

// GET /api/tags/auto?jobId=xxx - Check job status
export const GET: RequestHandler = async ({ locals, url }) => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  const jobId = url.searchParams.get('jobId');
  
  if (!jobId) {
    throw error(400, 'Job ID is required');
  }

  try {
    // Create Redis client
    const redis = createClient({ url: envConfig.REDIS_URL });
    await redis.connect();

    // Check job status in completed jobs stream
    const results = await redis.xRead(
      redis.commandOptions({ returnBuffers: false }),
      [
        {
          key: 'autotag:completed',
          id: '0'
        }
      ],
      {
        COUNT: 100
      }
    );

    await redis.quit();

    // Look for our job in completed results
    let jobResult = null;
    if (results && results.length > 0) {
      for (const stream of results) {
        for (const message of stream.messages) {
          const data = message.message;
          if (data.jobId === jobId) {
            jobResult = {
              status: 'completed',
              result: JSON.parse(data.result || '{}'),
              completedAt: data.completedAt
            };
            break;
          }
        }
      }
    }

    if (!jobResult) {
      jobResult = {
        status: 'pending',
        message: 'Job is still being processed'
      };
    }

    return json({
      success: true,
      data: jobResult
    });

  } catch (err) {
    console.error('Error checking job status:', err);
    throw error(500, 'Failed to check job status');
  }
};