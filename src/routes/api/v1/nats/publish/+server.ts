import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getNATSService, NATS_SUBJECTS } from '$lib/services/nats-messaging-service.js';

/**
 * POST /api/v1/nats/publish
 * Publish messages to NATS subjects
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { subject, data, options = {} } = await request.json();

    if (!subject || !data) {
      return json({ 
        error: 'Missing required fields: subject and data' 
      }, { status: 400 });
    }

    // Validate subject
    const validSubjects = Object.values(NATS_SUBJECTS);
    if (!validSubjects.includes(subject)) {
      return json({ 
        error: `Invalid subject. Must be one of: ${validSubjects.join(', ')}` 
      }, { status: 400 });
    }

    const natsService = getNATSService();
    if (!natsService) {
      return json({ 
        error: 'NATS service not available' 
      }, { status: 503 });
    }

    if (!natsService.isConnected) {
      return json({ 
        error: 'NATS service not connected' 
      }, { status: 503 });
    }

    // Publish message
    await natsService.publish(subject, data, {
      metadata: {
        source: 'api-endpoint',
        priority: options.priority || 'normal',
        ...options.metadata
      },
      ...options
    });

    return json({
      success: true,
      message: `Message published to ${subject}`,
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('NATS publish API error:', error);
    return json({ 
      error: 'Failed to publish message',
      details: error.message 
    }, { status: 500 });
  }
};