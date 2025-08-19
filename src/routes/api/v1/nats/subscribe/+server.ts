import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getNATSService, NATS_SUBJECTS } from '$lib/services/nats-messaging-service.js';

/**
 * POST /api/v1/nats/subscribe
 * Setup WebSocket subscription to NATS subjects
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { subject, options = {} } = await request.json();

    if (!subject) {
      return json({ 
        error: 'Missing required field: subject' 
      }, { status: 400 });
    }

    // Validate subject (allow wildcards for flexibility)
    const validSubjects = Object.values(NATS_SUBJECTS);
    const isValidSubject = validSubjects.includes(subject) || 
                          subject.includes('*') || 
                          subject.includes('>');

    if (!isValidSubject) {
      return json({ 
        error: `Invalid subject. Must be one of: ${validSubjects.join(', ')} or use wildcards` 
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

    // Subscribe to subject
    const subscriptionId = await natsService.subscribe(subject, (message) => {
      // Message handling would typically involve WebSocket or SSE
      console.log(`API subscription received message on ${subject}:`, message);
    }, options);

    return json({
      success: true,
      subscriptionId,
      subject,
      message: `Subscribed to ${subject}`,
      options,
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('NATS subscribe API error:', error);
    return json({ 
      error: 'Failed to subscribe to subject',
      details: error.message 
    }, { status: 500 });
  }
};

/**
 * DELETE /api/v1/nats/subscribe
 * Unsubscribe from NATS subject
 */
export const DELETE: RequestHandler = async ({ request }) => {
  try {
    const { subscriptionId } = await request.json();

    if (!subscriptionId) {
      return json({ 
        error: 'Missing required field: subscriptionId' 
      }, { status: 400 });
    }

    const natsService = getNATSService();
    if (!natsService) {
      return json({ 
        error: 'NATS service not available' 
      }, { status: 503 });
    }

    // Unsubscribe
    await natsService.unsubscribe(subscriptionId);

    return json({
      success: true,
      subscriptionId,
      message: `Unsubscribed from subscription ${subscriptionId}`,
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('NATS unsubscribe API error:', error);
    return json({ 
      error: 'Failed to unsubscribe',
      details: error.message 
    }, { status: 500 });
  }
};