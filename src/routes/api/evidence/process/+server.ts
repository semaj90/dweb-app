import type { RequestHandler } from '@sveltejs/kit';
import { v4 as uuidv4 } from 'uuid';
import { json } from '@sveltejs/kit';
import { publishToQueue } from '$lib/server/rabbitmq';
import { db } from '$lib/server/db';
import { evidenceProcess } from '$lib/database/schema/legal-documents';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    const { evidenceId, steps = ['ocr', 'embedding', 'analysis'] } = body;

    // Check authentication
    if (!locals.user) {
      return json({ error: 'Unauthorized' }, { status: 401 });
    }

    if (!evidenceId) {
      return json({ error: 'evidenceId required' }, { status: 400 });
    }

    // Validate steps
    const validSteps = ['ocr', 'embedding', 'rag', 'analysis'];
    const invalidSteps = steps.filter((step: string) => !validSteps.includes(step));
    if (invalidSteps.length > 0) {
      return json({ 
        error: `Invalid steps: ${invalidSteps.join(', ')}. Valid steps: ${validSteps.join(', ')}` 
      }, { status: 400 });
    }

    const sessionId = uuidv4();

    // Persist process request (drizzle)
    await db.insert(evidenceProcess).values({
      id: sessionId,
      evidenceId: evidenceId,
      requestedBy: locals.user.id,
      steps: steps,
      status: 'queued'
    });

    // Enqueue a job for worker(s) to pick up (RabbitMQ)
    await publishToQueue('evidence.process.queue', {
      sessionId,
      evidenceId,
      steps,
      userId: locals.user.id
    });

    return json({ 
      sessionId, 
      status: 'queued',
      steps,
      evidenceId
    });

  } catch (error) {
    console.error('Process evidence error:', error);
    return json({ 
      error: 'Failed to queue evidence processing',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};