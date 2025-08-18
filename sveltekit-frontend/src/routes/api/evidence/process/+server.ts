// src/routes/api/evidence/process/+server.ts
import type { RequestHandler } from '@sveltejs/kit';
import { v4 as uuidv4 } from 'uuid';
import { json } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { publishToQueue } from '$lib/server/rabbitmq';
import type { EvidenceProcessRequest } from '$lib/types/progress';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body: EvidenceProcessRequest = await request.json();
    const { evidenceId, steps = ['ocr', 'embedding', 'analysis'] } = body;

    // Authentication check
    if (!locals.user) {
      return new Response('Unauthorized', { status: 401 });
    }

    // Validation
    if (!evidenceId) {
      return json({ error: 'evidenceId required' }, { status: 400 });
    }

    const sessionId = uuidv4();

    // Persist process request (drizzle)
    await db.insert(evidenceProcessTable).values({
      id: sessionId,
      evidence_id: evidenceId,
      requested_by: locals.user.id,
      steps: JSON.stringify(steps),
      status: 'queued',
      created_at: new Date()
    });

    // Enqueue a job for worker(s) to pick up (RabbitMQ)
    await publishToQueue('evidence.process.queue', {
      sessionId,
      evidenceId,
      steps,
      userId: locals.user.id,
      timestamp: new Date().toISOString()
    });

    console.log(`📋 Evidence processing queued: ${sessionId} for evidence: ${evidenceId}`);

    return json({ 
      sessionId, 
      status: 'queued',
      message: 'Evidence processing request queued successfully'
    });

  } catch (error) {
    console.error('❌ Error in evidence processing endpoint:', error);
    return json(
      { error: 'Internal server error processing evidence request' }, 
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    if (!locals.user) {
      return new Response('Unauthorized', { status: 401 });
    }

    const sessionId = url.searchParams.get('sessionId');
    
    if (sessionId) {
      // Get specific session status
      const session = await db
        .select()
        .from(evidenceProcessTable)
        .where(eq(evidenceProcessTable.id, sessionId))
        .limit(1);

      if (session.length === 0) {
        return json({ error: 'Session not found' }, { status: 404 });
      }

      return json(session[0]);
    } else {
      // Get all sessions for user
      const sessions = await db
        .select()
        .from(evidenceProcessTable)
        .where(eq(evidenceProcessTable.requested_by, locals.user.id))
        .orderBy(desc(evidenceProcessTable.created_at))
        .limit(50);

      return json({ sessions });
    }

  } catch (error) {
    console.error('❌ Error getting evidence process status:', error);
    return json(
      { error: 'Internal server error' }, 
      { status: 500 }
    );
  }
};

export const DELETE: RequestHandler = async ({ request, locals }) => {
  try {
    const { sessionId } = await request.json();

    if (!locals.user) {
      return new Response('Unauthorized', { status: 401 });
    }

    if (!sessionId) {
      return json({ error: 'sessionId required' }, { status: 400 });
    }

    // Update status to cancelled
    await db
      .update(evidenceProcessTable)
      .set({ 
        status: 'cancelled',
        finished_at: new Date()
      })
      .where(eq(evidenceProcessTable.id, sessionId));

    // Send cancellation message to worker
    await publishToQueue('evidence.process.control', {
      action: 'cancel',
      sessionId,
      timestamp: new Date().toISOString()
    });

    return json({ 
      message: 'Processing cancelled successfully',
      sessionId 
    });

  } catch (error) {
    console.error('❌ Error cancelling evidence processing:', error);
    return json(
      { error: 'Internal server error' }, 
      { status: 500 }
    );
  }
};

// Schema imports (you'll need to define these in your schema)
import { evidenceProcessTable } from '$lib/server/schema';
import { eq, desc } from 'drizzle-orm';
