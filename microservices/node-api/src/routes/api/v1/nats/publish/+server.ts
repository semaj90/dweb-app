import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { NATS_SUBJECTS, publishMessage } from '$lib/services/nats-messaging-service';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { subject, payload } = await request.json();
    if (!subject || payload === undefined){
      return json({ error: 'subject and payload required' }, { status: 400 });
    }
    if (!Object.values(NATS_SUBJECTS).includes(subject)){
      return json({ error: 'invalid subject' }, { status: 400 });
    }
    await publishMessage(subject, payload);
    return json({ status: 'ok', subject, payload });
  } catch (err: any) {
    console.error('NATS publish error:', err);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
};
