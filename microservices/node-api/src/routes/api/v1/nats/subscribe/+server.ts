import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { subscribe } from '$lib/services/nats-messaging-service';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { subject } = await request.json();
    if(!subject){
      return json({ error: 'subject required' }, { status: 400 });
    }
    subscribe(subject, (msg) => {
      console.log('Received message on', subject, msg);
      // TODO: integrate GPU / WASM / RAG pipeline here
    });
    return json({ status: 'subscribed', subject });
  } catch (err: unknown) {
    console.error('Subscribe error:', err);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
};
