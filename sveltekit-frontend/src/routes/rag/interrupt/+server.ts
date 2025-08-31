import { json } from '@sveltejs/kit';
import { interruptStream } from "$lib/server/ragStreamRegistry";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { streamId, mode = 'graceful' } = await request.json();
    if (!streamId) return json({ ok: false, error: 'streamId required' }, { status: 400 });
    const ok = interruptStream(streamId, mode);
    return json({ ok, streamId, mode });
  } catch (e: any) {
    return json({ ok: false, error: e.message }, { status: 500 });
  }
};
