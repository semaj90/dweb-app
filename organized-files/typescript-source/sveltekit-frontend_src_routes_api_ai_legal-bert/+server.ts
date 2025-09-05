import { json, type RequestHandler } from '@sveltejs/kit';
import { predictLegalBert } from '$lib/grpc/legal-bert-client';

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json();
  const text = body?.text || body?.query || '';
  if (!text) return json({ error: 'missing text' }, { status: 400 });

  try {
    const resp = await predictLegalBert(text, { protoPath: 'protos/legal_bert.proto' });
    return json({ ok: true, result: resp });
  } catch (err: any) {
    return json({ ok: false, error: err?.message || String(err) }, { status: 500 });
  }
};
