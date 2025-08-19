// Recent pipeline logs endpoint
// Returns latest N pipeline_logs entries including embedding_hash for debugging
import { db } from '$lib/db/drizzle';
import { pipeline_logs } from '$lib/db/schema';
import { desc } from 'drizzle-orm';

export async function GET(req: Request) {
  const url = new URL(req.url);
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '25', 10), 200);
  try {
    const rows = await db.select().from(pipeline_logs).orderBy(desc(pipeline_logs.id)).limit(limit);
    return new Response(JSON.stringify({ ok: true, count: rows.length, rows }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ ok: false, error: e.message }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}
