import type { RequestHandler } from './$types';
import { db } from '$lib/db';
import { users } from '$lib/db/schema';
import { eq } from 'drizzle-orm';
import bcrypt from 'bcrypt';

export const POST: RequestHandler = async ({ request }) => {
  const { email, password } = await request.json();
  const row = await db.select().from(users).where(eq(users.email, email));
  const user = row[0];
  if (!user) return new Response('Not found', { status: 401 });
  const ok = await bcrypt.compare(password, user.password_hash);
  if (!ok) return new Response('Invalid', { status: 401 });

  const sessionToken = crypto.randomUUID();
  const headers = new Headers();
  headers.append('Set-Cookie', `session=${sessionToken}; HttpOnly; Path=/; Max-Age=${60*60*24}`);

  return new Response(JSON.stringify({ ok: true }), { status: 200, headers });
};
