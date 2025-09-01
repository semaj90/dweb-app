import type { RequestHandler } from './$types';
import { db } from '$lib/db';
import { users } from '$lib/db/schema';
import bcrypt from "bcrypt";

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json();
  const { email, password } = body;

  if (!email || !password) return new Response('Missing', { status: 400 });

  const hash = await bcrypt.hash(password, 10);
  const res = await db.insert(users).values({
    email,
    password_hash: hash
  }).returning();

  const sessionToken = crypto.randomUUID();
  const headers = new Headers();
  headers.append('Set-Cookie', `session=${sessionToken}; HttpOnly; Path=/; Max-Age=${60*60*24}`);

  return new Response(JSON.stringify({ ok: true, user: res }), { status: 201, headers });
};
