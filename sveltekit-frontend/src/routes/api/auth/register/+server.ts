import type { RequestHandler } from './$types.js';
import { db } from '$lib/database/connection';
import { users, sessions } from '$lib/database/schema';
import { hash } from '@node-rs/argon2';
import crypto from "crypto";

export const POST: RequestHandler = async ({ request, cookies }) => {
  const { email, password } = await request.json();
  if (!email || !password) return new Response('Missing fields', { status: 400 });

  const passwordHash = await hash(password);
  const [user] = await db.insert(users).values({ email, passwordHash }).returning();

  const sessionId = crypto.randomUUID();
  await db.insert(sessions).values({
    id: sessionId,
    userId: user.id,
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000)
  });

  cookies.set('session', sessionId, {
    httpOnly: true,
    path: '/',
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 60 * 60 * 24
  });
  return new Response(JSON.stringify({ user }));
};