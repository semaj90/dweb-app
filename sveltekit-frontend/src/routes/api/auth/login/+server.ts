import type { RequestHandler } from './$types.js';
import { db } from '$lib/database/connection';
import { users, sessions } from '$lib/database/schema';
import { verify } from '@node-rs/argon2';
import { eq } from 'drizzle-orm';
import crypto from "crypto";

export const POST: RequestHandler = async ({ request, cookies }) => {
  const { email, password } = await request.json();

  const [user] = await db.select().from(users).where(eq(users.email, email));
  if (!user || !(await verify(user.passwordHash, password))) {
    return new Response('Invalid credentials', { status: 401 });
  }

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
  return new Response(JSON.stringify({ user: { id: user.id, email: user.email } }));
};