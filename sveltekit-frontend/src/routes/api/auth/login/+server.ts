import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/drizzle';
import { users } from '$lib/server/db/schema-postgres';
import { authService, lucia } from '$lib/server/auth';
import { eq } from 'drizzle-orm';

export const POST: RequestHandler = async ({ request, cookies }) => {
  const { email, password } = await request.json();
  try {
    const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
    if (!user || !user.hashed_password) return new Response('Invalid credentials', { status: 401 });

    // Delegate password verification to AuthService (argon2id)
    const valid = await authService.login(email, password).catch(() => null);
    if (!valid) return new Response('Invalid credentials', { status: 401 });

    const session = await lucia.createSession(user.id, {});
    const sessionCookie = lucia.createSessionCookie(session.id);
    cookies.set(sessionCookie.name, sessionCookie.value, { ...sessionCookie.attributes, path: '/' });
    return new Response(JSON.stringify({ user: { id: user.id, email: user.email } }), { status: 200 });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: 'Login failed', detail: e.message }), { status: 500 });
  }
};