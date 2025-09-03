import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/drizzle';
import { users } from '$lib/server/db/schema-postgres';
import { authService, lucia } from '$lib/server/auth';

export const POST: RequestHandler = async ({ request, cookies }) => {
  const { email, password, firstName, lastName } = await request.json();
  if (!email || !password) return new Response('Missing fields', { status: 400 });
  try {
    const newUser = await authService.register({ email, password, firstName, lastName });
    const session = await lucia.createSession(newUser.id, {});
    const sessionCookie = lucia.createSessionCookie(session.id);
    cookies.set(sessionCookie.name, sessionCookie.value, { ...sessionCookie.attributes, path: '/' });
    return new Response(JSON.stringify({ user: { id: newUser.id, email: newUser.email } }), { status: 201 });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: 'Registration failed', detail: e.message }), { status: 500 });
  }
};