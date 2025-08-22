import { fail, redirect } from '@sveltejs/kit';
import { lucia } from '$lib/server/auth';
import { db, users } from '$lib/server/db';
import { eq } from 'drizzle-orm';
import bcrypt from 'bcryptjs';
import type { PageServerLoad, Actions } from './$types';

export const load: PageServerLoad = async () => {
  return {};
};

export const actions: Actions = {
  register: async ({ request, cookies }) => {
    const data = await request.formData();
    const email = data.get('email') as string;
    const firstName = data.get('firstName') as string;
    const lastName = data.get('lastName') as string;
    const password = data.get('password') as string;
    const confirmPassword = data.get('confirmPassword') as string;
    const role = data.get('role') as string;
    const department = data.get('department') as string;
    const jurisdiction = data.get('jurisdiction') as string;
    const badgeNumber = data.get('badgeNumber') as string;

    // Basic validation
    if (!email || !firstName || !lastName || !password || !department || !jurisdiction) {
      return fail(400, { error: 'All required fields must be filled' });
    }

    if (password !== confirmPassword) {
      return fail(400, { error: 'Passwords do not match' });
    }

    if (password.length < 8) {
      return fail(400, { error: 'Password must be at least 8 characters' });
    }

    try {
      // Check if user already exists
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, email.toLowerCase()))
        .limit(1);

      if (existingUser.length > 0) {
        return fail(400, { error: 'An account with this email already exists' });
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(password, 10);

      // Create user with basic fields that match the unified schema
      const newUser = await db
        .insert(users)
        .values({
          email: email.toLowerCase(),
          passwordHash: hashedPassword,
          displayName: `${firstName} ${lastName}`,
          role: role || 'user',
          createdAt: new Date(),
          updatedAt: new Date()
        })
        .returning({ id: users.id });

      const userId = newUser[0].id;

      // Create a session
      const session = await lucia.createSession(userId, {});
      const sessionCookie = lucia.createSessionCookie(session.id);

      cookies.set(sessionCookie.name, sessionCookie.value, {
        path: '/',
        ...sessionCookie.attributes
      });

      console.log('User registered successfully:', { userId, email });

    } catch (error) {
      console.error('Registration error:', error);
      return fail(500, { error: 'An error occurred during registration. Please try again.' });
    }

    // Redirect to dashboard
    throw redirect(302, '/dashboard');
  }
};