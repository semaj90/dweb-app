import { fail, redirect } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';
import { simpleAuthService } from '$lib/server/auth-simple';
import { createUserSession, setSessionCookie, verifyPassword } from '$lib/server/lucia';
import { db } from '$lib/server/db';
import { users } from '$lib/server/db';
import { eq } from 'drizzle-orm';
import type { PageServerLoad, Actions } from './$types';

const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  rememberMe: z.boolean().default(false),
  twoFactorCode: z.string().optional()
});

export const load: PageServerLoad = async () => {
  const form = await superValidate(zod(loginSchema));
  return { form };
};

export const actions: Actions = {
  login: async ({ request, cookies, getClientAddress }) => {
    const form = await superValidate(request, zod(loginSchema));

    if (!form.valid) {
      return fail(400, { form });
    }

    const { email, password, rememberMe, twoFactorCode } = form.data;
    const clientIP = getClientAddress();
    const userAgent = request.headers.get('user-agent') || '';

    try {
      // Find user in database
      console.log('🔄 Authenticating user with corrected session system...');
      
      const userRecord = await db.query.users.findFirst({
        where: eq(users.email, email.toLowerCase()),
        columns: {
          id: true,
          email: true,
          hashed_password: true,
          first_name: true,
          last_name: true,
          role: true,
          is_active: true
        }
      });

      if (!userRecord || !userRecord.hashed_password) {
        throw new Error('Invalid email or password');
      }

      if (!userRecord.is_active) {
        throw new Error('Account is deactivated');
      }

      // Verify password
      const passwordValid = await verifyPassword(userRecord.hashed_password, password);
      if (!passwordValid) {
        throw new Error('Invalid email or password');
      }
      
      console.log('✅ User authenticated successfully:', userRecord.email);

      // Create session with IP and user agent
      const { sessionId, expiresAt } = await createUserSession(
        userRecord.id, 
        rememberMe ? 90 : 30, // 90 days if remember me, otherwise 30
        clientIP,
        userAgent
      );
      
      // Set session cookie
      setSessionCookie(cookies, sessionId, expiresAt);

      console.log('✅ Session created successfully for:', userRecord.email);

    } catch (error) {
      console.error('Login error with PostgreSQL auth:', error);

      // Handle specific error messages
      const errorMessage = (error as Error).message;
      
      if (errorMessage.includes('Invalid email or password') || errorMessage.includes('Account is deactivated')) {
        return fail(400, {
          form: {
            ...form,
            errors: { email: [errorMessage] }
          }
        });
      }

      return fail(500, {
        form: {
          ...form,
          errors: { email: ['An error occurred during login. Please try again.'] }
        }
      });
    }

    // Redirect to dashboard or intended page
    throw redirect(302, '/dashboard');
  }
};