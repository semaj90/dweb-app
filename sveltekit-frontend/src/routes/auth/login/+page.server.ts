import { fail, redirect } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';
import { simpleAuthService } from '$lib/server/auth-simple';
import { lucia } from '$lib/server/auth';
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
      // Use simple authentication with PostgreSQL
      console.log('🔄 Using simple authentication with PostgreSQL...');
      
      // Login user using simpleAuthService
      const user = await simpleAuthService.login(email.toLowerCase(), password);
      
      console.log('✅ User authenticated successfully:', user.email);

      // Create session using simple auth service
      const session = await simpleAuthService.createSession(user.id);
      
      // Set session cookie
      const sessionCookie = lucia.createSessionCookie(session.id);
      cookies.set(sessionCookie.name, sessionCookie.value, {
        ...sessionCookie.attributes,
        path: '/'
      });

      console.log('✅ Session created successfully for:', user.email);

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