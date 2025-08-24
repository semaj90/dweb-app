import { fail, redirect } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';
import { relayAuthService } from '$lib/server/services/relay-auth-service';
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
      // Use relay auth service instead of direct database calls
      console.log('🔄 Using RelayAuthService for login authentication...');
      
      const user = await relayAuthService.getUserByEmail(email.toLowerCase());

      if (!user) {
        console.log('❌ User not found via relay service:', email);
        return fail(400, {
          form: {
            ...form,
            errors: { email: ['Invalid email or password'] }
          }
        });
      }

      // Check if account is active
      if (!user.is_active) {
        console.log('❌ Account disabled via relay service:', email);
        return fail(400, {
          form: {
            ...form,
            errors: { email: ['Account is disabled. Please contact support.'] }
          }
        });
      }

      // Verify password using relay auth service
      const validPassword = await relayAuthService.validatePassword(user, password);

      if (!validPassword) {
        console.log('❌ Invalid password via relay service:', email);
        return fail(400, {
          form: {
            ...form,
            errors: { password: ['Invalid email or password'] }
          }
        });
      }

      console.log('✅ Password verified via relay service:', email);

      // Skip two-factor for demo (relay service doesn't handle 2FA yet)
      // Future enhancement: add 2FA support to relay service

      // Create session using relay auth service
      const session = await relayAuthService.createSession(user.id, {
        ipAddress: clientIP,
        userAgent,
        deviceInfo: {
          platform: request.headers.get('sec-ch-ua-platform') || 'unknown',
          mobile: request.headers.get('sec-ch-ua-mobile') === '?1'
        }
      });

      // Create session cookie manually to avoid Lucia's database connection
      const cookieName = 'auth-session';
      const cookieValue = session.id;
      const isSecure = false; // dev mode
      
      cookies.set(cookieName, cookieValue, {
        path: '/',
        httpOnly: true,
        secure: isSecure,
        sameSite: 'strict',
        maxAge: 60 * 60 * 24 * 7 // 7 days
      });

      // Skip database updates for now - relay service handles this
      // Future enhancement: add login tracking to relay service
      
      console.log('✅ Session created via relay service for:', email);

    } catch (error) {
      console.error('Login error via relay service:', error);

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