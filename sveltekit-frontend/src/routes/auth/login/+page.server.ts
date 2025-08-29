import { fail, redirect } from '@sveltejs/kit';
import { ExistingUserAuthService } from '$lib/server/db/existing-user-operations.js';
import type { PageServerLoad, Actions } from './$types';

export const load: PageServerLoad = async () => {
  return {};
};

export const actions: Actions = {
  login: async ({ request, cookies, getClientAddress }) => {
    const data = await request.formData();
    const email = data.get('email') as string;
    const password = data.get('password') as string;
    const rememberMe = data.get('rememberMe') === 'on';

    // Basic validation
    if (!email || !password) {
      return fail(400, { error: 'Email and password are required' });
    }

    const clientIP = getClientAddress();
    const userAgent = request.headers.get('user-agent') || '';

    try {
      console.log('🔄 Attempting login with existing auth service...');
      
      // Use the existing auth service that works with our database
      const result = await ExistingUserAuthService.loginUser({
        email: email.toLowerCase(),
        password: password,
        rememberMe: rememberMe,
        ipAddress: clientIP,
        userAgent: userAgent
      });
      
      if (!result.success) {
        return fail(400, { error: result.error || 'Invalid email or password' });
      }

      if (result.session) {
        // Set session cookie
        cookies.set('session_id', result.session.id, {
          path: '/',
          httpOnly: true,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'strict',
          maxAge: rememberMe ? 60 * 60 * 24 * 30 : 60 * 60 * 24 // 30 days or 1 day
        });

        console.log('✅ Login successful for:', result.user?.email);
      }

    } catch (error: any) {
      console.error('Login error:', error);
      return fail(500, { error: 'An error occurred during login. Please try again.' });
    }

    // Redirect to dashboard
    throw redirect(302, '/yorha/dashboard');
  }
};