import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals }) => {
  // If user is already logged in, redirect to dashboard
  if (locals.user) {
    throw redirect(302, '/cases');
  }

  return {};
};

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    const data = await request.formData();
    const email = data.get('email')?.toString();
    const password = data.get('password')?.toString();
    const rememberMe = data.get('rememberMe') === 'on';

    // Basic validation
    const fieldErrors: Record<string, string> = {};

    if (!email) {
      fieldErrors.email = 'Email is required';
    }

    if (!password) {
      fieldErrors.password = 'Password is required';
    }

    if (Object.keys(fieldErrors).length > 0) {
      return fail(400, {
        fieldErrors,
        message: 'Please fill in all required fields',
        type: 'error'
      });
    }

    try {
      // Demo authentication - in production, you'd verify against database
      const demoUsers = [
        { email: 'admin@legal.ai', password: 'admin123', role: 'admin', name: 'System Administrator' },
        { email: 'prosecutor@legal.ai', password: 'prosecutor123', role: 'prosecutor', name: 'John Prosecutor' },
        { email: 'detective@legal.ai', password: 'detective123', role: 'detective', name: 'Jane Detective' },
        { email: 'user@legal.ai', password: 'user123', role: 'user', name: 'Regular User' }
      ];

      const user = demoUsers.find(u => u.email === email && u.password === password);

      if (!user) {
        return fail(400, {
          fieldErrors: { email: 'Invalid email or password' },
          message: 'Authentication failed',
          type: 'error'
        });
      }

      // Simulate authentication delay
      await new Promise(resolve => setTimeout(resolve, 500));

      // Create session token (in production, use proper JWT or session management)
      const sessionToken = `demo_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const maxAge = rememberMe ? 60 * 60 * 24 * 30 : 60 * 60 * 24; // 30 days or 1 day

      // Set authentication cookie
      cookies.set('session', sessionToken, {
        path: '/',
        maxAge,
        httpOnly: true,
        secure: false, // Set to true in production with HTTPS
        sameSite: 'lax'
      });

      // Log successful authentication
      console.log('Demo authentication successful:', {
        email: user.email,
        role: user.role,
        name: user.name,
        rememberMe,
        sessionToken,
        timestamp: new Date().toISOString()
      });

      return {
        message: 'Login successful',
        type: 'success'
      };

    } catch (error) {
      console.error('Authentication error:', error);
      return fail(500, {
        message: 'An error occurred during login. Please try again.',
        type: 'error'
      });
    }
  }
};