import { fail, redirect } from '@sveltejs/kit';
import { ExistingUserAuthService } from '$lib/server/db/existing-user-operations.js';
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
      // Use the existing auth service
      const result = await ExistingUserAuthService.registerUser({
        email: email.toLowerCase(),
        password: password,
        firstName: firstName,
        lastName: lastName,
        role: role || 'user',
        profileData: {
          department: department,
          jurisdiction: jurisdiction,
          badgeNumber: badgeNumber
        }
      });

      if (!result.success) {
        return fail(400, { error: result.error });
      }

      // Create a session using existing auth service
      const loginResult = await ExistingUserAuthService.loginUser({
        email: email.toLowerCase(),
        password: password
      });

      if (loginResult.success && loginResult.session) {
        // Set session cookie
        cookies.set('session_id', loginResult.session.id, {
          path: '/',
          httpOnly: true,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'strict',
          maxAge: 60 * 60 * 24 // 1 day
        });

        console.log('User registered and logged in successfully:', { 
          userId: result.user?.id, 
          email: email 
        });
      }

    } catch (error: any) {
      console.error('Registration error:', error);
      return fail(500, { error: 'An error occurred during registration. Please try again.' });
    }

    // Redirect to dashboard
    throw redirect(302, '/yorha/dashboard');
  }
};