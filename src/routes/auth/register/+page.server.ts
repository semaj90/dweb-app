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
    const confirmPassword = data.get('confirmPassword')?.toString();
    const firstName = data.get('firstName')?.toString();
    const lastName = data.get('lastName')?.toString();
    const role = data.get('role')?.toString() || 'user';

    // Basic validation
    const fieldErrors: Record<string, string> = {};

    if (!email) {
      fieldErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      fieldErrors.email = 'Please enter a valid email address';
    }

    if (!password) {
      fieldErrors.password = 'Password is required';
    } else if (password.length < 8) {
      fieldErrors.password = 'Password must be at least 8 characters long';
    }

    if (!confirmPassword) {
      fieldErrors.confirmPassword = 'Please confirm your password';
    } else if (password !== confirmPassword) {
      fieldErrors.confirmPassword = 'Passwords do not match';
    }

    if (!firstName) {
      fieldErrors.firstName = 'First name is required';
    }

    if (!lastName) {
      fieldErrors.lastName = 'Last name is required';
    }

    if (Object.keys(fieldErrors).length > 0) {
      return fail(400, {
        fieldErrors,
        message: 'Please correct the errors below',
        type: 'error'
      });
    }

    try {
      // Here you would typically:
      // 1. Hash the password
      // 2. Save user to database
      // 3. Send verification email
      
      // For demo purposes, we'll simulate successful registration
      console.log('Demo registration:', {
        email,
        firstName,
        lastName,
        role,
        timestamp: new Date().toISOString()
      });

      // Simulate database operation delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // In a real app, you'd check if email already exists
      if (email === 'admin@example.com') {
        return fail(400, {
          fieldErrors: { email: 'An account with this email already exists' },
          message: 'Registration failed',
          type: 'error'
        });
      }

      return {
        message: 'Account created successfully! You can now sign in.',
        type: 'success'
      };

    } catch (error) {
      console.error('Registration error:', error);
      return fail(500, {
        message: 'An error occurred during registration. Please try again.',
        type: 'error'
      });
    }
  }
};