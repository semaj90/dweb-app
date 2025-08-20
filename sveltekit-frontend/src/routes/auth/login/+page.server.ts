import { fail, redirect } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';
import { lucia } from '$lib/server/auth';
import { db } from '$lib/server/db';
import { authUsers, authAuditLog } from '$lib/server/db/auth-schema';
import { eq } from 'drizzle-orm';
import { verify } from '@node-rs/argon2';
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
      // Find user by email
      const existingUser = await db
        .select()
        .from(authUsers)
        .where(eq(authUsers.email, email.toLowerCase()))
        .limit(1);

      if (existingUser.length === 0) {
        // Log failed login attempt
        await logAuthEvent({
          action: 'login',
          success: false,
          details: { email, error: 'User not found' },
          ipAddress: clientIP,
          userAgent
        });

        return fail(400, {
          form: {
            ...form,
            errors: { email: ['Invalid email or password'] }
          }
        });
      }

      const user = existingUser[0];

      // Check if account is active
      if (!user.isActive) {
        await logAuthEvent({
          userId: user.id,
          action: 'login',
          success: false,
          details: { email, error: 'Account disabled' },
          ipAddress: clientIP,
          userAgent
        });

        return fail(400, {
          form: {
            ...form,
            errors: { email: ['Account is disabled. Please contact support.'] }
          }
        });
      }

      // Verify password
      const validPassword = await verify(user.passwordHash, password, {
        memoryCost: 19456,
        timeCost: 2,
        outputLen: 32,
        parallelism: 1
      });

      if (!validPassword) {
        await logAuthEvent({
          userId: user.id,
          action: 'login',
          success: false,
          details: { email, error: 'Invalid password' },
          ipAddress: clientIP,
          userAgent
        });

        return fail(400, {
          form: {
            ...form,
            errors: { password: ['Invalid email or password'] }
          }
        });
      }

      // Check for two-factor authentication
      if (user.twoFactorEnabled && !twoFactorCode) {
        return fail(400, {
          form: {
            ...form,
            message: 'Two-factor authentication required'
          },
          requiresTwoFactor: true
        });
      }

      // Validate two-factor code if provided
      if (user.twoFactorEnabled && twoFactorCode) {
        // TODO: Implement TOTP validation
        // For now, accept any 6-digit code for testing
        if (!/^\d{6}$/.test(twoFactorCode)) {
          return fail(400, {
            form: {
              ...form,
              errors: { twoFactorCode: ['Invalid two-factor code'] }
            }
          });
        }
      }

      // Create session
      const session = await lucia.createSession(user.id, {
        ipAddress: clientIP,
        userAgent,
        deviceInfo: {
          platform: request.headers.get('sec-ch-ua-platform') || 'unknown',
          mobile: request.headers.get('sec-ch-ua-mobile') === '?1'
        }
      });

      const sessionCookie = lucia.createSessionCookie(session.id);
      cookies.set(sessionCookie.name, sessionCookie.value, {
        path: '.',
        ...sessionCookie.attributes
      });

      // Update last login
      await db
        .update(authUsers)
        .set({
          lastLoginAt: new Date(),
          lastLoginIp: clientIP,
          updatedAt: new Date()
        })
        .where(eq(authUsers.id, user.id));

      // Log successful login
      await logAuthEvent({
        userId: user.id,
        action: 'login',
        success: true,
        details: { 
          email, 
          rememberMe,
          twoFactorUsed: user.twoFactorEnabled 
        },
        ipAddress: clientIP,
        userAgent
      });

    } catch (error) {
      console.error('Login error:', error);
      
      await logAuthEvent({
        action: 'login',
        success: false,
        details: { 
          email, 
          error: error instanceof Error ? error.message : 'Unknown error' 
        },
        ipAddress: clientIP,
        userAgent
      });

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

// Helper function to log authentication events
async function logAuthEvent(event: {
  userId?: string;
  action: string;
  success: boolean;
  details: any;
  ipAddress?: string;
  userAgent?: string;
}): Promise<void> {
  try {
    await db.insert(authAuditLog).values({
      userId: event.userId,
      action: event.action,
      success: event.success,
      details: event.details,
      ipAddress: event.ipAddress,
      userAgent: event.userAgent,
      createdAt: new Date()
    });
  } catch (error) {
    console.error('Failed to log auth event:', error);
  }
}