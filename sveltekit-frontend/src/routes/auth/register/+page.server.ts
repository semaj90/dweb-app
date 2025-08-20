import { fail, redirect } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';
import { lucia } from '$lib/server/auth';
import { db } from '$lib/server/db';
import { authUsers, authAuditLog } from '$lib/server/db/auth-schema';
import { eq } from 'drizzle-orm';
import { hash } from '@node-rs/argon2';
import type { PageServerLoad, Actions } from './$types';

const registerSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  firstName: z.string().min(2, 'First name must be at least 2 characters'),
  lastName: z.string().min(2, 'Last name must be at least 2 characters'),
  password: z.string()
    .min(12, 'Password must be at least 12 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 
      'Password must include uppercase, lowercase, number, and special character'),
  confirmPassword: z.string(),
  role: z.enum(['prosecutor', 'investigator', 'analyst', 'admin']),
  department: z.string().min(2, 'Department is required'),
  jurisdiction: z.string().min(2, 'Jurisdiction is required'),
  badgeNumber: z.string().optional(),
  agreeToTerms: z.boolean().refine(val => val === true, 'You must agree to the terms'),
  agreeToPrivacy: z.boolean().refine(val => val === true, 'You must agree to privacy policy'),
  enableTwoFactor: z.boolean().default(false)
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export const load: PageServerLoad = async () => {
  const form = await superValidate(zod(registerSchema));
  return { form };
};

export const actions: Actions = {
  register: async ({ request, cookies, getClientAddress }) => {
    const form = await superValidate(request, zod(registerSchema));
    
    if (!form.valid) {
      return fail(400, { form });
    }

    const { 
      email, firstName, lastName, password, role, department, 
      jurisdiction, badgeNumber, enableTwoFactor 
    } = form.data;
    
    const clientIP = getClientAddress();
    const userAgent = request.headers.get('user-agent') || '';

    try {
      // Check if user already exists
      const existingUser = await db
        .select()
        .from(authUsers)
        .where(eq(authUsers.email, email.toLowerCase()))
        .limit(1);

      if (existingUser.length > 0) {
        await logAuthEvent({
          action: 'register',
          success: false,
          details: { email, error: 'Email already exists' },
          ipAddress: clientIP,
          userAgent
        });

        return fail(400, {
          form: {
            ...form,
            errors: { email: ['An account with this email already exists'] }
          }
        });
      }

      // Validate legal professional credentials (basic validation)
      const emailDomain = email.split('@')[1];
      const isGovDomain = emailDomain.endsWith('.gov') || 
                         emailDomain.includes('police') || 
                         emailDomain.includes('prosecutor') ||
                         emailDomain.includes('court') ||
                         emailDomain.includes('legal');

      // Enhanced validation for legal professionals
      if (role !== 'analyst' && !isGovDomain) {
        await logAuthEvent({
          action: 'register',
          success: false,
          details: { 
            email, 
            role, 
            error: 'Email domain validation failed' 
          },
          ipAddress: clientIP,
          userAgent
        });

        return fail(400, {
          form: {
            ...form,
            errors: { 
              email: ['Legal professionals must use an official government or legal institution email address'] 
            }
          }
        });
      }

      // Hash password
      const passwordHash = await hash(password, {
        memoryCost: 19456,
        timeCost: 2,
        outputLen: 32,
        parallelism: 1
      });

      // Create user
      const newUser = await db
        .insert(authUsers)
        .values({
          email: email.toLowerCase(),
          firstName,
          lastName,
          passwordHash,
          role,
          department,
          jurisdiction,
          badgeNumber,
          twoFactorEnabled: enableTwoFactor,
          isActive: true,
          isVerified: false, // Require email verification
          permissions: getDefaultPermissions(role),
          createdAt: new Date(),
          updatedAt: new Date()
        })
        .returning({ id: authUsers.id });

      const userId = newUser[0].id;

      // Log successful registration
      await logAuthEvent({
        userId,
        action: 'register',
        success: true,
        details: { 
          email, 
          role, 
          department, 
          jurisdiction,
          twoFactorEnabled: enableTwoFactor 
        },
        ipAddress: clientIP,
        userAgent
      });

      // For demo purposes, we'll create a session immediately
      // In production, you'd typically require email verification first
      const session = await lucia.createSession(userId, {
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

    } catch (error) {
      console.error('Registration error:', error);
      
      await logAuthEvent({
        action: 'register',
        success: false,
        details: { 
          email, 
          role,
          error: error instanceof Error ? error.message : 'Unknown error' 
        },
        ipAddress: clientIP,
        userAgent
      });

      return fail(500, {
        form: {
          ...form,
          errors: { email: ['An error occurred during registration. Please try again.'] }
        }
      });
    }

    // Redirect to dashboard
    throw redirect(302, '/dashboard');
  }
};

// Helper function to get default permissions based on role
function getDefaultPermissions(role: string): string[] {
  const permissions: Record<string, string[]> = {
    prosecutor: [
      'cases:read',
      'cases:write',
      'cases:manage',
      'evidence:read',
      'evidence:write',
      'documents:read',
      'documents:write',
      'ai:legal_analysis',
      'reports:generate'
    ],
    investigator: [
      'cases:read',
      'cases:write',
      'evidence:read',
      'evidence:write',
      'evidence:collect',
      'documents:read',
      'documents:write',
      'ai:evidence_analysis',
      'reports:generate'
    ],
    analyst: [
      'cases:read',
      'documents:read',
      'evidence:read',
      'ai:legal_analysis',
      'ai:document_analysis',
      'reports:read',
      'reports:generate'
    ],
    admin: [
      'system:admin',
      'users:manage',
      'cases:read',
      'cases:write',
      'cases:manage',
      'evidence:read',
      'evidence:write',
      'documents:read',
      'documents:write',
      'ai:all',
      'reports:all',
      'audit:read'
    ]
  };

  return permissions[role] || permissions.analyst;
}

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