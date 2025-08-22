import { lucia } from '$lib/auth/session';
import { db } from '$lib/server/db/pg';
import { users, sessions, userAuditLogs, type User, type NewUser, type NewUserAuditLog } from '$lib/database/schema';
import { eq, and, gte, lt, count } from 'drizzle-orm';
import bcrypt from 'bcrypt';
import crypto from 'crypto';
import type { RequestEvent } from '@sveltejs/kit';

export interface AuthResult {
  success: boolean;
  user?: User;
  session?: any;
  error?: string;
  requiresVerification?: boolean;
  lockoutUntil?: Date;
}

export interface LoginAttempt {
  email: string;
  password: string;
  ipAddress: string;
  userAgent: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  role?: string;
}

export interface SecuritySettings {
  maxLoginAttempts: number;
  lockoutDurationMinutes: number;
  sessionExpiryDays: number;
  requireEmailVerification: boolean;
  enforcePasswordComplexity: boolean;
  enable2FA: boolean;
}

export class EnhancedAuthService {
  private securitySettings: SecuritySettings = {
    maxLoginAttempts: 5,
    lockoutDurationMinutes: 15,
    sessionExpiryDays: 30,
    requireEmailVerification: true,
    enforcePasswordComplexity: true,
    enable2FA: false
  };

  constructor(settings?: Partial<SecuritySettings>) {
    if (settings) {
      this.securitySettings = { ...this.securitySettings, ...settings };
    }
  }

  /**
   * Enhanced login with security features
   */
  async login(loginData: LoginAttempt): Promise<AuthResult> {
    try {
      // Check for existing user
      const [existingUser] = await db
        .select()
        .from(users)
        .where(eq(users.email, loginData.email.toLowerCase()))
        .limit(1);

      if (!existingUser) {
        await this.logAuthEvent({
          userId: null,
          action: 'login_failed',
          ipAddress: loginData.ipAddress,
          userAgent: loginData.userAgent,
          metadata: { email: loginData.email, reason: 'user_not_found' }
        });
        
        return { success: false, error: 'Invalid email or password' };
      }

      // Check if account is locked
      if (existingUser.lockoutUntil && existingUser.lockoutUntil > new Date()) {
        await this.logAuthEvent({
          userId: existingUser.id,
          action: 'login_blocked',
          ipAddress: loginData.ipAddress,
          userAgent: loginData.userAgent,
          metadata: { reason: 'account_locked', lockoutUntil: existingUser.lockoutUntil }
        });

        return { 
          success: false, 
          error: 'Account is temporarily locked due to multiple failed login attempts',
          lockoutUntil: existingUser.lockoutUntil
        };
      }

      // Check if account is active
      if (!existingUser.isActive) {
        return { success: false, error: 'Account is deactivated' };
      }

      // Verify password
      const isValidPassword = await bcrypt.compare(loginData.password, existingUser.passwordHash);
      
      if (!isValidPassword) {
        await this.handleFailedLogin(existingUser, loginData);
        return { success: false, error: 'Invalid email or password' };
      }

      // Check email verification if required
      if (this.securitySettings.requireEmailVerification && !existingUser.emailVerified) {
        return { 
          success: false, 
          error: 'Please verify your email address before logging in',
          requiresVerification: true
        };
      }

      // Reset login attempts on successful login
      await this.resetLoginAttempts(existingUser.id);

      // Create session
      const session = await lucia.createSession(existingUser.id, {
        ipAddress: loginData.ipAddress,
        userAgent: loginData.userAgent
      });

      // Update last login
      await db
        .update(users)
        .set({ lastLoginAt: new Date() })
        .where(eq(users.id, existingUser.id));

      // Log successful login
      await this.logAuthEvent({
        userId: existingUser.id,
        action: 'login_success',
        ipAddress: loginData.ipAddress,
        userAgent: loginData.userAgent,
        metadata: { rememberMe: loginData.rememberMe }
      });

      return {
        success: true,
        user: existingUser,
        session
      };

    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'An unexpected error occurred' };
    }
  }

  /**
   * Enhanced registration with validation
   */
  async register(registerData: RegisterData, request: RequestEvent): Promise<AuthResult> {
    try {
      const { email, password, firstName, lastName, role = 'user' } = registerData;

      // Validate input
      const validation = this.validateRegistrationData(registerData);
      if (!validation.isValid) {
        return { success: false, error: validation.error };
      }

      // Check if user already exists
      const [existingUser] = await db
        .select()
        .from(users)
        .where(eq(users.email, email.toLowerCase()))
        .limit(1);

      if (existingUser) {
        return { success: false, error: 'An account with this email already exists' };
      }

      // Hash password
      const passwordHash = await bcrypt.hash(password, 12);

      // Generate email verification token if required
      const emailVerificationToken = this.securitySettings.requireEmailVerification 
        ? crypto.randomBytes(32).toString('hex')
        : null;

      // Create user
      const [newUser] = await db
        .insert(users)
        .values({
          email: email.toLowerCase(),
          passwordHash,
          firstName,
          lastName,
          role,
          emailVerificationToken,
          isActive: !this.securitySettings.requireEmailVerification
        })
        .returning();

      // Log registration
      await this.logAuthEvent({
        userId: newUser.id,
        action: 'register_success',
        ipAddress: this.getClientIP(request),
        userAgent: request.request.headers.get('user-agent') || '',
        metadata: { role, requiresVerification: this.securitySettings.requireEmailVerification }
      });

      // Send verification email if required
      if (this.securitySettings.requireEmailVerification && emailVerificationToken) {
        await this.sendVerificationEmail(newUser.email, emailVerificationToken);
        
        return {
          success: true,
          user: newUser,
          requiresVerification: true
        };
      }

      // Create session for immediate login
      const session = await lucia.createSession(newUser.id, {
        ipAddress: this.getClientIP(request),
        userAgent: request.request.headers.get('user-agent') || ''
      });

      return {
        success: true,
        user: newUser,
        session
      };

    } catch (error) {
      console.error('Registration error:', error);
      return { success: false, error: 'Registration failed. Please try again.' };
    }
  }

  /**
   * Logout with session cleanup
   */
  async logout(sessionId: string, request: RequestEvent): Promise<void> {
    try {
      const session = await lucia.validateSession(sessionId);
      
      if (session.session) {
        await lucia.invalidateSession(sessionId);
        
        await this.logAuthEvent({
          userId: session.user?.id || null,
          action: 'logout_success',
          ipAddress: this.getClientIP(request),
          userAgent: request.request.headers.get('user-agent') || '',
          metadata: { sessionId }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    }
  }

  /**
   * Verify email address
   */
  async verifyEmail(token: string): Promise<AuthResult> {
    try {
      const [user] = await db
        .select()
        .from(users)
        .where(eq(users.emailVerificationToken, token))
        .limit(1);

      if (!user) {
        return { success: false, error: 'Invalid or expired verification token' };
      }

      await db
        .update(users)
        .set({
          emailVerified: new Date(),
          emailVerificationToken: null,
          isActive: true
        })
        .where(eq(users.id, user.id));

      await this.logAuthEvent({
        userId: user.id,
        action: 'email_verified',
        ipAddress: 'system',
        userAgent: 'system',
        metadata: { token }
      });

      return { success: true, user };

    } catch (error) {
      console.error('Email verification error:', error);
      return { success: false, error: 'Verification failed' };
    }
  }

  /**
   * Request password reset
   */
  async requestPasswordReset(email: string, request: RequestEvent): Promise<{ success: boolean; error?: string }> {
    try {
      const [user] = await db
        .select()
        .from(users)
        .where(eq(users.email, email.toLowerCase()))
        .limit(1);

      if (!user) {
        // Don't reveal if email exists
        return { success: true };
      }

      const resetToken = crypto.randomBytes(32).toString('hex');
      const resetExpires = new Date(Date.now() + 60 * 60 * 1000); // 1 hour

      await db
        .update(users)
        .set({
          passwordResetToken: resetToken,
          passwordResetExpires: resetExpires
        })
        .where(eq(users.id, user.id));

      await this.sendPasswordResetEmail(user.email, resetToken);

      await this.logAuthEvent({
        userId: user.id,
        action: 'password_reset_requested',
        ipAddress: this.getClientIP(request),
        userAgent: request.request.headers.get('user-agent') || '',
        metadata: { resetExpires }
      });

      return { success: true };

    } catch (error) {
      console.error('Password reset request error:', error);
      return { success: false, error: 'Failed to process password reset request' };
    }
  }

  /**
   * Reset password with token
   */
  async resetPassword(token: string, newPassword: string): Promise<AuthResult> {
    try {
      const [user] = await db
        .select()
        .from(users)
        .where(
          and(
            eq(users.passwordResetToken, token),
            gte(users.passwordResetExpires, new Date())
          )
        )
        .limit(1);

      if (!user) {
        return { success: false, error: 'Invalid or expired reset token' };
      }

      // Validate new password
      if (!this.validatePassword(newPassword)) {
        return { success: false, error: 'Password does not meet security requirements' };
      }

      const passwordHash = await bcrypt.hash(newPassword, 12);

      await db
        .update(users)
        .set({
          passwordHash,
          passwordResetToken: null,
          passwordResetExpires: null,
          loginAttempts: 0,
          lockoutUntil: null
        })
        .where(eq(users.id, user.id));

      await this.logAuthEvent({
        userId: user.id,
        action: 'password_reset_success',
        ipAddress: 'system',
        userAgent: 'system',
        metadata: { token }
      });

      return { success: true, user };

    } catch (error) {
      console.error('Password reset error:', error);
      return { success: false, error: 'Password reset failed' };
    }
  }

  /**
   * Get user security summary
   */
  async getSecuritySummary(userId: string): Promise<any> {
    try {
      // Get recent audit logs
      const recentLogs = await db
        .select()
        .from(userAuditLogs)
        .where(eq(userAuditLogs.userId, userId))
        .orderBy(userAuditLogs.createdAt)
        .limit(10);

      // Get active sessions count
      const activeSessions = await db
        .select({ count: count() })
        .from(sessions)
        .where(
          and(
            eq(sessions.userId, userId),
            gte(sessions.expiresAt, new Date())
          )
        );

      return {
        recentActivity: recentLogs,
        activeSessionsCount: activeSessions[0]?.count || 0,
        securitySettings: this.securitySettings
      };

    } catch (error) {
      console.error('Security summary error:', error);
      return null;
    }
  }

  // Private helper methods

  private async handleFailedLogin(user: User, loginData: LoginAttempt): Promise<void> {
    const newAttempts = (user.loginAttempts || 0) + 1;
    const lockoutUntil = newAttempts >= this.securitySettings.maxLoginAttempts
      ? new Date(Date.now() + this.securitySettings.lockoutDurationMinutes * 60 * 1000)
      : null;

    await db
      .update(users)
      .set({
        loginAttempts: newAttempts,
        lockoutUntil
      })
      .where(eq(users.id, user.id));

    await this.logAuthEvent({
      userId: user.id,
      action: 'login_failed',
      ipAddress: loginData.ipAddress,
      userAgent: loginData.userAgent,
      metadata: { 
        attempts: newAttempts,
        locked: !!lockoutUntil,
        lockoutUntil
      }
    });
  }

  private async resetLoginAttempts(userId: string): Promise<void> {
    await db
      .update(users)
      .set({
        loginAttempts: 0,
        lockoutUntil: null
      })
      .where(eq(users.id, userId));
  }

  private validateRegistrationData(data: RegisterData): { isValid: boolean; error?: string } {
    const { email, password, firstName, lastName } = data;

    if (!email || !password || !firstName || !lastName) {
      return { isValid: false, error: 'All fields are required' };
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return { isValid: false, error: 'Invalid email format' };
    }

    if (firstName.trim().length < 2 || lastName.trim().length < 2) {
      return { isValid: false, error: 'First and last name must be at least 2 characters' };
    }

    if (!this.validatePassword(password)) {
      return { 
        isValid: false, 
        error: 'Password must be at least 8 characters with uppercase, lowercase, number, and special character' 
      };
    }

    return { isValid: true };
  }

  private validatePassword(password: string): boolean {
    if (!this.securitySettings.enforcePasswordComplexity) {
      return password.length >= 6;
    }

    return password.length >= 8 &&
      /[a-z]/.test(password) &&
      /[A-Z]/.test(password) &&
      /\d/.test(password) &&
      /[@$!%*?&]/.test(password);
  }

  private async logAuthEvent(log: Omit<NewUserAuditLog, 'id' | 'createdAt'>): Promise<void> {
    try {
      await db.insert(userAuditLogs).values(log);
    } catch (error) {
      console.error('Failed to log auth event:', error);
    }
  }

  private getClientIP(request: RequestEvent): string {
    return request.getClientAddress() || 
           request.request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ||
           request.request.headers.get('x-real-ip') ||
           'unknown';
  }

  private async sendVerificationEmail(email: string, token: string): Promise<void> {
    // TODO: Implement email sending service
    console.log(`Verification email would be sent to ${email} with token ${token}`);
  }

  private async sendPasswordResetEmail(email: string, token: string): Promise<void> {
    // TODO: Implement email sending service
    console.log(`Password reset email would be sent to ${email} with token ${token}`);
  }
}