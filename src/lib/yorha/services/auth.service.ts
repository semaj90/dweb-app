// Authentication Service for YoRHa Interface
import { db } from '../db';
import { units, sessions, userActivity, passwordResetTokens } from '../db/schema';
import { eq, and, gt, lt, sql } from 'drizzle-orm';
import bcrypt from 'bcryptjs';
import { nanoid } from 'nanoid';
import jwt from 'jsonwebtoken';
import { env } from '$env/dynamic/private';
import type { Unit, NewUnit, Session, NewSession, NewUserActivity } from '../db/schema';
import { QueueService } from './queue.service';
import { EnhancedVectorService } from './vector.service';

const JWT_SECRET = env.JWT_SECRET || 'yorha-secret-key-change-in-production';
const SESSION_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days
const RESET_TOKEN_DURATION = 60 * 60 * 1000; // 1 hour

export class AuthService {
  private queueService: QueueService;
  private vectorService: EnhancedVectorService;

  constructor() {
    this.queueService = new QueueService();
    this.vectorService = new EnhancedVectorService();
  }

  // Generate unique Unit ID (e.g., "2B-4827")
  private generateUnitId(): string {
    const prefixes = ['2B', '9S', 'A2', '6O', '21O', '4S', '11B', '7E'];
    const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    const suffix = Math.floor(Math.random() * 9999).toString().padStart(4, '0');
    return `${prefix}-${suffix}`;
  }

  // Register new unit
  async register(data: {
    email: string;
    password: string;
    name: string;
    unitType?: 'combat' | 'scanner' | 'support' | 'operator' | 'healer';
  }): Promise<{ unit: Unit; session: Session }> {
    try {
      // Check if email already exists
      const existingUnit = await db.query.units.findFirst({
        where: eq(units.email, data.email)
      });

      if (existingUnit) {
        throw new Error('Email already registered');
      }

      // Generate unique unit ID
      let unitId = this.generateUnitId();
      let attempts = 0;
      while (attempts < 10) {
        const existing = await db.query.units.findFirst({
          where: eq(units.unitId, unitId)
        });
        if (!existing) break;
        unitId = this.generateUnitId();
        attempts++;
      }

      // Hash password
      const passwordHash = await bcrypt.hash(data.password, 12);

      // Create unit
      const [newUnit] = await db.insert(units).values({
        unitId,
        email: data.email,
        passwordHash,
        name: data.name,
        unitType: data.unitType || 'combat',
        emailVerificationToken: nanoid(32)
      }).returning();

      // Generate embedding for the unit profile
      await this.vectorService.generateUserEmbedding(newUnit.id);

      // Create session
      const session = await this.createSession(newUnit.id);

      // Log activity
      await this.logActivity({
        userId: newUnit.id,
        activityType: 'login',
        description: 'Unit registration completed',
        sessionId: session.id
      });

      // Queue welcome email
      await this.queueService.publishMessage('email', {
        type: 'welcome',
        to: newUnit.email,
        unitName: newUnit.name,
        verificationToken: newUnit.emailVerificationToken
      });

      // Queue achievement check
      await this.queueService.publishMessage('achievements', {
        userId: newUnit.id,
        action: 'check',
        type: 'registration'
      });

      return { unit: newUnit, session };
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  }

  // Login unit
  async login(email: string, password: string): Promise<{ unit: Unit; session: Session }> {
    try {
      // Find unit by email
      const unit = await db.query.units.findFirst({
        where: and(
          eq(units.email, email),
          eq(units.deletedAt, null)
        )
      });

      if (!unit) {
        throw new Error('Invalid credentials');
      }

      // Verify password
      const isValid = await bcrypt.compare(password, unit.passwordHash);
      if (!isValid) {
        throw new Error('Invalid credentials');
      }

      // Update last login
      await db.update(units)
        .set({ lastLoginAt: new Date() })
        .where(eq(units.id, unit.id));

      // Create session
      const session = await this.createSession(unit.id);

      // Log activity
      await this.logActivity({
        userId: unit.id,
        activityType: 'login',
        description: 'Unit authenticated successfully',
        sessionId: session.id
      });

      // Queue activity processing
      await this.queueService.publishMessage('activity', {
        userId: unit.id,
        type: 'login',
        timestamp: new Date().toISOString()
      });

      return { unit, session };
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  // Create session
  private async createSession(userId: string, userAgent?: string, ipAddress?: string): Promise<Session> {
    const token = nanoid(64);
    const expiresAt = new Date(Date.now() + SESSION_DURATION);

    const [session] = await db.insert(sessions).values({
      userId,
      token,
      userAgent,
      ipAddress,
      expiresAt
    }).returning();

    return session;
  }

  // Validate session
  async validateSession(token: string): Promise<{ unit: Unit; session: Session } | null> {
    try {
      const session = await db.query.sessions.findFirst({
        where: and(
          eq(sessions.token, token),
          gt(sessions.expiresAt, new Date())
        ),
        with: {
          user: true
        }
      });

      if (!session || !session.user) {
        return null;
      }

      // Update last activity
      await db.update(sessions)
        .set({ lastActivityAt: new Date() })
        .where(eq(sessions.id, session.id));

      return { unit: session.user, session };
    } catch (error) {
      console.error('Session validation error:', error);
      return null;
    }
  }

  // Logout
  async logout(sessionId: string): Promise<void> {
    try {
      const session = await db.query.sessions.findFirst({
        where: eq(sessions.id, sessionId)
      });

      if (session) {
        // Log activity before deleting session
        await this.logActivity({
          userId: session.userId,
          activityType: 'logout',
          description: 'Unit logged out',
          sessionId: session.id
        });

        // Delete session
        await db.delete(sessions).where(eq(sessions.id, sessionId));
      }
    } catch (error) {
      console.error('Logout error:', error);
      throw error;
    }
  }

  // Request password reset
  async requestPasswordReset(email: string): Promise<void> {
    try {
      const unit = await db.query.units.findFirst({
        where: eq(units.email, email)
      });

      if (!unit) {
        // Don't reveal if email exists
        return;
      }

      const token = nanoid(32);
      const expiresAt = new Date(Date.now() + RESET_TOKEN_DURATION);

      await db.insert(passwordResetTokens).values({
        userId: unit.id,
        token,
        expiresAt
      });

      // Queue reset email
      await this.queueService.publishMessage('email', {
        type: 'password-reset',
        to: unit.email,
        unitName: unit.name,
        resetToken: token
      });
    } catch (error) {
      console.error('Password reset request error:', error);
      throw error;
    }
  }

  // Reset password
  async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      const resetToken = await db.query.passwordResetTokens.findFirst({
        where: and(
          eq(passwordResetTokens.token, token),
          gt(passwordResetTokens.expiresAt, new Date()),
          eq(passwordResetTokens.usedAt, null)
        )
      });

      if (!resetToken) {
        throw new Error('Invalid or expired reset token');
      }

      // Hash new password
      const passwordHash = await bcrypt.hash(newPassword, 12);

      // Update password
      await db.update(units)
        .set({ passwordHash })
        .where(eq(units.id, resetToken.userId));

      // Mark token as used
      await db.update(passwordResetTokens)
        .set({ usedAt: new Date() })
        .where(eq(passwordResetTokens.id, resetToken.id));

      // Log activity
      await this.logActivity({
        userId: resetToken.userId,
        activityType: 'profile_update',
        description: 'Password reset completed'
      });

      // Invalidate all sessions
      await db.delete(sessions).where(eq(sessions.userId, resetToken.userId));
    } catch (error) {
      console.error('Password reset error:', error);
      throw error;
    }
  }

  // Verify email
  async verifyEmail(token: string): Promise<void> {
    try {
      const unit = await db.query.units.findFirst({
        where: eq(units.emailVerificationToken, token)
      });

      if (!unit) {
        throw new Error('Invalid verification token');
      }

      await db.update(units)
        .set({ 
          emailVerified: true,
          emailVerificationToken: null
        })
        .where(eq(units.id, unit.id));

      // Log activity
      await this.logActivity({
        userId: unit.id,
        activityType: 'profile_update',
        description: 'Email verified successfully'
      });

      // Queue achievement check
      await this.queueService.publishMessage('achievements', {
        userId: unit.id,
        action: 'check',
        type: 'email_verified'
      });
    } catch (error) {
      console.error('Email verification error:', error);
      throw error;
    }
  }

  // Enable 2FA
  async enable2FA(userId: string): Promise<string> {
    try {
      const secret = nanoid(32);
      
      await db.update(units)
        .set({ 
          twoFactorEnabled: true,
          twoFactorSecret: secret
        })
        .where(eq(units.id, userId));

      // Log activity
      await this.logActivity({
        userId,
        activityType: 'profile_update',
        description: '2FA enabled'
      });

      return secret;
    } catch (error) {
      console.error('2FA enable error:', error);
      throw error;
    }
  }

  // Log user activity
  private async logActivity(data: NewUserActivity): Promise<void> {
    try {
      await db.insert(userActivity).values(data);
    } catch (error) {
      console.error('Activity logging error:', error);
      // Don't throw, just log the error
    }
  }

  // Clean up expired sessions
  async cleanupSessions(): Promise<void> {
    try {
      const deleted = await db.delete(sessions)
        .where(lt(sessions.expiresAt, new Date()));
      
      console.log(`Cleaned up ${deleted.length || 0} expired sessions`);
    } catch (error) {
      console.error('Session cleanup error:', error);
    }
  }

  // Generate JWT token (for API access)
  generateJWT(userId: string): string {
    return jwt.sign(
      { userId, type: 'api' },
      JWT_SECRET,
      { expiresIn: '7d' }
    );
  }

  // Verify JWT token
  verifyJWT(token: string): { userId: string } | null {
    try {
      const payload = jwt.verify(token, JWT_SECRET) as any;
      return { userId: payload.userId };
    } catch (error) {
      return null;
    }
  }
}