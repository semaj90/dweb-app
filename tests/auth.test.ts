// Authentication Tests for YoRHa Interface
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { AuthService } from '../src/lib/yorha/services/auth.service';
import { db } from '../src/lib/yorha/db';
import { units, sessions, userActivity, passwordResetTokens } from '../src/lib/yorha/db/schema';
import { eq, sql } from 'drizzle-orm';

describe('AuthService', () => {
  let authService: AuthService;
  let testUserId: string;

  beforeAll(async () => {
    authService = new AuthService();
    // Initialize database for testing
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`);
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector`);
  });

  afterAll(async () => {
    // Clean up test data
    if (testUserId) {
      await db.delete(units).where(eq(units.id, testUserId));
    }
  });

  describe('User Registration', () => {
    it('should register a new user successfully', async () => {
      const registrationData = {
        email: 'test-unit@yorha.net',
        password: 'TestPassword123!',
        name: 'Test Unit 9S',
        unitType: 'scanner' as const
      };

      const result = await authService.register(registrationData);

      expect(result).toBeDefined();
      expect(result.unit).toBeDefined();
      expect(result.session).toBeDefined();
      expect(result.unit.email).toBe(registrationData.email);
      expect(result.unit.name).toBe(registrationData.name);
      expect(result.unit.unitType).toBe(registrationData.unitType);
      expect(result.unit.unitId).toMatch(/^[A-Z0-9]{2,3}-\d{4}$/);

      testUserId = result.unit.id;
    });

    it('should not register duplicate email', async () => {
      const registrationData = {
        email: 'test-unit@yorha.net',
        password: 'AnotherPassword123!',
        name: 'Duplicate Unit',
        unitType: 'combat' as const
      };

      await expect(authService.register(registrationData)).rejects.toThrow('Email already registered');
    });

    it('should generate unique unit IDs', async () => {
      const unitIds = new Set<string>();
      
      for (let i = 0; i < 10; i++) {
        const data = {
          email: `unit-${i}@yorha.net`,
          password: 'Password123!',
          name: `Unit ${i}`,
          unitType: 'combat' as const
        };

        const result = await authService.register(data);
        unitIds.add(result.unit.unitId);
        
        // Clean up immediately
        await db.delete(units).where(eq(units.id, result.unit.id));
      }

      expect(unitIds.size).toBe(10);
    });
  });

  describe('User Login', () => {
    it('should login with correct credentials', async () => {
      const result = await authService.login('test-unit@yorha.net', 'TestPassword123!');

      expect(result).toBeDefined();
      expect(result.unit).toBeDefined();
      expect(result.session).toBeDefined();
      expect(result.unit.email).toBe('test-unit@yorha.net');
    });

    it('should fail login with incorrect password', async () => {
      await expect(authService.login('test-unit@yorha.net', 'WrongPassword')).rejects.toThrow('Invalid credentials');
    });

    it('should fail login with non-existent email', async () => {
      await expect(authService.login('nonexistent@yorha.net', 'Password123!')).rejects.toThrow('Invalid credentials');
    });

    it('should update last login timestamp', async () => {
      const beforeLogin = new Date();
      await authService.login('test-unit@yorha.net', 'TestPassword123!');
      
      const user = await db.query.units.findFirst({
        where: eq(units.email, 'test-unit@yorha.net')
      });

      expect(user?.lastLoginAt).toBeDefined();
      expect(new Date(user!.lastLoginAt!).getTime()).toBeGreaterThanOrEqual(beforeLogin.getTime());
    });
  });

  describe('Session Management', () => {
    let sessionToken: string;

    beforeEach(async () => {
      const result = await authService.login('test-unit@yorha.net', 'TestPassword123!');
      sessionToken = result.session.token;
    });

    it('should validate valid session', async () => {
      const result = await authService.validateSession(sessionToken);

      expect(result).toBeDefined();
      expect(result?.unit.email).toBe('test-unit@yorha.net');
      expect(result?.session.token).toBe(sessionToken);
    });

    it('should return null for invalid session', async () => {
      const result = await authService.validateSession('invalid-token');
      expect(result).toBeNull();
    });

    it('should logout and invalidate session', async () => {
      const validationBefore = await authService.validateSession(sessionToken);
      expect(validationBefore).toBeDefined();

      await authService.logout(validationBefore!.session.id);

      const validationAfter = await authService.validateSession(sessionToken);
      expect(validationAfter).toBeNull();
    });

    it('should clean up expired sessions', async () => {
      // Create an expired session
      const expiredSession = await db.insert(sessions).values({
        userId: testUserId,
        token: 'expired-token',
        expiresAt: new Date(Date.now() - 1000) // Expired 1 second ago
      }).returning();

      await authService.cleanupSessions();

      const session = await db.query.sessions.findFirst({
        where: eq(sessions.token, 'expired-token')
      });

      expect(session).toBeUndefined();
    });
  });

  describe('Password Reset', () => {
    it('should create password reset token', async () => {
      await authService.requestPasswordReset('test-unit@yorha.net');

      const resetToken = await db.query.passwordResetTokens.findFirst({
        where: eq(passwordResetTokens.userId, testUserId)
      });

      expect(resetToken).toBeDefined();
      expect(resetToken?.expiresAt.getTime()).toBeGreaterThan(Date.now());
    });

    it('should reset password with valid token', async () => {
      await authService.requestPasswordReset('test-unit@yorha.net');

      const resetToken = await db.query.passwordResetTokens.findFirst({
        where: eq(passwordResetTokens.userId, testUserId)
      });

      await authService.resetPassword(resetToken!.token, 'NewPassword123!');

      // Should be able to login with new password
      const result = await authService.login('test-unit@yorha.net', 'NewPassword123!');
      expect(result).toBeDefined();
    });

    it('should fail with invalid reset token', async () => {
      await expect(authService.resetPassword('invalid-token', 'NewPassword123!')).rejects.toThrow('Invalid or expired reset token');
    });
  });

  describe('Email Verification', () => {
    it('should verify email with valid token', async () => {
      // Get the user's verification token
      const user = await db.query.units.findFirst({
        where: eq(units.id, testUserId)
      });

      if (user?.emailVerificationToken) {
        await authService.verifyEmail(user.emailVerificationToken);

        const updatedUser = await db.query.units.findFirst({
          where: eq(units.id, testUserId)
        });

        expect(updatedUser?.emailVerified).toBe(true);
        expect(updatedUser?.emailVerificationToken).toBeNull();
      }
    });

    it('should fail with invalid verification token', async () => {
      await expect(authService.verifyEmail('invalid-token')).rejects.toThrow('Invalid verification token');
    });
  });

  describe('2FA Management', () => {
    it('should enable 2FA', async () => {
      const secret = await authService.enable2FA(testUserId);

      expect(secret).toBeDefined();
      expect(secret.length).toBeGreaterThan(0);

      const user = await db.query.units.findFirst({
        where: eq(units.id, testUserId)
      });

      expect(user?.twoFactorEnabled).toBe(true);
      expect(user?.twoFactorSecret).toBe(secret);
    });
  });

  describe('JWT Token Management', () => {
    it('should generate valid JWT token', () => {
      const token = authService.generateJWT(testUserId);
      expect(token).toBeDefined();
      expect(token.split('.')).toHaveLength(3);
    });

    it('should verify valid JWT token', () => {
      const token = authService.generateJWT(testUserId);
      const result = authService.verifyJWT(token);

      expect(result).toBeDefined();
      expect(result?.userId).toBe(testUserId);
    });

    it('should return null for invalid JWT token', () => {
      const result = authService.verifyJWT('invalid.jwt.token');
      expect(result).toBeNull();
    });
  });

  describe('Activity Logging', () => {
    it('should log user activities', async () => {
      const result = await authService.login('test-unit@yorha.net', 'NewPassword123!');

      const activities = await db.query.userActivity.findMany({
        where: eq(userActivity.userId, testUserId),
        orderBy: (activity, { desc }) => [desc(activity.createdAt)]
      });

      expect(activities.length).toBeGreaterThan(0);
      
      const loginActivity = activities.find(a => a.activityType === 'login');
      expect(loginActivity).toBeDefined();
      expect(loginActivity?.description).toContain('authenticated successfully');
    });
  });
});