import { db, sessions } from "$lib/server/db/index";
import bcrypt from "bcryptjs";
import { randomBytes } from "crypto";
import { and, eq, gte } from "drizzle-orm";

// Modern server-managed session authentication utilities (no Lucia)

// --- Helper Functions ---
function generateId(length: number = 40): string {
  return randomBytes(Math.ceil(length / 2))
    .toString("hex")
    .slice(0, length);
}
function createDate(timeSpan: { days: number }): Date {
  const date = new Date();
  date.setDate(date.getDate() + timeSpan.days);
  return date;
}

// --- Password Hashing ---
export async function hashPassword(password: string): Promise<string> {
  // Use bcrypt for strong password hashing
  return await bcrypt.hash(password, 12);
}
export async function verifyPassword(
  hashedPassword: string,
  password: string,
): Promise<boolean> {
  return await bcrypt.compare(password, hashedPassword);
}

// --- Session Management ---
export async function createUserSession(
  userId: string,
  days = 30,
): Promise<{ sessionId: string; expiresAt: Date }> {
  try {
    const sessionId = generateId(40);
    const expiresAt = createDate({ days });
    
    await db.insert(sessions).values({
      id: sessionId,
      userId,
      expiresAt,
    });
    
    return { sessionId, expiresAt };
  } catch (error: any) {
    console.error('Session creation failed:', error);
    throw new Error(`Failed to create session: ${error.message}`);
  }
}

export async function validateSession(sessionId: string) {
  try {
    if (!sessionId) {
      return null;
    }

    const now = new Date();
    const sessionWithUser = await db.query.sessions.findFirst({
      where: and(eq(sessions.id, sessionId), gte(sessions.expiresAt, now)),
      with: {
        user: {
          columns: {
            id: true,
            email: true,
            name: true,
            role: true,
            isActive: true,
          },
        },
      },
    });

    if (!sessionWithUser?.user) {
      return null;
    }

    // Return user object with consistent typing
    return {
      id: sessionWithUser.user.id,
      email: sessionWithUser.user.email,
      name: sessionWithUser.user.name,
      role: sessionWithUser.user.role,
      isActive: sessionWithUser.user.isActive,
    };
  } catch (error: any) {
    console.error('Session validation failed:', error);
    return null;
  }
}

export async function invalidateSession(sessionId: string): Promise<void> {
  try {
    if (!sessionId) return;
    await db.delete(sessions).where(eq(sessions.id, sessionId));
  } catch (error: any) {
    console.error('Session invalidation failed:', error);
    throw new Error(`Failed to invalidate session: ${error.message}`);
  }
}

export async function invalidateUserSessions(userId: string): Promise<void> {
  try {
    if (!userId) return;
    await db.delete(sessions).where(eq(sessions.userId, userId));
  } catch (error: any) {
    console.error('User sessions invalidation failed:', error);
    throw new Error(`Failed to invalidate user sessions: ${error.message}`);
  }
}

// --- Cookie Helper ---
export function setSessionCookie(
  cookies: any,
  sessionId: string,
  expiresAt: Date,
) {
  try {
    cookies.set("session_id", sessionId, {
      path: "/",
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      expires: expiresAt,
    });
  } catch (error: any) {
    console.error('Cookie setting failed:', error);
    throw new Error(`Failed to set session cookie: ${error.message}`);
  }
}

export function clearSessionCookie(cookies: any) {
  try {
    cookies.delete("session_id", { path: "/" });
  } catch (error: any) {
    console.error('Cookie clearing failed:', error);
    // Don't throw here as it's cleanup
  }
}

// --- Enhanced Session Types ---
export interface SessionUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  isActive: boolean;
}

export interface SessionValidationResult {
  user: SessionUser | null;
  isValid: boolean;
}

export async function validateSessionWithResult(sessionId: string): Promise<SessionValidationResult> {
  const user = await validateSession(sessionId);
  return {
    user,
    isValid: user !== null,
  };
}
