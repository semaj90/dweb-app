import { and, eq, gte } from 'drizzle-orm';
// Modern server-managed session authentication utilities (no external Lucia runtime)
import { db } from '$lib/server/db';
// sessions/users live in schema-postgres which are re-exported; import explicitly to avoid type issues
import { sessions as sessionsTable, users as usersTable } from '$lib/server/db/unified-schema';
import bcrypt from "bcryptjs";
// Dynamic import for server-side crypto to prevent browser leakage
// import { randomBytes } from "crypto";

// --- Helper Functions ---
async function generateId(length: number = 40): Promise<string> {
  const { randomBytes } = await import("crypto");
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
  password: string
): Promise<boolean> {
  return await bcrypt.compare(password, hashedPassword);
}

// --- Session Management ---
export async function createUserSession(
  userId: string,
  days = 30,
  ipAddress?: string,
  userAgent?: string
): Promise<{ sessionId: string; expiresAt: Date }> {
  const sessionId = await generateId(40);
  const expiresAt = createDate({ days });
  await db.insert(sessionsTable).values({
    id: sessionId,
    user_id: userId,
    expires_at: expiresAt,
    ip_address: ipAddress,
    user_agent: userAgent,
    session_context: {},
  });
  return { sessionId, expiresAt };
}

export async function validateSession(sessionId: string): Promise<any> {
  const now = new Date();
  const session = await (db as any).query.sessions.findFirst({
    where: and(eq((sessionsTable as any).id, sessionId), gte((sessionsTable as any).expires_at, now)),
    with: {
      user: {
        columns: {
          id: true,
          email: true,
          first_name: true,
          last_name: true,
          role: true,
        },
      },
    },
  });
  if (session && session.user) {
    const { user, ...rest } = session;
    return {
      session: rest,
      user: {
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
        role: user.role,
      },
    };
  }
  return { session: null, user: null };
}

export async function invalidateSession(sessionId: string): Promise<void> {
  await db.delete(sessionsTable).where(eq((sessionsTable as any).id, sessionId));
}

export async function invalidateUserSessions(userId: string): Promise<void> {
  await db.delete(sessionsTable).where(eq((sessionsTable as any).user_id, userId));
}

// --- Cookie Helper ---
export function setSessionCookie(
  cookies: any,
  sessionId: string,
  expiresAt: Date
) {
  const options = {
    path: "/",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    expires: expiresAt,
  } as const;

  // Primary cookie name (used by newer code)
  cookies.set("session_id", sessionId, options);

  // Also set legacy/compat cookie name so older code or examples that
  // read `session` continue to work during local development.
  // This avoids silent failures on localhost when different cookie names
  // are expected across the codebase.
  try {
    cookies.set("session", sessionId, options);
  } catch (e: any) {
    // ignore - some runtimes may restrict duplicate cookie writes
  }
}

export function clearSessionCookie(cookies: any) {
  cookies.delete("session_id", { path: "/" });
}
