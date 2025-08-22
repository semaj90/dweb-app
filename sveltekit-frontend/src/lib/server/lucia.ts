import { and, eq, gte } from "drizzle-orm";
// Modern server-managed session authentication utilities (no Lucia)
import { db } from "$lib/server/db/index";
import { sessions } from "$lib/server/db/schema-postgres";
import bcrypt from "bcryptjs";
import { randomBytes } from "crypto";

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
  password: string
): Promise<boolean> {
  return await bcrypt.compare(password, hashedPassword);
}

// --- Session Management ---
export async function createUserSession(
  userId: string,
  days = 30
): Promise<{ sessionId: string; expiresAt: Date }> {
  const sessionId = generateId(40);
  const expiresAt = createDate({ days });
  await db.insert(sessions).values({
    id: sessionId,
    userId,
    expiresAt,
  });
  return { sessionId, expiresAt };
}

export async function validateSession(sessionId: string) {
  // Find the session in the database that is not expired, and join user
  const now = new Date();
  const session = await db.query.sessions.findFirst({
    where: and(eq(sessions.id, sessionId), gte(sessions.expiresAt, now)),
    with: {
      user: {
        columns: {
          id: true,
          email: true,
          // Add more columns as needed
        },
      },
    },
  });
  return session && session.user ? session.user : null;
}

export async function invalidateSession(sessionId: string): Promise<void> {
  await db.delete(sessions).where(eq(sessions.id, sessionId));
}

export async function invalidateUserSessions(userId: string): Promise<void> {
  await db.delete(sessions).where(eq(sessions.userId, userId));
}

// --- Cookie Helper ---
export function setSessionCookie(
  cookies: any,
  sessionId: string,
  expiresAt: Date
) {
  cookies.set("session_id", sessionId, {
    path: "/",
    httpOnly: true,
    secure: import.meta.env.NODE_ENV === "production",
    sameSite: "lax",
    expires: expiresAt,
  });
}

export function clearSessionCookie(cookies: any) {
  cookies.delete("session_id", { path: "/" });
}
