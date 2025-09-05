// @ts-nocheck
import type { RequestEvent } from "@sveltejs/kit";
import { signJWT, verifyJWT } from "./authUtils";
import { getUserById } from "./db/queries";

// In-memory session store (for development)
const sessions = new Map<string, Session>();

export interface Session {
  id: string;
  userId: string;
  expiresAt: Date;
}
export interface User {
  id: string;
  email: string;
  name: string;
  role?: string;
}
export async function validateSessionToken(
  token: string,
): Promise<{ session: Session | null; user: User | null }> {
  try {
    // Try JWT token validation
    const payload = verifyJWT(token);
    if (payload && payload.userId) {
      const user = await getUserById(payload.userId);
      if (user) {
        const session: Session = {
          id: token,
          userId: user.id,
          expiresAt: new Date(payload.exp * 1000),
        };
        return {
          session,
          user: {
            ...user,
            name: user.name || user.email || "Unknown User",
          } as User,
        };
      }
    }
    return { session: null, user: null };
  } catch (error) {
    console.error("Session validation error:", error);
    return { session: null, user: null };
  }
}
export function invalidateSession(sessionId: string): void {
  sessions.delete(sessionId);
}
export function setSessionTokenCookie(
  event: RequestEvent,
  token: string,
  expiresAt: Date,
): void {
  event.cookies.set("session", token, {
    path: "/",
    expires: expiresAt,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
  });
}
export function deleteSessionTokenCookie(event: RequestEvent): void {
  event.cookies.delete("session", {
    path: "/",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
  });
}
export function generateSessionToken(userId: string): string {
  return signJWT({ userId });
}
