import { lucia } from "$lib/server/auth";
import type { Handle } from "@sveltejs/kit";

export const handle: Handle = async ({ event, resolve }) => {
  const sessionId = event.cookies.get(lucia.sessionCookieName);
  if (!sessionId) {
    event.locals.user = null;
    event.locals.session = null;
    return resolve(event);
  }

  const { session, user } = await lucia.validateSession(sessionId);
  if (session && session.fresh) {
    const sessionCookie = lucia.createSessionCookie(session.id);
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });
  }
  if (!session) {
    const sessionCookie = lucia.createBlankSessionCookie();
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });
  }

  // Transform lucia user to app User type
  if (user) {
    event.locals.user = {
      id: user.id,
      email: user.email,
      name: user.name || user.email,
      firstName: user.firstName,
      lastName: user.lastName,
      avatarUrl: user.avatarUrl,
      role: (user.role as "prosecutor" | "investigator" | "admin" | "user") || "user",
      isActive: Boolean(user.isActive ?? true),
      emailVerified: Boolean(user.emailVerified),
      createdAt: (user as any).createdAt ? new Date((user as any).createdAt) : new Date(),
      updatedAt: (user as any).updatedAt ? new Date((user as any).updatedAt) : new Date(),
    };
  } else {
    event.locals.user = null;
  }

  event.locals.session = session;
  return resolve(event);
};
