import { Lucia } from "lucia";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";
import { db } from "../db/index.js";
import { sessions, users } from "../db/schema-unified.js";
import { dev } from "$app/environment";
import type { DatabaseUserAttributes } from "../auth.js";

// Enhanced Lucia v3 configuration for legal AI platform
const adapter = new DrizzlePostgreSQLAdapter(db, sessions, users);

export const lucia = new Lucia(adapter, {
  sessionCookie: {
    attributes: {
      secure: !dev, // HTTPS in production
      sameSite: "strict"
    }
  },
  getUserAttributes: (attributes) => {
    return {
      id: attributes.id,
      email: attributes.email,
      firstName: attributes.firstName,
      lastName: attributes.lastName,
      displayName: attributes.displayName,
      role: attributes.role,
      bio: attributes.bio,
      avatarUrl: attributes.avatarUrl,
      timezone: attributes.timezone,
      locale: attributes.locale,
      isActive: attributes.isActive,
      isSuspended: attributes.isSuspended,
      emailVerified: attributes.emailVerified,
      lastLoginAt: attributes.lastLoginAt,
      loginAttempts: attributes.loginAttempts,
      lockedUntil: attributes.lockedUntil,
      legalSpecialties: attributes.legalSpecialties,
      preferences: attributes.preferences,
      createdAt: attributes.createdAt,
      updatedAt: attributes.updatedAt
    };
  }
});

declare module "lucia" {
  interface Register {
    Lucia: typeof lucia;
    DatabaseUserAttributes: DatabaseUserAttributes;
  }
}

// DatabaseUserAttributes interface is defined in auth.ts

export type Auth = typeof lucia;