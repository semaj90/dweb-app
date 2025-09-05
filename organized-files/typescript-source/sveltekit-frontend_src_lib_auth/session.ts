// @ts-nocheck
// Lucia v3 Authentication with PostgreSQL and bcrypt
import { dev } from "$app/environment";
import { sessions, users } from "$lib/server/db/schema-postgres";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";
import { Lucia } from "lucia";
import { db } from "../server/db/pg";

// Create the Lucia adapter for PostgreSQL
// Note: db can be null during build time, handle this gracefully
if (!db) {
  throw new Error("Database connection is not available");
}
const adapter = new DrizzlePostgreSQLAdapter(db, sessions as any, users as any);

// Initialize Lucia with proper configuration
export const lucia = new Lucia(adapter, {
  sessionCookie: {
    attributes: {
      // Set to `true` when using HTTPS in production
      secure: !dev,
      // Recommended for SPA and server-side rendered apps
      sameSite: "lax",
      // Cookie path
      path: "/",
      // Note: Session expiration is handled by Lucia's session management
    },
  },
  // Define which user attributes to expose to the frontend
  getUserAttributes: (attributes) => {
    return {
      id: attributes.id,
      email: attributes.email,
      name: attributes.name,
      firstName: attributes.firstName,
      lastName: attributes.lastName,
      role: attributes.role,
      isActive: attributes.isActive,
      avatarUrl: attributes.avatarUrl,
      emailVerified: attributes.emailVerified,
    };
  },
});

// TypeScript declaration for user attributes
declare module "lucia" {
  interface Register {
    Lucia: typeof lucia;
    DatabaseUserAttributes: {
      id: string;
      email: string;
      name?: string;
      firstName?: string;
      lastName?: string;
      role: string;
      isActive: boolean;
      avatarUrl?: string;
      emailVerified?: Date | boolean; // Can be boolean or Date
      hashedPassword?: string;
      createdAt: Date;
      updatedAt: Date;
    };
    DatabaseSessionAttributes: {};
  }
}
export type Auth = typeof lucia;
