
// Lucia v3 Authentication with PostgreSQL and bcrypt
import { dev } from "$app/environment";
import { sessions, users } from "../server/db/schema-postgres";
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
      firstName: attributes.firstName,
      lastName: attributes.lastName,
      role: attributes.role,
      isActive: attributes.isActive,
      emailVerified: attributes.emailVerified,
    };
  },
});

// TypeScript declaration for user attributes - using server auth definitions
export type Auth = typeof lucia;
