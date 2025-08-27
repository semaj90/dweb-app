import { Lucia } from "lucia";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";
import { db } from "../db/index.js";
import { sessions, users } from "../db/schema-postgres.js";
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
      username: attributes.username,
      firstName: attributes.first_name,
      lastName: attributes.last_name,
      displayName: `${attributes.first_name || ''} ${attributes.last_name || ''}`.trim() || attributes.username || attributes.email,
      role: attributes.role,
      department: attributes.department,
      jurisdiction: attributes.jurisdiction,
      avatarUrl: attributes.avatar_url,
      isActive: attributes.is_active,
      emailVerified: attributes.email_verified,
      lastLoginAt: attributes.last_login_at,
      practiceAreas: attributes.practice_areas,
      barNumber: attributes.bar_number,
      firmName: attributes.firm_name,
      metadata: attributes.metadata,
      createdAt: attributes.created_at,
      updatedAt: attributes.updated_at
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