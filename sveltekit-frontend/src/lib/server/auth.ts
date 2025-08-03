import { dev } from "$app/environment";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";
import { Lucia } from "lucia";
import { db } from "./db/index";
import { sessions, users } from "./db/schema-postgres";

// Ensure database connection exists
if (!db) {
  throw new Error("Database connection is not available for Lucia authentication");
}

const adapter = new DrizzlePostgreSQLAdapter(db, sessions as any, users as any);

export const lucia = new Lucia(adapter, {
  sessionCookie: {
    attributes: {
      secure: !dev,
      sameSite: "lax",
      path: "/",
    },
  },
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
      emailVerified?: Date | boolean;
      hashedPassword?: string;
      createdAt: Date;
      updatedAt: Date;
    };
    DatabaseSessionAttributes: {};
  }
}
