import { sessions, users } from "./db/schema-unified";
import { dev } from "$app/environment";
import { db } from "./db/index";
import { Lucia } from "lucia";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";

const adapter = new DrizzlePostgreSQLAdapter(db, sessions, users);

export const lucia = new Lucia(adapter, {
  sessionCookie: {
    attributes: {
      secure: !dev,
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
