import { sessions, users } from "./db/schema-postgres";
import { dev } from "$app/environment";
// Orphaned content: import {

import { Lucia } from "lucia";
// Orphaned content: import {

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
