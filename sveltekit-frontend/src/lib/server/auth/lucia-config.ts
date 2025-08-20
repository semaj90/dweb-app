import { lucia } from "lucia";
import { pg } from "@lucia-auth/adapter-postgresql";
import { db } from "../db/index.js";
import { dev } from "$app/environment";

// Enhanced Lucia configuration for legal AI platform
export const auth = lucia({
  env: dev ? "DEV" : "PROD",
  middleware: {
    user: {},
    session: {}
  },
  adapter: pg(db, {
    user: "auth_users",
    key: "auth_keys", 
    session: "auth_sessions"
  }),
  getUserAttributes: (data) => {
    return {
      userId: data.id,
      email: data.email,
      firstName: data.first_name,
      lastName: data.last_name,
      role: data.role,
      department: data.department,
      permissions: data.permissions,
      isActive: data.is_active,
      lastLoginAt: data.last_login_at,
      createdAt: data.created_at
    };
  }
});

export type Auth = typeof auth;