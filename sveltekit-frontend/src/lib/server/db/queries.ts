
import { users } from "$lib/server/db/schema-unified";
import { db, eq } from "./index";

export interface User {
  id: string;
  email: string;
  displayName: string | null;
  firstName: string | null;
  lastName: string | null;
  role: string;
  bio: string | null;
  avatarUrl: string | null;
  timezone: string | null;
  locale: string | null;
  isActive: boolean;
  isSuspended: boolean;
  emailVerified: Date | null;
  lastLoginAt: Date | null;
  loginAttempts: number;
  lockedUntil: Date | null;
  legalSpecialties: unknown;
  preferences: unknown;
  createdAt: Date;
  updatedAt: Date;
}
export async function getUserById(id: string): Promise<User | null> {
  try {
    const result = await db
      .select()
      .from(users)
      .where(eq(users.id, id))
      .limit(1);
    return result[0] || null;
  } catch (error) {
    console.error("Error fetching user by ID:", error);
    return null;
  }
}
export async function getUserByEmail(email: string): Promise<User | null> {
  try {
    const result = await db
      .select()
      .from(users)
      .where(eq(users.email, email))
      .limit(1);
    return result[0] || null;
  } catch (error) {
    console.error("Error fetching user by email:", error);
    return null;
  }
}
export async function createUser(userData: {
  email: string;
  passwordHash: string;
  displayName?: string;
  firstName?: string;
  lastName?: string;
  role?: string;
}): Promise<User | null> {
  try {
    const result = await db
      .insert(users)
      .values({
        email: userData.email,
        passwordHash: userData.passwordHash,
        displayName: userData.displayName,
        firstName: userData.firstName,
        lastName: userData.lastName,
        role: userData.role || "user",
        isActive: true
      })
      .returning();

    return result[0] || null;
  } catch (error) {
    console.error("Error creating user:", error);
    return null;
  }
}
export async function updateUser(
  id: string,
  updates: Partial<User>,
): Promise<User | null> {
  try {
    const result = await db
      .update(users)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(users.id, id))
      .returning();

    return result[0] || null;
  } catch (error) {
    console.error("Error updating user:", error);
    return null;
  }
}
