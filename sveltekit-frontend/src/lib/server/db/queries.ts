
import { users } from "./schema-postgres";
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
    
    if (!result[0]) return null;
    
    const dbUser = result[0];
    return {
      id: dbUser.id,
      email: dbUser.email,
      displayName: dbUser.username || null,
      firstName: dbUser.first_name || null,
      lastName: dbUser.last_name || null,
      role: dbUser.role,
      bio: null,
      avatarUrl: dbUser.avatar_url || null,
      timezone: null,
      locale: null,
      isActive: dbUser.is_active,
      isSuspended: false,
      emailVerified: dbUser.email_verified ? new Date() : null,
      lastLoginAt: dbUser.last_login_at || null,
      loginAttempts: 0,
      lockedUntil: null,
      legalSpecialties: dbUser.practice_areas || null,
      preferences: dbUser.metadata || null,
      createdAt: dbUser.created_at,
      updatedAt: dbUser.updated_at
    } as User;
  } catch (error: any) {
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
    
    if (!result[0]) return null;
    
    const dbUser = result[0];
    return {
      id: dbUser.id,
      email: dbUser.email,
      displayName: dbUser.username || null,
      firstName: dbUser.first_name || null,
      lastName: dbUser.last_name || null,
      role: dbUser.role,
      bio: null,
      avatarUrl: dbUser.avatar_url || null,
      timezone: null,
      locale: null,
      isActive: dbUser.is_active,
      isSuspended: false,
      emailVerified: dbUser.email_verified ? new Date() : null,
      lastLoginAt: dbUser.last_login_at || null,
      loginAttempts: 0,
      lockedUntil: null,
      legalSpecialties: dbUser.practice_areas || null,
      preferences: dbUser.metadata || null,
      createdAt: dbUser.created_at,
      updatedAt: dbUser.updated_at
    } as User;
  } catch (error: any) {
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
        hashed_password: userData.passwordHash,
        username: userData.displayName,
        first_name: userData.firstName,
        last_name: userData.lastName,
        role: userData.role || "user",
        is_active: true
      })
      .returning();

    if (!result[0]) return null;
    
    const dbUser = result[0];
    return {
      id: dbUser.id,
      email: dbUser.email,
      displayName: dbUser.username || null,
      firstName: dbUser.first_name || null,
      lastName: dbUser.last_name || null,
      role: dbUser.role,
      bio: null,
      avatarUrl: dbUser.avatar_url || null,
      timezone: null,
      locale: null,
      isActive: dbUser.is_active,
      isSuspended: false,
      emailVerified: dbUser.email_verified ? new Date() : null,
      lastLoginAt: dbUser.last_login_at || null,
      loginAttempts: 0,
      lockedUntil: null,
      legalSpecialties: dbUser.practice_areas || null,
      preferences: dbUser.metadata || null,
      createdAt: dbUser.created_at,
      updatedAt: dbUser.updated_at
    } as User;
  } catch (error: any) {
    console.error("Error creating user:", error);
    return null;
  }
}
export async function updateUser(
  id: string,
  updates: Partial<User>,
): Promise<User | null> {
  try {
    // Convert camelCase updates to snake_case for database
    const dbUpdates: any = {
      updated_at: new Date()
    };
    
    if (updates.email) dbUpdates.email = updates.email;
    if (updates.displayName !== undefined) dbUpdates.username = updates.displayName;
    if (updates.firstName !== undefined) dbUpdates.first_name = updates.firstName;
    if (updates.lastName !== undefined) dbUpdates.last_name = updates.lastName;
    if (updates.role) dbUpdates.role = updates.role;
    if (updates.avatarUrl !== undefined) dbUpdates.avatar_url = updates.avatarUrl;
    if (updates.isActive !== undefined) dbUpdates.is_active = updates.isActive;
    if (updates.emailVerified !== undefined) dbUpdates.email_verified = !!updates.emailVerified;
    if (updates.lastLoginAt !== undefined) dbUpdates.last_login_at = updates.lastLoginAt;
    if (updates.legalSpecialties !== undefined) dbUpdates.practice_areas = updates.legalSpecialties;
    if (updates.preferences !== undefined) dbUpdates.metadata = updates.preferences;
    
    const result = await db
      .update(users)
      .set(dbUpdates)
      .where(eq(users.id, id))
      .returning();

    if (!result[0]) return null;
    
    const dbUser = result[0];
    return {
      id: dbUser.id,
      email: dbUser.email,
      displayName: dbUser.username || null,
      firstName: dbUser.first_name || null,
      lastName: dbUser.last_name || null,
      role: dbUser.role,
      bio: null,
      avatarUrl: dbUser.avatar_url || null,
      timezone: null,
      locale: null,
      isActive: dbUser.is_active,
      isSuspended: false,
      emailVerified: dbUser.email_verified ? new Date() : null,
      lastLoginAt: dbUser.last_login_at || null,
      loginAttempts: 0,
      lockedUntil: null,
      legalSpecialties: dbUser.practice_areas || null,
      preferences: dbUser.metadata || null,
      createdAt: dbUser.created_at,
      updatedAt: dbUser.updated_at
    } as User;
  } catch (error: any) {
    console.error("Error updating user:", error);
    return null;
  }
}
