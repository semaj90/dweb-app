/**
 * User Profile Types
 * Based on the unified database schema
 */

export interface UserProfile {
  firstName?: string;
  lastName?: string;
  avatarUrl?: string;
  role: string;
  isActive: boolean;
  emailVerified?: Date | null;
  createdAt: Date;
  updatedAt: Date;
}
export interface User {
  id: string;
  email: string;
  name: string;
  firstName?: string;
  lastName?: string;
  avatarUrl?: string;
  role: string;
  isActive: boolean;
  emailVerified?: Date | null;
  createdAt: Date;
  updatedAt: Date;
  hashedPassword?: string; // Only for backend use
}
export interface UserSession {
  user: {
    id: string;
    email: string;
    name: string;
    image?: string;
    username?: string;
    role?: string;
    profile?: UserProfile;
  } | null;
  expires: Date | null;
  id: string;
  userId: string;
  fresh: boolean;
  expiresAt: Date;
}
export type UserRole =
  | "prosecutor"
  | "detective"
  | "admin"
  | "analyst"
  | "viewer";

export interface UserWithTheme extends User {
  activeTheme?: {
    id: string;
    name: string;
    cssVariables: Record<string, string>;
    colorPalette: Record<string, string>;
  };
}
