// User types consolidated
export interface User {
  id: string;
  email: string;
  name: string;
  firstName: string;
  lastName: string;
  avatarUrl: string;
  role: "prosecutor" | "investigator" | "admin" | "user";
  isActive: boolean;
  emailVerified: Date | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface UserSession {
  id: string;
  userId: string;
  expiresAt: Date;
}

export interface UserProfile extends User {
  preferences?: Record<string, any>;
  settings?: Record<string, any>;
}
