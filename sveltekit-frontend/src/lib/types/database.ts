
import type { User } from "./user";
// Orphaned content: import type { InferInsertModel, InferSelectModel
import {
cases,
  criminals,
  evidence,
  users,
} from "$lib/server/db/schema-postgres";

// Database model types (inferred from schema)
export type DatabaseCase = InferSelectModel<typeof cases>;
export type NewCase = InferInsertModel<typeof cases>;

export type Criminal = InferSelectModel<typeof criminals>;
export type NewCriminal = InferInsertModel<typeof criminals>;

export type DatabaseEvidence = InferSelectModel<typeof evidence>;
export type NewEvidence = InferInsertModel<typeof evidence>;

export type DatabaseUser = InferSelectModel<typeof users>;
export type NewUser = InferInsertModel<typeof users>;

// Define Profile and Session types manually since they may not be in the schema
export interface Profile {
  id: string;
  userId: string;
  firstName: string;
  lastName: string;
  title?: string;
  department?: string;
  phone?: string;
  officeAddress?: string;
  avatar?: string;
  bio?: string;
  specializations: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface NewProfile
  extends Omit<Profile, "id" | "createdAt" | "updatedAt"> {
  id?: string;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface Session {
  id: string;
  userId: string;
  expiresAt: Date;
  createdAt: Date;
}

export interface NewSession extends Omit<Session, "createdAt"> {
  createdAt?: Date;
}

// Extended user types for better type safety
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

// Type for the user object returned by Auth.js session
export interface SessionUser {
  id: string;
  name?: string | null;
  email?: string | null;
  image?: string | null;
  role?: string | null;
  profile?: UserProfile;
}

// Complete user session interface
export interface UserSession {
  user: SessionUser | null;
  expires: Date | null;
}

// Role-based type safety
export type UserRole =
  | "prosecutor"
  | "detective"
  | "admin"
  | "analyst"
  | "viewer";

// Case status types for better type safety
export type CaseStatus =
  | "open"
  | "closed"
  | "pending"
  | "archived"
  | "under_review";

// Evidence types for better categorization
export type EvidenceType =
  | "document"
  | "image"
  | "video"
  | "audio"
  | "physical"
  | "digital"
  | "testimony";

// Priority levels
export type Priority = "low" | "medium" | "high" | "urgent";

// Case with related data
export interface CaseWithRelations extends DatabaseCase {
  criminal?: Criminal;
  evidence?: DatabaseEvidence[];
  assignedTo?: User;
  documents?: unknown[];
}

// User with profile
export interface UserWithProfile {
  id: string;
  email: string;
  name: string;
  profile?: Profile;
}

// Evidence with metadata (using intersection to avoid conflicts)
export interface EvidenceWithMetadata
  extends Omit<DatabaseEvidence, "uploadedBy"> {
  uploadedBy?: UserWithProfile; // Replace string ID with full User object
  uploadedById?: string; // Keep the original ID for reference
  case?: DatabaseCase;
  additionalTags?: string[]; // Additional tags beyond the required base tags
}
