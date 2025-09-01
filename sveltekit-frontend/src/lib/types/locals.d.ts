// Minimal App.Locals declarations to satisfy common server-side usages
// Add more fields as needed. Keep types permissive to avoid cascading errors.

declare namespace App {
  interface User {
    id: string;
    email?: string;
    name?: string;
    roles?: string[];
    [key: string]: any;
  }

  interface Session {
    id: string;
    user?: User;
    expires?: string | Date;
    [key: string]: any;
  }

  interface Locals {
    // Auth / identity
    user?: User | null;
    session?: Session | null;

    // Database connection / ORM (keep as any to avoid tight coupling)
    db?: any;

    // Audit / telemetry helper
    audit?: {
      log: (entry: { timestamp?: string | Date; action?: string; details?: any }) => Promise<void>;
    } | null;

    // Generic bag for other injected values
    [key: string]: any;
  }
}


