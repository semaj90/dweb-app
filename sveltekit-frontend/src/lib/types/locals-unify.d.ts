// Canonical App.Locals augmentation to stabilize route `locals` usage.
// This is intentionally permissive as a short-term remediation. Replace
// with precise types for `user`, `session`, `db`, etc. in a follow-up pass.

declare namespace App {
  interface Locals {
    // Common session/user properties used across routes
    user?: {
      id?: string;
      email?: string;
      [key: string]: any;
    } | null;

    session?: {
      id?: string;
      user?: {
        id?: string;
        email?: string;
        [key: string]: any;
      } | null;
      [key: string]: any;
    } | null;

    // Common runtime helpers attached to locals in hooks
    db?: any;
    audit?: any;
    // Generic catch-all to avoid brittle failures while remediating types
    [key: string]: any;
  }
}



// Also augment the SvelteKit module so `Locals` used in RequestEvent and routes
// matches the canonical App.Locals above.
declare module '@sveltejs/kit' {
  // eslint-disable-next-line @typescript-eslint/no-empty-interface
  interface Locals extends App.Locals {}
}
// locals-unify.d.ts
// Ensure a single canonical mapping for Locals across module and global scopes.
// This helps files that reference `Locals` (global) or the SvelteKit module `Locals` to share the same shape.
import type { RequestEvent } from '@sveltejs/kit';

declare global {
  // Make the global `Locals` extend App.Locals so plain `Locals` references include user/session/db
  interface Locals extends App.Locals {}
}

declare module '@sveltejs/kit' {
  // Ensure module augmentation also extends App.Locals
  interface Locals extends App.Locals {}

  // Ensure RequestEvent.locals is recognized as App.Locals everywhere
  interface RequestEvent {
    locals: App.Locals;
  }
}


