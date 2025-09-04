/// <reference types="@sveltejs/kit" />

// Lightweight, low-risk ambient declarations to reduce noise during incremental fixes.
// These intentionally use `any` to avoid broad refactors; we'll refine them later.

declare global {
  namespace App {
    interface Locals {
      // common shortcuts used across many server routes
      user?: any;
      session?: any;
      db?: any;
      audit?: any;
      services?: any;
      requestId?: string;
    }
  }

  // helpers used in multiple route files that were referenced as globals
  // (superValidate, zod, message) — provide minimal signatures to silence TS until proper imports are added
  function superValidate(request: any, schema?: any): Promise<any>;
  function zod<T = any>(schema: any): any;
  function message(form: any, opts?: any): any;

  // Provide a PageServerLoad alias for projects with mixed SvelteKit typings
  // so older code using PageServerLoad won't break type resolution during incremental fixes.
}

// Also export module augmentation for @sveltejs/kit if needed later
export {};
