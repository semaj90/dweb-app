// fast-locals-shim.d.ts
// Minimal, non-invasive augmentation for common `event.locals` properties used across routes.
import type { RequestEvent } from '@sveltejs/kit';

declare module '@sveltejs/kit' {
  interface Locals {
    user?: { id: string; email?: string; name?: string } | null;
    session?: { id: string; user?: { id: string; email?: string } } | null;
    db?: any;
    services?: Partial<Record<string, any>>;
    requestId?: string;
    [key: string]: any;
  }
}


