// locals-fix.d.ts
// Ensure SvelteKit's RequestEvent.locals is typed as App.Locals (merged from App namespace)
import type { RequestEvent } from '@sveltejs/kit';

declare module '@sveltejs/kit' {
  // In many places the code expects event.locals to be the App.Locals shape.
  // This merge ensures type consistency between App.Locals and RequestEvent.locals
  interface RequestEvent {
    locals: App.Locals;
  }
  // Also expose Locals at module level for direct references to `Locals`
  interface Locals extends App.Locals {}
}


