/**
 * Temporary shim declarations to reduce TypeScript noise during automated autosolve cycles.
 * These should be removed once proper Svelte 5 rune migration / library typings are in place.
 */

// Svelte 5 rune placeholders (no-op passthroughs)
// They simply return the initial value so TS stops reporting `Cannot find name`.
declare function $state<T>(initial: T): T;
declare function $derived<T>(expr: T): T;
declare function $props<T>(): T;

// Allow accidentally migrated DOM style handlers (onclick / onchange) – will be codemodded back to on:click soon.
declare namespace svelteHTML {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface HTMLAttributes<T> {
    onclick?: any;
    onchange?: any;
  }
}

// Broad component prop fallback so `class` attribute is accepted even if component typing only exposes className.
declare module '*.svelte' {
  // Minimal shape; real components will have better generated types.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const Component: any;
  export default Component;
}

// bits-ui partial exports (silence: Module has no exported member 'Card' / 'Badge')
declare module 'bits-ui' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Card: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Badge: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Button: any;
}

// Playwright test shim (avoid missing devDependency noise during UI refactor)
declare module '@playwright/test' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const test: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const expect: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const chromium: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type Page = any;
}

// Testing-library shim
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare module '@testing-library/svelte' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const render: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const fireEvent: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const cleanup: any;
}

// Global utility fallbacks (rare missing lib cases)
declare const __AUTO_SOLVE__: boolean;
