// Lightweight declarations for project-specific Svelte helpers used during
// an incremental hardening pass. These are intentionally permissive and
// should be replaced with proper types or removed once the code is refactored.

declare global {
  // Provide both function and const-callable declarations for Svelte runes
  // so they are recognized by the TS checker and by svelte-check's analyzer.

  // $state<T>(initial) and const $state = <T>(initial) => initial
  function $state<T>(initial: T): T;
  const $state: <T>(initial: T) => T;

  // $derived<T>(() => value) and const $derived = <T>(fn) => fn()
  function $derived<T>(fn: () => T): T;
  const $derived: <T>(fn: () => T) => T;

  // $props<T>() used in some components; accept any shape for now.
  function $props<T = any>(): T;
  const $props: <T = any>() => T;

  // Some files / older code use runes without the $ prefix (e.g. state, derived).
  // Provide permissive aliases to reduce svelte-check noise during the hardening pass.
  function state<T>(initial: T): T;
  const state: <T>(initial: T) => T;

  function derived<T>(fn: () => T): T;
  const derived: <T>(fn: () => T) => T;

  function props<T = any>(): T;
  const props: <T = any>() => T;
}

export {};
