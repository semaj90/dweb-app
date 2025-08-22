// Global runtime declarations for Svelte v5 runes used across the project.
// This file provides permissive types so the language server and svelte-check
// won't complain that runes like $state, $derived, and $props are undefined.

declare global {
  // callable form: const state = $state(initial)
  function $state<T = any>(initial: T): T;
  const $state: <T = any>(initial: T) => T;

  // derived state helper
  function $derived<T = any>(fn: () => T): T;
  const $derived: <T = any>(fn: () => T) => T;

  // props accessor
  function $props<T = any>(): T;
  const $props: <T = any>() => T;
}

export {};
