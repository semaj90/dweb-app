// Project-level ambient declarations expected by svelte-check and TypeScript.
// The repo uses custom Svelte runes like $state/$derived/$props. Declare them
// as globals here to reduce massive 'is not defined (svelte)' warnings during
// the incremental hardening pass. Replace with proper types later.

declare var $state: any;
declare var $derived: any;
declare var $props: any;

// Also provide shorthand generic callable signatures for TypeScript consumers.
declare function $state<T = any>(initial: T): T;
declare function $derived<T = any>(fn: () => T): T;
declare function $props<T = any>(): T;

export {};
