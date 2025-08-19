/// <reference types="svelte" />
// Relax Svelte component and HTML attribute typings for monorepo-wide incremental fixes.
// This makes all imported .svelte components accept any props (safe shim for large legacy codebases)
// and allows arbitrary HTML attributes (UnoCSS attributify, custom data-* attributes, etc.).

declare module '*.svelte' {
  import { SvelteComponentTyped } from 'svelte';
  // any props, any events, any slots
  export default class Component extends SvelteComponentTyped<Record<string, any>, Record<string, any>, Record<string, any>> {}
}

declare namespace svelte.JSX {
  // Allow arbitrary attributes on HTML elements (UnoCSS attributify uses arbitrary attributes like "mb-8").
  interface HTMLAttributes<T> {
    [key: string]: any;
  }
}
