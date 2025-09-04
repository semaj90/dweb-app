// SvelteKit module declarations for TypeScript

declare module '$app/environment' {
  export const browser: boolean;
  export const dev: boolean;
  export const building: boolean;
  export const version: string;
}

declare module '$app/stores' {
  import type { Readable } from 'svelte/store';
  export const page: Readable<any>;
  export const navigating: Readable<any>; 
  export const updated: Readable<boolean>;
}

declare module '$app/navigation' {
  export function goto(url: string | URL, options?: any): Promise<any>;
  export function invalidate(resource?: string | ((url: URL) => boolean)): Promise<any>;
  export function invalidateAll(): Promise<any>;
  export const preloadCode: any;
  export const preloadData: any;
}