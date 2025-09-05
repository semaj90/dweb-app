// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
import type { Unit, Session } from '$lib/yorha/db/schema';

declare global {
  namespace App {
    interface Error {
      message: string;
      code?: string;
    }
    
    interface Locals {
      user?: Unit;
      session?: Session;
    }
    
    interface PageData {
      user?: Unit;
    }
    
    interface Platform {}
  }
}

// Mock $env modules for TypeScript checking
declare module '$env/dynamic/private' {
  export const env: Record<string, string>;
}

declare module '$env/static/private' {
  const env: Record<string, string>;
  export = env;
}

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
  export function goto(url: string | URL, options?: {
    replaceState?: boolean;
    noscroll?: boolean;
    keepfocus?: boolean;
    invalidateAll?: boolean;
    state?: any;
  }): Promise<any>;
  export function invalidate(resource?: string | ((url: URL) => boolean)): Promise<any>;
  export function invalidateAll(): Promise<any>;
  export function preloadCode(...paths: string[]): Promise<any>;
  export function preloadData(href: string): Promise<any>;
  export function beforeNavigate(fn: (navigation: any) => void): void;
  export function afterNavigate(fn: (navigation: any) => void): void;
  export function onNavigate(fn: (navigation: any) => void | Promise<any>): void;
  export function pushState(url: string | URL, state: any): void;
  export function replaceState(url: string | URL, state: any): void;
}

export {};