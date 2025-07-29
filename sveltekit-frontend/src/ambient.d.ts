declare global {
  namespace App {
    interface Error {}
    interface Locals {}
    interface PageData {}
    interface Platform {}
  }
}

// Environment variable types for SvelteKit
declare module '$env/dynamic/private' {
  export const env: Record<string, string>;
}

declare module '$env/dynamic/public' {
  export const env: Record<string, string>;
}

declare module '$env/static/private' {
  export const env: Record<string, string>;
}

declare module '$env/static/public' {
  export const env: Record<string, string>;
}

// Tauri API types
declare module '@tauri-apps/api/tauri' {
  export function invoke(cmd: string, args?: any): Promise<any>;
}

export {};
