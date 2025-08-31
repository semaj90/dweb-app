// Minimal shims to reduce TypeScript noise during bulk codemods

declare module 'drizzle-orm' {
  // common utility stubs used across the codebase
  export function eq(...args: any[]): any;
  export function and(...args: any[]): any;
  export function or(...args: any[]): any;
  export const placeholder: any;
  export type Any = any;
}

declare module '$lib/server/db/*' {
  const db: any;
  export default db;
}

declare module '$lib/mcp-context72-get-library-docs' {
  export function mcpContext72GetLibraryDocs(...args: any[]): any;
}

declare module 'glob' {
  export function glob(pattern: string, opts?: any, cb?: any): void;
  export function sync(pattern: string, opts?: any): string[];
  const globDefault: any;
  export default globDefault;
}

// Generic any fallback for unknown modules used during bulk fixes
declare module '*/*' {
  const whatever: any;
  export default whatever;
}
