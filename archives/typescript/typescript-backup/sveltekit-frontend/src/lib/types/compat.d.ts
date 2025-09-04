// Compatibility shims for legacy/alias imports used across the codebase.
// These are deliberately permissive (any) to reduce type noise while we
// incrementally migrate to stricter, accurate typings.

declare module '$lib/server/db/drizzle' {
  // many modules import `db` as named or default — provide both
  export const db: any;
  export default db;
  export function connect(...args: any[]): any;
}

declare module '$lib/server/db/schema-postgres' {
  // Provide permissive aliases for different naming conventions observed
  // in the repo (snake_case vs camelCase). Keep as `any` to avoid cascade
  // failures while refactoring.
  export const legal_documents: any;
  export const legalDocuments: any;
  export const vectors: any;
  export const vectorJobs: any;
  export const evidence: any;
  export const reports: any;
  export const cases: any;
  export const pgvector: any;
  export default {} as any;
}

declare module '$lib/server/db/schema-postgres.js' {
  export * from '$lib/server/db/schema-postgres';
}

declare module '$lib/server/db/*' {
  const whatever: any;
  export default whatever;
}

// Generic fallback for runtime-only modules used in a few paths.
declare module '$lib/server/*' {
  const whatever: any;
  export default whatever;
}
