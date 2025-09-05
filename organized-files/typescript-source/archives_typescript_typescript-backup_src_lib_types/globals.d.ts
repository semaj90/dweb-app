// Project-global lightweight type declarations to reduce temporary tsc noise

declare module '$lib/db/client' {
  import type { DBClient } from '$lib/db/client';
  export const db: DBClient;
  export default db;
}

declare module '$lib/db/client-db' {
  export const legalDB: any;
  export const LegalDBUtils: any;
  export type GraphVisualizationData = any;
  export type DocumentCache = any;
  export type VectorSearchCache = any;
}

declare global {
  interface Window {
    __DEV__: boolean;
  }
}
