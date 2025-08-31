// Wildcard $lib shim: expose common named exports as permissive "any" to reduce
// large-volume errors during migration. Keep minimal and extend as needed.

declare module '$lib/*' {
  const _any: any;
  export default _any;

  // common high-noise named exports used across the repo — declared permissively
  export const Case: any;
  export const Evidence: any;
  export const AIFindResult: any;
  export const VectorService: any;
  export const ollamaService: any;
  export const db: any;
  export const generateEmbedding: any;
  export const generateBatchEmbeddings: any;
  export const logger: any;
  export const productionAPIClient: any;
  export const PROTOCOL_TIERS: any;
  export const productionServiceRegistry: any;
}

export {};
