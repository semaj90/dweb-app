// Quick fixes: declare commonly missing globals/modules as `any` to reduce TS noise
// This is an incremental remediation step. Replace with proper types over time.

declare const legalBERT: any;
declare const legalRAG: any;
declare const qdrantService: any;
declare const nomicEmbedText: any;
declare const vectorOps: any;

type RecommendationRequest = any;
type VectorSearchOptions = any;

declare module '$lib/server/db/drizzle' {
  const _default: any;
  export default _default;
}

declare module '$lib/server/cache/redis-service' {
  const redis: any;
  export { redis };
}

declare module 'node-fetch' {
  const fetch: any;
  export default fetch;
}

// Generic augmentations to App.Locals (complements locals-allow-any)
declare namespace App {
  interface Locals {
    db?: any;
    user?: any;
    orchestrator?: any;
    minio?: any;
    embed?: any;
    llama?: any;
    llamaWasm?: any;
    pgClient?: any;
    pgVectorClient?: any;
    pg?: any;
    [key: string]: any;
  }
}

// Allow unknown modules used in the codebase to be imported without type errors
declare module '*-service' { const x: any; export default x; }
declare module '*service' { const x: any; export default x; }
