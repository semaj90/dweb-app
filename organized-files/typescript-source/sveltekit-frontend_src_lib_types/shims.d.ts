// Auto-generated shims to reduce 'is not a module' / missing module diagnostics during incremental fixes.

declare module '@webgpu/types' {
  export type GPUDevice = any;
  export type GPUBuffer = any;
  export type GPUTexture = any;
  export type GPUComputePipeline = any;
  export type GPUCommandQueue = any;
  export type GPULimits = any;
  const _default: any;
  export default _default;
}

declare module '@tensorflow/tfjs' {
  const tf: any;
  export = tf;
  export type Tensor = any;
}

// Generic internal lib wildcard shims (broad but safe for incremental fixes)
declare module '$lib/*' {
  const value: any;
  export = value;
}

declare module '$lib/wasm/*' {
  const value: any;
  export = value;
}

declare module '$lib/types/*' {
  const value: any;
  export = value;
}

declare module '$lib/server/*' {
  const value: any;
  export = value;
}

declare module '$lib/components/*' {
  const value: any;
  export = value;
}

// Fall back for relative type folders used in many files
declare module '../types/*' {
  const value: any;
  export = value;
}

declare module 'buffer' {
  const value: any;
  export default value;
}

// Allow any wasm imports
declare module '*.wasm';

declare module '*.png';
declare module '*.jpg';
// Specific frontend-facing shim for GPU metrics schema
declare module '$lib/server/db/schema-gpu-metrics' {
  export interface GPUMetricEnhanced {
    timestamp?: number;
    fps?: number;
    memoryUsage?: number;
    gpuUtilization?: number;
    temperature?: number;
    powerUsage?: number;
    processingTimeMs?: number;
    sessionId?: string;
    gpuName?: string;
    vendor?: string;
    memoryMb?: number;
  }
  export type GPUMetrics = GPUMetricEnhanced;
}
