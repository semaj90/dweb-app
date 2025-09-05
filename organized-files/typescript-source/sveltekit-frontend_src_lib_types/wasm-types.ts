/**
 * Lightweight, exported WASM type helpers for local imports.
 *
 * These avoid augmenting the global WebAssembly namespace while providing
 * explicit, safe typings for code that imports WASM helpers from this module.
 */

export type WasmBytes = ArrayBuffer | Uint8Array;

export interface WasmModule {
  exports: Record<string, any>;
}

export interface WasmInstance {
  exports: Record<string, any>;
}

export interface WasmInstantiateResult {
  instance: WasmInstance;
  module: WasmModule;
}

export interface WasmImports {
  [moduleKey: string]: {
	[exportKey: string]: any;
  };
}

/**
 * Optional helper interface for objects that load/instantiate WASM.
 */
export interface WasmLoader {
  instantiate(bytes: WasmBytes, imports?: WasmImports): Promise<WasmInstantiateResult>;
  instantiateStreaming?(response: Response | Promise<Response>, imports?: WasmImports): Promise<WasmInstantiateResult>;
}
