/**
 * Enhanced WebAssembly type declarations for the project.
 *
 * Provides safer, explicit typings for WebAssembly usage, worker messages,
 * and simple LLAMA.cpp-related parameter shapes used across the codebase.
 *
 * This file intentionally augments the global scope so imports can use these
 * types without explicit imports.
 */

declare global {
  interface WebAssembly {
	Module: {
	  new (): WebAssemblyModule;
	};
	instantiate(bytes: ArrayBuffer | Uint8Array, imports?: WebAssemblyImports): Promise<WebAssemblyInstantiateResult>;
	instantiateStreaming(response: Response | Promise<Response>, imports?: WebAssemblyImports): Promise<WebAssemblyInstantiateResult>;
  }

  interface WebAssemblyModule {
	exports: Record<string, any>;
  }

  interface WebAssemblyInstance {
	exports: Record<string, any>;
  }

  interface WebAssemblyInstantiateResult {
	instance: WebAssemblyInstance;
	module: WebAssemblyModule;
  }

  interface WebAssemblyImports {
	[moduleKey: string]: {
	  [exportKey: string]: any;
	};
  }

  // Worker message/response shapes for WebAssembly operations
  interface WebAssemblyWorkerMessage {
	type: 'initialize' | 'generate' | 'cleanup' | 'status' | string;
	data?: any;
	requestId?: string;
  }

  interface WebAssemblyWorkerResponse {
	type: 'initialized' | 'generated' | 'cleaned' | 'status' | 'error' | string;
	data?: any;
	error?: string;
	requestId?: string;
  }

  // LLAMA.cpp specific interfaces (used by inference layers)
  interface LlamaContext {
	model_path: string;
	context_size: number;
	threads: number;
	temperature: number;
	top_p: number;
	top_k: number;
	repeat_penalty: number;
  }

  interface LlamaGenerationParams {
	prompt: string;
	max_tokens?: number;
	temperature?: number;
	top_p?: number;
	top_k?: number;
  }
}

export {};
