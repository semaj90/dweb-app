// WebAssembly Type Definitions
// Provides type definitions for WebAssembly API and related interfaces

declare global {
  interface WebAssembly {
    Module: {
      new(): WebAssemblyModule;
    };
    instantiate(bytes: ArrayBuffer | Uint8Array, imports?: WebAssemblyImports): Promise<WebAssemblyInstantiateResult>;
    instantiateStreaming(response: Response, imports?: WebAssemblyImports): Promise<WebAssemblyInstantiateResult>;
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

  // Worker types for WebAssembly operations
  interface WebAssemblyWorkerMessage {
    type: 'initialize' | 'generate' | 'cleanup' | 'status';
    data?: any;
    requestId?: string;
  }

  interface WebAssemblyWorkerResponse {
    type: 'initialized' | 'generated' | 'cleaned' | 'status' | 'error';
    data?: any;
    error?: string;
    requestId?: string;
  }

  // LLAMA.cpp specific interfaces
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
    repeat_penalty?: number;
    stop_sequences?: string[];
    seed?: number;
  }

  interface LlamaGenerationResult {
    text: string;
    tokens_generated: number;
    processing_time: number;
    tokens_per_second: number;
    stopped_eos: boolean;
    stopped_limit: boolean;
    stopped_word: boolean;
  }

  // Import value types for WebAssembly imports (use WebAssembly.ImportValue instead)
  interface WASMImportValue extends WebAssembly.ImportValue {}
}

// Export empty object to make this a module
export {};