// Shim module to select between real Context7 WASM and the mock implementation
// Usage: import context7 from '$lib/integrations/context7-wasm';

const useMock = Boolean(import.meta.env.VITE_USE_CONTEXT7_WASM_MOCK || import.meta.env.USE_CONTEXT7_WASM_MOCK);

let impl: any = null;

if (useMock) {
  // dynamic import of the mock to keep bundle small
  impl = await import('./context7-wasm-mock').then(m => m.default);
} else {
  try {
    // attempt to load a real WASM binding (may not exist in this repo)
    impl = await import('@context7/wasm').then(m => m.default).catch(() => null);
    if (!impl) {
      // fallback to mock automatically
      impl = await import('./context7-wasm-mock').then(m => m.default);
    }
  } catch {
    impl = await import('./context7-wasm-mock').then(m => m.default);
  }
}

export default impl;
