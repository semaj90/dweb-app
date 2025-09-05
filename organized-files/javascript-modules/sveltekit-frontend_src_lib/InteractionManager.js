// InteractionManager.js
// Persistent BVH loader with Wasm accelerator (lazy) and JS fallback.
export default class InteractionManager {
  constructor() {
    this.module = null; // Emscripten module instance
    this.built = false;
    this.documents = [];
    this.dim = 0;
    this.flatBuffer = null;
    this.useWasm = false;
  }

  async init(documents = []) {
    if (this.built) return;
    this.documents = documents || [];
    if (this.documents.length === 0) { this.built = true; return; }
    this.dim = (this.documents[0].vector || []).length || 0;

    // build flat Float32Array
    const n = this.documents.length;
    this.flatBuffer = new Float32Array(n * this.dim);
    for (let i = 0; i < n; i++) {
      const v = this.documents[i].vector || [];
      for (let d = 0; d < this.dim; d++) {
        this.flatBuffer[i * this.dim + d] = v[d] || 0;
      }
    }

    // try to lazy-load Wasm BVH
    try {
      // Check if WASM module is already available in window
      let modImport = null;
      if (typeof window !== 'undefined' && (window as any).createBVHModule) {
        modImport = { default: (window as any).createBVHModule };
      } else if (typeof window !== 'undefined') {
        // Load via script tag to avoid Vite import restrictions
        const script = document.createElement('script');
        script.src = '/wasm/bvh_accelerator.js';
        script.type = 'text/javascript';
        
        await new Promise((resolve, reject) => {
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);
        });
        
        if ((window as any).createBVHModule) {
          modImport = { default: (window as any).createBVHModule };
        }
      }
      
      if (modImport) {
        // Emscripten MODULARIZE=1 factory is default export or named
        const factory = modImport.default || modImport.createBVHModule || window.createBVHModule || modImport.createModule;
        if (typeof factory === 'function') {
          const Module = await factory();
          this.module = Module;
          // Call exported embind build function
          if (typeof Module.build_index === 'function') {
            Module.build_index(this.flatBuffer, this.dim, n);
            this.useWasm = true;
          }
        }
      }
    } catch (e) {
      console.warn('Wasm BVH not available or failed to initialize, falling back to JS:', e);
      this.module = null;
      this.useWasm = false;
    }

    this.built = true;
  }

  // k-NN query returns array of {index, distance}
  async queryNearest(queryVec, k = 5) {
    if (!this.built) await this.init(this.documents);
    if (!queryVec || queryVec.length !== this.dim) {
      throw new Error('query vector length mismatch');
    }

    if (this.useWasm && this.module && typeof this.module.knn_search === 'function') {
      try {
        // knn_search accepts Float32Array and k, returns Int32Array or Array
        const res = this.module.knn_search(new Float32Array(queryVec), k);
        // embind may return a JS array
        const indices = Array.isArray(res) ? res : Array.from(res);
        return indices.map((idx, rank) => ({ index: idx, distance: null }));
      } catch (e) {
        console.warn('Wasm knn_search failed, falling back to JS', e);
        // fall through to JS fallback
      }
    }

    // JS brute-force fallback (cosine similarity)
    const results = [];
    for (let i = 0; i < this.documents.length; i++) {
      const base = i * this.dim;
      let dot = 0, a2 = 0, b2 = 0;
      for (let d = 0; d < this.dim; d++) {
        const a = this.flatBuffer[base + d];
        const b = queryVec[d] || 0;
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
      }
      const denom = Math.sqrt(a2) * Math.sqrt(b2) || 1e-12;
      const cos = dot / denom;
      results.push({ index: i, score: cos });
    }
    results.sort((x, y) => y.score - x.score);
    return results.slice(0, k).map(r => ({ index: r.index, distance: 1 - r.score }));
  }

  dispose() {
    try {
      if (this.module && typeof this.module.free_index === 'function') {
        this.module.free_index();
      }
    } catch (e) {
      // ignore
    }
    this.module = null;
    this.built = false;
    this.documents = [];
    this.flatBuffer = null;
    this.useWasm = false;
  }
}
