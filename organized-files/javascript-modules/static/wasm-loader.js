// WebAssembly Loader for Cyber Elephant Components
// Handles loading and initialization of WASM modules with fallbacks

class WasmLoader {
  constructor() {
    this.modules = new Map();
    this.initialized = false;
    this.fallbackMode = false;
  }

  async initialize() {
    if (this.initialized) return;

    try {
      // Check WebAssembly support
      if (typeof WebAssembly === 'undefined') {
        console.warn('WebAssembly not supported, using JavaScript fallbacks');
        this.fallbackMode = true;
      } else {
        // Test basic WASM functionality
        const testModule = await WebAssembly.instantiate(
          new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00])
        );
        console.log('✅ WebAssembly support confirmed');
      }
    } catch (error) {
      console.warn('WebAssembly test failed, using fallbacks:', error);
      this.fallbackMode = true;
    }

    this.initialized = true;
  }

  async loadModule(name, wasmPath, jsPath) {
    await this.initialize();

    if (this.modules.has(name)) {
      return this.modules.get(name);
    }

    let module;
    
    if (!this.fallbackMode) {
      try {
        // Try to load actual WASM
        const wasmResponse = await fetch(wasmPath);
        if (wasmResponse.ok) {
          const wasmBytes = await wasmResponse.arrayBuffer();
          const wasmModule = await WebAssembly.instantiate(wasmBytes);
          module = {
            type: 'wasm',
            instance: wasmModule.instance,
            module: wasmModule.module,
            exports: wasmModule.instance.exports
          };
          console.log(`🚀 Loaded WASM module: ${name}`);
        } else {
          throw new Error(`Failed to fetch WASM: ${wasmResponse.status}`);
        }
      } catch (error) {
        console.warn(`WASM load failed for ${name}, falling back to JS:`, error);
        this.fallbackMode = true;
      }
    }

    if (this.fallbackMode || !module) {
      // Load JavaScript fallback
      try {
        if (jsPath.endsWith('.js')) {
          // Dynamic import for ES modules
          const jsModule = await import(jsPath);
          module = {
            type: 'js',
            exports: jsModule.default || jsModule
          };
        } else {
          // Script tag loading for compatibility
          const script = document.createElement('script');
          script.src = jsPath;
          script.async = true;
          
          await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
          });
          
          module = {
            type: 'js',
            exports: window[name] || eval(name)
          };
        }
        console.log(`📜 Loaded JS fallback: ${name}`);
      } catch (error) {
        console.error(`Failed to load module ${name}:`, error);
        throw error;
      }
    }

    this.modules.set(name, module);
    return module;
  }

  async loadCyberElephantBVH() {
    return this.loadModule(
      'CyberElephantBVH',
      '/cyber-elephant-bvh.wasm', // Would be actual WASM in production
      '/cyber-elephant-bvh.wasm.js' // JavaScript implementation
    );
  }

  async loadRankingCache() {
    return this.loadModule(
      'RankingCache', 
      '/ranking-cache.wasm',
      '/src/lib/wasm/ranking_wasm_stub.js'
    );
  }

  getModuleInfo(name) {
    const module = this.modules.get(name);
    if (!module) return null;

    return {
      name,
      type: module.type,
      loaded: true,
      memoryUsage: this.getMemoryUsage(module),
      performance: this.getPerformanceMetrics(module)
    };
  }

  getMemoryUsage(module) {
    if (module.type === 'wasm' && module.instance.exports.memory) {
      return module.instance.exports.memory.buffer.byteLength;
    }
    return 0; // JS modules don't expose memory directly
  }

  getPerformanceMetrics(module) {
    // Basic performance estimation
    return {
      type: module.type,
      estimatedSpeed: module.type === 'wasm' ? '10x faster' : 'baseline',
      parallelizable: module.type === 'wasm'
    };
  }

  // Utility method to check if specific features are available
  hasFeature(feature) {
    switch (feature) {
      case 'simd':
        return typeof WebAssembly !== 'undefined' && 
               WebAssembly.validate && 
               this.testSIMDSupport();
      case 'threads':
        return typeof SharedArrayBuffer !== 'undefined' && 
               typeof Atomics !== 'undefined';
      case 'bulk-memory':
        return this.testBulkMemorySupport();
      default:
        return false;
    }
  }

  testSIMDSupport() {
    // Test for WebAssembly SIMD support
    try {
      return WebAssembly.validate(new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
        0x03, 0x02, 0x01, 0x00,
        0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c, 0x00,
        0x00, 0x00, 0x00, 0x0b
      ]));
    } catch {
      return false;
    }
  }

  testBulkMemorySupport() {
    try {
      return WebAssembly.validate(new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        0x03, 0x02, 0x01, 0x00,
        0x05, 0x03, 0x01, 0x00, 0x01,
        0x0a, 0x07, 0x01, 0x05, 0x00, 0xfc, 0x08, 0x00, 0x00, 0x0b
      ]));
    } catch {
      return false;
    }
  }

  // Global performance monitoring
  getSystemInfo() {
    return {
      wasmSupported: typeof WebAssembly !== 'undefined',
      fallbackMode: this.fallbackMode,
      loadedModules: Array.from(this.modules.keys()),
      features: {
        simd: this.hasFeature('simd'),
        threads: this.hasFeature('threads'),
        bulkMemory: this.hasFeature('bulk-memory')
      },
      memoryInfo: this.getMemoryInfo()
    };
  }

  getMemoryInfo() {
    const info = { totalAllocated: 0, moduleBreakdown: {} };
    
    for (const [name, module] of this.modules) {
      const usage = this.getMemoryUsage(module);
      info.totalAllocated += usage;
      info.moduleBreakdown[name] = usage;
    }
    
    return info;
  }
}

// Global instance
const wasmLoader = new WasmLoader();

// Export for both browser and Node.js
if (typeof window !== 'undefined') {
  window.WasmLoader = WasmLoader;
  window.wasmLoader = wasmLoader;
} else if (typeof module !== 'undefined') {
  module.exports = { WasmLoader, wasmLoader };
}