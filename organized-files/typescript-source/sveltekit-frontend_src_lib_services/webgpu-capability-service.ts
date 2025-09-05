/**
 * WebGPU Capability Service - Client-side WebGPU feature detection and gating
 * Provides centralized capability checking and fallback management
 */

import { browser } from '$app/environment';

export interface WebGPUCapabilities {
  isSupported: boolean;
  isAvailable: boolean;
  adapter?: GPUAdapter;
  device?: GPUDevice;
  features: Set<string>;
  limits: Record<string, number>;
  supportLevel: 'none' | 'basic' | 'advanced' | 'full';
  fallbackReason?: string;
}

export interface WebGPUConfig {
  requiredFeatures?: string[];
  preferredPowerPreference?: GPUPowerPreference;
  requiredLimits?: Record<string, number>;
  enableFallback: boolean;
  timeoutMs: number;
}

class WebGPUCapabilityService {
  private capabilities: WebGPUCapabilities | null = null;
  private initPromise: Promise<WebGPUCapabilities> | null = null;
  private config: WebGPUConfig = {
    enableFallback: true,
    timeoutMs: 5000
  };

  /**
   * Initialize WebGPU capabilities with configuration
   */
  async initialize(config?: Partial<WebGPUConfig>): Promise<WebGPUCapabilities> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.config = { ...this.config, ...config };
    this.initPromise = this.detectCapabilities();
    return this.initPromise;
  }

  /**
   * Get cached capabilities (must call initialize first)
   */
  getCapabilities(): WebGPUCapabilities | null {
    return this.capabilities;
  }

  /**
   * Check if WebGPU is supported and available
   */
  isSupported(): boolean {
    return this.capabilities?.isSupported ?? false;
  }

  /**
   * Check if WebGPU is available for use
   */
  isAvailable(): boolean {
    return this.capabilities?.isAvailable ?? false;
  }

  /**
   * Get support level for progressive enhancement
   */
  getSupportLevel(): 'none' | 'basic' | 'advanced' | 'full' {
    return this.capabilities?.supportLevel ?? 'none';
  }

  /**
   * Check if specific features are supported
   */
  hasFeatures(requiredFeatures: string[]): boolean {
    if (!this.capabilities?.features) return false;
    return requiredFeatures.every(feature => this.capabilities!.features.has(feature));
  }

  /**
   * Check if device meets minimum limits
   */
  meetsLimits(requiredLimits: Record<string, number>): boolean {
    if (!this.capabilities?.limits) return false;
    return Object.entries(requiredLimits).every(([key, value]) => {
      const deviceLimit = this.capabilities!.limits[key];
      return deviceLimit !== undefined && deviceLimit >= value;
    });
  }

  /**
   * Get fallback reason if WebGPU is not available
   */
  getFallbackReason(): string | null {
    return this.capabilities?.fallbackReason ?? null;
  }

  /**
   * Create a WebGPU-enabled component props with capability gating
   */
  createComponentProps(componentName: string) {
    const capabilities = this.getCapabilities();
    return {
      webgpuSupported: capabilities?.isSupported ?? false,
      webgpuAvailable: capabilities?.isAvailable ?? false,
      supportLevel: capabilities?.supportLevel ?? 'none',
      fallbackReason: capabilities?.fallbackReason,
      enableWebGPU: capabilities?.isAvailable ?? false,
      componentName
    };
  }

  private async detectCapabilities(): Promise<WebGPUCapabilities> {
    console.log('🎮 WebGPU Capability Detection Starting...');

    // Server-side rendering check
    if (!browser) {
      this.capabilities = {
        isSupported: false,
        isAvailable: false,
        features: new Set(),
        limits: {},
        supportLevel: 'none',
        fallbackReason: 'Server-side rendering environment'
      };
      console.log('⚠️ WebGPU: SSR environment detected, using fallback');
      return this.capabilities;
    }

    // Browser WebGPU API check
    if (!navigator.gpu) {
      this.capabilities = {
        isSupported: false,
        isAvailable: false,
        features: new Set(),
        limits: {},
        supportLevel: 'none',
        fallbackReason: 'WebGPU API not available in this browser'
      };
      console.log('⚠️ WebGPU: API not available, using fallback');
      return this.capabilities;
    }

    try {
      // Request adapter with timeout
      const adapterPromise = navigator.gpu.requestAdapter({
        powerPreference: this.config.preferredPowerPreference ?? 'high-performance'
      });

      const timeoutPromise = new Promise<null>((_, reject) => {
        setTimeout(() => reject(new Error('WebGPU adapter request timeout')), this.config.timeoutMs);
      });

      const adapter = await Promise.race([adapterPromise, timeoutPromise]);

      if (!adapter) {
        this.capabilities = {
          isSupported: true,
          isAvailable: false,
          features: new Set(),
          limits: {},
          supportLevel: 'none',
          fallbackReason: 'No suitable WebGPU adapter found'
        };
        console.log('⚠️ WebGPU: No adapter available, using fallback');
        return this.capabilities;
      }

      // Request device
      const devicePromise = adapter.requestDevice({
        requiredFeatures: this.config.requiredFeatures as GPUFeatureName[] ?? []
      });

      const device = await Promise.race([
        devicePromise,
        new Promise<null>((_, reject) => {
          setTimeout(() => reject(new Error('WebGPU device request timeout')), this.config.timeoutMs);
        })
      ]);

      if (!device) {
        this.capabilities = {
          isSupported: true,
          isAvailable: false,
          adapter,
          features: new Set(),
          limits: {},
          supportLevel: 'basic',
          fallbackReason: 'WebGPU device request failed'
        };
        console.log('⚠️ WebGPU: Device request failed, using fallback');
        return this.capabilities;
      }

      // Determine support level based on features and limits
      const features = new Set(device.features);
      const limits = Object.fromEntries(
        Object.entries(device.limits).map(([key, value]) => [key, value as number])
      );

      const supportLevel = this.determineSupportLevel(features, limits);

      this.capabilities = {
        isSupported: true,
        isAvailable: true,
        adapter,
        device,
        features,
        limits,
        supportLevel
      };

      console.log(`✅ WebGPU: Initialized successfully with ${supportLevel} support`);
      console.log(`   Features: ${Array.from(features).join(', ') || 'none'}`);
      console.log(`   Max compute workgroups: ${limits.maxComputeWorkgroupsPerDimension || 'unknown'}`);
      
      return this.capabilities;

    } catch (error) {
      this.capabilities = {
        isSupported: true,
        isAvailable: false,
        features: new Set(),
        limits: {},
        supportLevel: 'none',
        fallbackReason: `WebGPU initialization error: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
      console.error('❌ WebGPU: Initialization failed:', error);
      return this.capabilities;
    }
  }

  private determineSupportLevel(features: Set<string>, limits: Record<string, number>): 'none' | 'basic' | 'advanced' | 'full' {
    // Full support: compute shaders with advanced features
    if (features.has('timestamp-query') && features.has('indirect-first-instance') && 
        limits.maxComputeWorkgroupsPerDimension >= 65535) {
      return 'full';
    }

    // Advanced support: compute shaders with good limits
    if (limits.maxComputeWorkgroupsPerDimension >= 32768 && 
        limits.maxStorageBufferBindingSize >= 1024 * 1024 * 128) {
      return 'advanced';
    }

    // Basic support: basic compute capability
    if (limits.maxComputeWorkgroupsPerDimension >= 256) {
      return 'basic';
    }

    return 'none';
  }

  /**
   * Create a wrapper that gracefully handles WebGPU operations with fallback
   */
  wrapWebGPUOperation<T>(
    webgpuOperation: () => Promise<T>,
    fallbackOperation: () => Promise<T>,
    operationName: string
  ): Promise<T> {
    if (!this.isAvailable()) {
      console.log(`🔄 ${operationName}: Using fallback (WebGPU not available)`);
      return fallbackOperation();
    }

    return webgpuOperation().catch(error => {
      console.warn(`⚠️ ${operationName}: WebGPU operation failed, using fallback:`, error);
      return fallbackOperation();
    });
  }
}

// Singleton instance
export const webgpuCapability = new WebGPUCapabilityService();

// Auto-initialize on module load in browser
if (browser) {
  webgpuCapability.initialize().catch(error => {
    console.error('WebGPU capability service auto-initialization failed:', error);
  });
}

// Utility function for components
export function useWebGPUCapability() {
  return {
    service: webgpuCapability,
    initialize: (config?: Partial<WebGPUConfig>) => webgpuCapability.initialize(config),
    isSupported: () => webgpuCapability.isSupported(),
    isAvailable: () => webgpuCapability.isAvailable(),
    getSupportLevel: () => webgpuCapability.getSupportLevel(),
    getCapabilities: () => webgpuCapability.getCapabilities(),
    createComponentProps: (name: string) => webgpuCapability.createComponentProps(name)
  };
}