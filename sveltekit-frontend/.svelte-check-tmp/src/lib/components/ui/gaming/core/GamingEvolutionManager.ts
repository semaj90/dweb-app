/**
 * Gaming Evolution Manager
 * Handles progressive enhancement from 8-bit → 16-bit → N64
 * 
 * Features:
 * - Performance-based era selection
 * - Smooth transitions between gaming eras
 * - Device capability detection
 * - Memory and CPU optimization
 */

import type { 
  GamingEra, 
  GamingThemeState, 
  ProgressiveGamingConfig,
  GamepadState 
} from '../types/gaming-types.js';

import { 
  GAMING_BREAKPOINTS, 
  GAMING_ERA_SPECS,
  N64_TEXTURE_PRESETS 
} from '../constants/gaming-constants.js';

interface DeviceCapabilities {
  memory: number; // GB
  cores: number;
  gpu: 'basic' | 'discrete' | 'integrated' | 'unknown';
  connection: 'slow' | 'fast' | 'unknown';
  screenSize: { width: number; height: number };
  pixelRatio: number;
  webgl: boolean;
  webgpu: boolean;
}

export class GamingEvolutionManager {
  private static instance: GamingEvolutionManager;
  private capabilities: DeviceCapabilities | null = null;
  private currentState: GamingThemeState;
  private config: ProgressiveGamingConfig;
  private performanceObserver: PerformanceObserver | null = null;
  private frameMetrics: number[] = [];
  private listeners: Set<(state: GamingThemeState) => void> = new Set();

  private constructor(config: Partial<ProgressiveGamingConfig> = {}) {
    this.config = {
      defaultEra: 'auto',
      enableAutoEvolution: true,
      performanceThreshold: 16.67, // 60fps in milliseconds
      
      nesSettings: {
        strictPalette: true,
        enableScanlines: true,
        pixelScale: 2
      },
      
      snesSettings: {
        enableGradients: true,
        enableModeViitColors: true,
        layerCount: 4
      },
      
      n64Settings: {
        ...N64_TEXTURE_PRESETS.balanced,
        enableRealTimeReflections: false,
        textureQuality: 'standard'
      },
      
      yorhaIntegration: true,
      bitsUICompatibility: true,
      ...config
    };

    this.currentState = {
      currentEra: this.config.defaultEra,
      availableEras: ['8bit', '16bit', 'n64'],
      isTransitioning: false,
      transitionDuration: 300,
      performanceLevel: 'medium'
    };

    this.initialize();
  }

  public static getInstance(config?: Partial<ProgressiveGamingConfig>): GamingEvolutionManager {
    if (!GamingEvolutionManager.instance) {
      GamingEvolutionManager.instance = new GamingEvolutionManager(config);
    }
    return GamingEvolutionManager.instance;
  }

  private async initialize(): Promise<void> {
    if (typeof window !== 'undefined') {
      await this.detectDeviceCapabilities();
      this.setupPerformanceMonitoring();
      this.determineOptimalEra();
      
      // Listen for device changes
      window.addEventListener('resize', () => this.handleDeviceChange());
      
      // Check for memory pressure events
      if ('memory' in performance) {
        this.monitorMemoryPressure();
      }
    }
  }

  private async detectDeviceCapabilities(): Promise<void> {
    if (typeof window === 'undefined') return;

    const capabilities: DeviceCapabilities = {
      memory: (navigator as any).deviceMemory || 4, // Default to 4GB
      cores: navigator.hardwareConcurrency || 2,
      gpu: await this.detectGPUCapability(),
      connection: this.detectConnectionSpeed(),
      screenSize: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      pixelRatio: window.devicePixelRatio || 1,
      webgl: this.hasWebGL(),
      webgpu: await this.hasWebGPU()
    };

    this.capabilities = capabilities;
    console.log('🎮 Detected device capabilities:', capabilities);
  }

  private async detectGPUCapability(): Promise<DeviceCapabilities['gpu']> {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
      
      if (!gl) return 'basic';
      
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (!debugInfo) return 'integrated';
      
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      
      if (typeof renderer === 'string' && (renderer.includes('NVIDIA') || renderer.includes('AMD') || renderer.includes('Radeon'))) {
        return 'discrete';
      }
      
      return 'integrated';
    } catch {
      return 'unknown';
    }
  }

  private detectConnectionSpeed(): DeviceCapabilities['connection'] {
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      const effectiveType = connection.effectiveType;
      
      return effectiveType === '4g' || effectiveType === '5g' ? 'fast' : 'slow';
    }
    
    return 'unknown';
  }

  private hasWebGL(): boolean {
    try {
      const canvas = document.createElement('canvas');
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    } catch {
      return false;
    }
  }

  private async hasWebGPU(): Promise<boolean> {
    try {
      return 'gpu' in navigator && await (navigator as any).gpu?.requestAdapter() !== null;
    } catch {
      return false;
    }
  }

  private setupPerformanceMonitoring(): void {
    if (typeof window === 'undefined' || !this.config.enableAutoEvolution) return;

    try {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        
        entries.forEach(entry => {
          if (entry.entryType === 'measure' || entry.entryType === 'navigation') {
            this.frameMetrics.push(entry.duration);
            
            // Keep only last 60 measurements
            if (this.frameMetrics.length > 60) {
              this.frameMetrics.shift();
            }
            
            this.evaluatePerformance();
          }
        });
      });

      this.performanceObserver.observe({ entryTypes: ['measure', 'navigation'] });
    } catch (error) {
      console.warn('Performance monitoring not supported:', error);
    }
  }

  private monitorMemoryPressure(): void {
    const checkMemory = () => {
      const memory = (performance as any).memory;
      if (memory) {
        const memoryRatio = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
        
        if (memoryRatio > 0.9) {
          // High memory pressure - downgrade era
          this.downgradeEra();
        } else if (memoryRatio < 0.5 && this.currentState.performanceLevel === 'low') {
          // Memory freed up - potentially upgrade
          this.upgradeEra();
        }
      }
    };

    setInterval(checkMemory, 5000); // Check every 5 seconds
  }

  private evaluatePerformance(): void {
    if (this.frameMetrics.length < 10) return;

    const averageFrameTime = this.frameMetrics.reduce((a, b) => a + b, 0) / this.frameMetrics.length;
    const performanceLevel = this.getPerformanceLevel(averageFrameTime);

    if (performanceLevel !== this.currentState.performanceLevel) {
      this.updatePerformanceLevel(performanceLevel);
    }
  }

  private getPerformanceLevel(frameTime: number): GamingThemeState['performanceLevel'] {
    if (frameTime > this.config.performanceThreshold * 2) {
      return 'low';
    } else if (frameTime > this.config.performanceThreshold) {
      return 'medium';
    } else {
      return 'high';
    }
  }

  private updatePerformanceLevel(level: GamingThemeState['performanceLevel']): void {
    this.currentState = {
      ...this.currentState,
      performanceLevel: level
    };

    // Auto-adjust era based on performance
    if (this.config.enableAutoEvolution) {
      if (level === 'low' && this.currentState.currentEra === 'n64') {
        this.downgradeEra();
      } else if (level === 'high' && this.currentState.currentEra === '8bit') {
        this.upgradeEra();
      }
    }

    this.notifyListeners();
  }

  private determineOptimalEra(): GamingEra {
    if (!this.capabilities) return '8bit';

    const { memory, cores, gpu, webgl, webgpu } = this.capabilities;

    // N64 requirements: Good GPU, WebGL, 4GB+ memory
    if (webgpu || (webgl && gpu !== 'basic' && memory >= 4 && cores >= 4)) {
      return 'n64';
    }
    
    // SNES requirements: Moderate specs
    if (memory >= 2 && cores >= 2) {
      return '16bit';
    }
    
    // NES: Universal fallback
    return '8bit';
  }

  public async setEra(era: GamingEra): Promise<void> {
    if (era === this.currentState.currentEra) return;

    this.currentState = {
      ...this.currentState,
      isTransitioning: true
    };

    this.notifyListeners();

    // Wait for transition
    await new Promise(resolve => setTimeout(resolve, this.currentState.transitionDuration));

    this.currentState = {
      ...this.currentState,
      currentEra: era,
      isTransitioning: false
    };

    this.notifyListeners();
    console.log(`🎮 Gaming era switched to: ${era}`);
  }

  public async upgradeEra(): Promise<void> {
    const currentIndex = this.currentState.availableEras.indexOf(this.currentState.currentEra);
    if (currentIndex < this.currentState.availableEras.length - 1) {
      const nextEra = this.currentState.availableEras[currentIndex + 1];
      await this.setEra(nextEra);
    }
  }

  public async downgradeEra(): Promise<void> {
    const currentIndex = this.currentState.availableEras.indexOf(this.currentState.currentEra);
    if (currentIndex > 0) {
      const prevEra = this.currentState.availableEras[currentIndex - 1];
      await this.setEra(prevEra);
    }
  }

  private handleDeviceChange(): void {
    // Re-detect capabilities on device change
    setTimeout(() => {
      this.detectDeviceCapabilities().then(() => {
        if (this.config.enableAutoEvolution) {
          const optimalEra = this.determineOptimalEra();
          if (optimalEra !== this.currentState.currentEra) {
            this.setEra(optimalEra);
          }
        }
      });
    }, 100);
  }

  public subscribe(callback: (state: GamingThemeState) => void): () => void {
    this.listeners.add(callback);
    
    // Immediately call with current state
    callback(this.currentState);
    
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notifyListeners(): void {
    this.listeners.forEach(callback => callback(this.currentState));
  }

  public getCurrentState(): GamingThemeState {
    return { ...this.currentState };
  }

  public getCapabilities(): DeviceCapabilities | null {
    return this.capabilities ? { ...this.capabilities } : null;
  }

  public getConfig(): ProgressiveGamingConfig {
    return { ...this.config };
  }

  public updateConfig(updates: Partial<ProgressiveGamingConfig>): void {
    this.config = { ...this.config, ...updates };
    
    if (updates.enableAutoEvolution !== undefined) {
      if (updates.enableAutoEvolution) {
        this.setupPerformanceMonitoring();
      } else if (this.performanceObserver) {
        this.performanceObserver.disconnect();
        this.performanceObserver = null;
      }
    }
  }

  public dispose(): void {
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
      this.performanceObserver = null;
    }
    
    this.listeners.clear();
    this.frameMetrics = [];

    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', this.handleDeviceChange);
    }
  }
}