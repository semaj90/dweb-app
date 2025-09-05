/**
 * NES.css + YoRHa 3D Integration Service
 * Revolutionary service that bridges 8-bit DOM aesthetics with 3D GPU interfaces
 * 
 * This service manages:
 * - Hybrid rendering between DOM and WebGL
 * - NES-style memory management for 3D components
 * - Canvas-to-DOM synchronization
 * - Predictive caching based on user interactions
 * - CRT and scanline effects for authentic retro feel
 */

import { NESYoRHaHybrid3D, type NESYoRHaHybridStyle, NES_YORHA_PALETTE } from '$lib/components/three/yorha-ui/NESYoRHaHybrid3D';
import { nesCacheOrchestrator } from './nes-cache-orchestrator';
import type { CanvasState } from '$lib/stores/canvas-states';
import { dev } from '$app/environment';
import * as THREE from 'three';

// Integration configuration
export interface NESYoRHaConfig {
  enableHybridSync: boolean;
  enableCRTEffects: boolean;
  enablePixelPerfect: boolean;
  cachePreloadThreshold: number;
  maxHybridComponents: number;
  renderMode: '2d-overlay' | '3d-embedded' | 'hybrid-sync';
  memoryBudget: number; // in bytes
}

export interface ComponentRegistry {
  id: string;
  component: NESYoRHaHybrid3D;
  domElement?: HTMLElement;
  lastInteraction: number;
  predictedStates: string[];
  cacheKeys: string[];
}

export interface UserInteractionPattern {
  componentId: string;
  interactionType: 'hover' | 'click' | 'focus' | 'drag';
  timestamp: number;
  context: Record<string, any>;
  followingStates: string[];
}

export class NESYoRHaIntegrationService {
  private config: NESYoRHaConfig;
  private componentRegistry: Map<string, ComponentRegistry> = new Map();
  private interactionHistory: UserInteractionPattern[] = [];
  private predictionModel: Map<string, string[]> = new Map();
  private syncWorker: Worker | null = null;
  private crtPostProcessor: THREE.ShaderPass | null = null;
  private isInitialized = false;

  constructor(config: Partial<NESYoRHaConfig> = {}) {
    this.config = {
      enableHybridSync: true,
      enableCRTEffects: true,
      enablePixelPerfect: true,
      cachePreloadThreshold: 0.7, // 70% confidence threshold
      maxHybridComponents: 50,
      renderMode: 'hybrid-sync',
      memoryBudget: 1024 * 1024 * 2, // 2MB budget for hybrid components
      ...config
    };

    console.log('🎮 NES + YoRHa Integration Service created');
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Initialize NES cache orchestrator
      await nesCacheOrchestrator.start();

      // Set up Web Worker for hybrid sync if enabled
      if (this.config.enableHybridSync && typeof Worker !== 'undefined') {
        await this.initializeSyncWorker();
      }

      // Set up interaction prediction system
      this.setupPredictionSystem();

      // Initialize CRT post-processing effects
      if (this.config.enableCRTEffects) {
        this.initializeCRTEffects();
      }

      this.isInitialized = true;
      console.log('✅ NES + YoRHa Integration Service initialized');
      
    } catch (error: any) {
      console.error('❌ Failed to initialize NES + YoRHa Integration Service:', error);
      throw error;
    }
  }

  // =============================================================================
  // COMPONENT REGISTRATION AND MANAGEMENT
  // =============================================================================

  async registerHybridComponent(
    id: string,
    style: NESYoRHaHybridStyle,
    domContainer?: HTMLElement
  ): Promise<NESYoRHaHybrid3D> {
    if (this.componentRegistry.size >= this.config.maxHybridComponents) {
      await this.evictOldestComponent();
    }

    // Create hybrid component with integrated styling
    const component = new NESYoRHaHybrid3D({
      ...style,
      renderMode: this.config.renderMode,
      pixelPerfect: this.config.enablePixelPerfect,
      crtEffect: this.config.enableCRTEffects
    });

    // Create DOM overlay if in hybrid mode
    let domElement: HTMLElement | undefined;
    if (this.config.renderMode === 'hybrid-sync' || this.config.renderMode === '2d-overlay') {
      domElement = await this.createDOMOverlay(id, style, domContainer);
    }

    // Register component
    const registry: ComponentRegistry = {
      id,
      component,
      domElement,
      lastInteraction: Date.now(),
      predictedStates: [],
      cacheKeys: []
    };

    this.componentRegistry.set(id, registry);

    // Set up component event listeners
    this.setupComponentListeners(component, registry);

    // Pre-cache component states
    await this.precacheComponentStates(id, component);

    // Add to sync worker if available
    if (this.syncWorker && domElement) {
      this.syncWorker.postMessage({
        type: 'REGISTER_HYBRID_COMPONENT',
        componentId: id,
        domElementId: domElement.id,
        position: component.position.toArray(),
        rotation: component.rotation.toArray(),
        scale: component.scale.toArray()
      });
    }

    if (dev) {
      console.log(`🔗 Registered hybrid component: ${id}`);
    }

    return component;
  }

  private async createDOMOverlay(
    id: string,
    style: NESYoRHaHybridStyle,
    container?: HTMLElement
  ): Promise<HTMLElement> {
    const domElement = document.createElement('div');
    domElement.id = `nes-yorha-${id}`;
    domElement.className = this.buildNESCSSClasses(style);
    
    // Add NES.css container styling
    if (style.nesContainer) {
      domElement.classList.add('nes-container', style.nesContainer);
    }

    // Create inner content based on component type
    if (style.nesButton) {
      const button = document.createElement('button');
      button.className = `nes-btn ${style.nesButton}`;
      button.textContent = style.variant || 'Button';
      domElement.appendChild(button);
    } else {
      const content = document.createElement('div');
      content.textContent = style.variant || 'Component';
      domElement.appendChild(content);
    }

    // Apply initial positioning
    domElement.style.position = 'absolute';
    domElement.style.pointerEvents = 'auto';
    domElement.style.zIndex = '1000';
    domElement.style.transform = 'translate(-50%, -50%)';

    // Append to container or body
    (container || document.body).appendChild(domElement);

    // Set up DOM event listeners
    this.setupDOMEventListeners(domElement, id);

    return domElement;
  }

  private buildNESCSSClasses(style: NESYoRHaHybridStyle): string {
    const classes: string[] = [];

    // Add base NES classes
    if (style.nesCssClass) {
      classes.push(style.nesCssClass);
    }

    // Add variant-based classes
    switch (style.variant) {
      case 'primary':
        classes.push('nes-text', 'is-primary');
        break;
      case 'success':
        classes.push('nes-text', 'is-success');
        break;
      case 'warning':
        classes.push('nes-text', 'is-warning');
        break;
      case 'danger':
        classes.push('nes-text', 'is-error');
        break;
      default:
        classes.push('nes-text');
    }

    // Add pixel perfect class if enabled
    if (style.pixelPerfect) {
      classes.push('pixel-perfect');
    }

    return classes.join(' ');
  }

  // =============================================================================
  // EVENT HANDLING AND INTERACTION TRACKING
  // =============================================================================

  private setupComponentListeners(component: NESYoRHaHybrid3D, registry: ComponentRegistry): void {
    // Track 3D component interactions
    component.addEventListener('click', () => {
      this.recordInteraction(registry.id, 'click', { source: '3d' });
    });

    component.addEventListener('hover', () => {
      this.recordInteraction(registry.id, 'hover', { source: '3d' });
    });

    // Update sync position
    component.addCustomAnimation('domSync', (deltaTime) => {
      if (this.config.enableHybridSync && registry.domElement) {
        this.syncDOMPosition(component, registry.domElement);
      }
    });
  }

  private setupDOMEventListeners(domElement: HTMLElement, componentId: string): void {
    domElement.addEventListener('click', (event: any) => {
      this.recordInteraction(componentId, 'click', { 
        source: 'dom',
        position: { x: event.clientX, y: event.clientY }
      });
      event.preventDefault();
    });

    domElement.addEventListener('mouseenter', () => {
      this.recordInteraction(componentId, 'hover', { source: 'dom' });
    });

    domElement.addEventListener('focus', () => {
      this.recordInteraction(componentId, 'focus', { source: 'dom' });
    });
  }

  private recordInteraction(
    componentId: string, 
    type: UserInteractionPattern['interactionType'],
    context: Record<string, any>
  ): void {
    const registry = this.componentRegistry.get(componentId);
    if (!registry) return;

    registry.lastInteraction = Date.now();

    // Record interaction pattern
    const interaction: UserInteractionPattern = {
      componentId,
      interactionType: type,
      timestamp: Date.now(),
      context,
      followingStates: []
    };

    this.interactionHistory.push(interaction);

    // Keep history manageable
    if (this.interactionHistory.length > 1000) {
      this.interactionHistory.splice(0, 100);
    }

    // Trigger prediction update
    this.updatePredictions(componentId);

    if (dev) {
      console.log(`🎯 Recorded ${type} interaction for ${componentId}`);
    }
  }

  // =============================================================================
  // DOM-3D SYNCHRONIZATION
  // =============================================================================

  private syncDOMPosition(component: NESYoRHaHybrid3D, domElement: HTMLElement): void {
    // Get 3D world position
    const worldPosition = new THREE.Vector3();
    component.getWorldPosition(worldPosition);

    // Project to screen coordinates (simplified)
    const screenX = (worldPosition.x + 4) / 8 * window.innerWidth;
    const screenY = (-worldPosition.y + 3) / 6 * window.innerHeight;

    // Update DOM position
    domElement.style.left = `${screenX}px`;
    domElement.style.top = `${screenY}px`;

    // Sync opacity and visibility
    const distance = worldPosition.distanceTo(new THREE.Vector3(0, 0, 5));
    const opacity = Math.max(0.1, 1 - distance / 10);
    domElement.style.opacity = opacity.toString();

    // Apply CRT distortion if enabled
    if (this.config.enableCRTEffects) {
      this.applyCRTDistortion(domElement, screenX, screenY);
    }
  }

  private applyCRTDistortion(element: HTMLElement, x: number, y: number): void {
    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight / 2;
    
    const dx = (x - centerX) / centerX;
    const dy = (y - centerY) / centerY;
    
    const distortion = (dx * dx + dy * dy) * 0.1;
    const scaleX = 1 + distortion;
    const scaleY = 1 + distortion * 0.8;
    
    element.style.transform = `translate(-50%, -50%) scale(${scaleX}, ${scaleY})`;
  }

  // =============================================================================
  // PREDICTIVE CACHING SYSTEM
  // =============================================================================

  private setupPredictionSystem(): void {
    // Update predictions periodically
    setInterval(() => {
      this.updateAllPredictions();
    }, 2000);
  }

  private updatePredictions(componentId: string): void {
    const patterns = this.findSimilarPatterns(componentId);
    const predictions = this.generatePredictions(patterns);
    
    const registry = this.componentRegistry.get(componentId);
    if (registry) {
      registry.predictedStates = predictions;
      
      // Preload predicted states if confidence is high
      predictions.forEach(async (stateId, index) => {
        if (index < 3) { // Only preload top 3 predictions
          await this.preloadComponentState(componentId, stateId);
        }
      });
    }
  }

  private updateAllPredictions(): void {
    for (const componentId of this.componentRegistry.keys()) {
      this.updatePredictions(componentId);
    }
  }

  private findSimilarPatterns(componentId: string): UserInteractionPattern[] {
    const recentInteractions = this.interactionHistory
      .filter(i => i.componentId === componentId)
      .slice(-10);

    if (recentInteractions.length < 2) return [];

    // Find similar interaction sequences
    const currentSequence = recentInteractions
      .map(i => `${i.interactionType}_${i.context.source || 'unknown'}`)
      .join('|');

    return this.interactionHistory
      .filter(i => {
        // Create sequence for this interaction and following ones
        const index = this.interactionHistory.indexOf(i);
        const sequence = this.interactionHistory
          .slice(index, index + recentInteractions.length)
          .map(ii => `${ii.interactionType}_${ii.context.source || 'unknown'}`)
          .join('|');
        
        return sequence === currentSequence;
      });
  }

  private generatePredictions(patterns: UserInteractionPattern[]): string[] {
    if (patterns.length === 0) return [];

    // Simple prediction based on most common following states
    const followingStates = patterns
      .flatMap(p => p.followingStates)
      .filter(Boolean);

    const stateFrequency = new Map<string, number>();
    followingStates.forEach(state => {
      stateFrequency.set(state, (stateFrequency.get(state) || 0) + 1);
    });

    return Array.from(stateFrequency.entries())
      .sort(([, a], [, b]) => b - a)
      .map(([state]) => state)
      .slice(0, 5);
  }

  private async preloadComponentState(componentId: string, stateId: string): Promise<void> {
    const cacheKey = `hybrid_${componentId}_${stateId}`;
    
    try {
      await nesCacheOrchestrator.loadSpriteSheet(cacheKey);
      if (dev) {
        console.log(`📋 Preloaded state ${stateId} for ${componentId}`);
      }
    } catch (error: any) {
      if (dev) {
        console.warn(`⚠️ Failed to preload state ${stateId}:`, error);
      }
    }
  }

  // =============================================================================
  // CACHING INTEGRATION
  // =============================================================================

  private async precacheComponentStates(id: string, component: NESYoRHaHybrid3D): Promise<void> {
    // Generate common state variations
    const variants = ['default', 'hover', 'active', 'disabled'];
    const canvasStates: CanvasState[] = [];

    for (const variant of variants) {
      const stateId = `hybrid_${id}_${variant}`;
      const canvasState: CanvasState = {
        id: stateId,
        animation: 'hybrid_component',
        frame: variants.indexOf(variant),
        fabricJSON: this.generateVariantJSON(component, variant),
        metadata: {
          componentId: id,
          variant,
          renderMode: this.config.renderMode,
          cacheRegion: 'CHR_ROM'
        }
      };

      canvasStates.push(canvasState);
    }

    // Cache states using NES orchestrator
    const cacheKey = await nesCacheOrchestrator.cacheCanvasStateAsSprite(
      `component_${id}`,
      canvasStates,
      {
        priority: 2,
        compression: true
      }
    );

    // Store cache keys in registry
    const registry = this.componentRegistry.get(id);
    if (registry) {
      registry.cacheKeys.push(cacheKey);
    }
  }

  private generateVariantJSON(component: NESYoRHaHybrid3D, variant: string): object {
    const colorMap = {
      default: NES_YORHA_PALETTE.yorhaBeige,
      hover: NES_YORHA_PALETTE.hybridAccent,
      active: NES_YORHA_PALETTE.nesSuccess,
      disabled: NES_YORHA_PALETTE.nesGray
    };

    return {
      version: '5.3.0',
      objects: [{
        type: 'nes-yorha-hybrid',
        left: component.position.x * 100,
        top: component.position.y * 100,
        width: (component.scale.x * 200),
        height: (component.scale.y * 100),
        fill: `#${(colorMap[variant as keyof typeof colorMap] || NES_YORHA_PALETTE.yorhaBeige).toString(16)}`,
        nesVariant: variant,
        yorhaStyle: component.getStyle()
      }]
    };
  }

  // =============================================================================
  // WEB WORKER INTEGRATION
  // =============================================================================

  private async initializeSyncWorker(): Promise<void> {
    try {
      const workerCode = this.generateSyncWorkerCode();
      const blob = new Blob([workerCode], { type: 'application/javascript' });
      const workerUrl = URL.createObjectURL(blob);
      
      this.syncWorker = new Worker(workerUrl);
      this.setupWorkerListeners();
      
      console.log('🔧 Sync worker initialized');
    } catch (error: any) {
      console.warn('⚠️ Failed to initialize sync worker:', error);
    }
  }

  private generateSyncWorkerCode(): string {
    return `
      class HybridSyncWorker {
        constructor() {
          this.components = new Map();
          this.syncInterval = null;
        }

        registerComponent(componentId, domElementId, position, rotation, scale) {
          this.components.set(componentId, {
            domElementId,
            position,
            rotation,
            scale,
            lastUpdate: Date.now()
          });

          if (!this.syncInterval) {
            this.startSyncLoop();
          }
        }

        startSyncLoop() {
          this.syncInterval = setInterval(() => {
            const updates = [];
            for (const [id, data] of this.components) {
              updates.push({ id, ...data });
            }
            
            if (updates.length > 0) {
              self.postMessage({
                type: 'SYNC_UPDATE',
                updates
              });
            }
          }, 16); // ~60fps
        }

        updateComponent(componentId, position, rotation, scale) {
          const component = this.components.get(componentId);
          if (component) {
            component.position = position;
            component.rotation = rotation;
            component.scale = scale;
            component.lastUpdate = Date.now();
          }
        }
      }

      const syncWorker = new HybridSyncWorker();

      self.addEventListener('message', (event: any) => {
        const { type, componentId, domElementId, position, rotation, scale } = event.data;

        switch (type) {
          case 'REGISTER_HYBRID_COMPONENT':
            syncWorker.registerComponent(componentId, domElementId, position, rotation, scale);
            break;
          case 'UPDATE_COMPONENT':
            syncWorker.updateComponent(componentId, position, rotation, scale);
            break;
        }
      });
    `;
  }

  private setupWorkerListeners(): void {
    if (!this.syncWorker) return;

    this.syncWorker.addEventListener('message', (event: any) => {
      const { type, updates } = event.data;

      if (type === 'SYNC_UPDATE') {
        updates.forEach(({ id, position, rotation, scale }: any) => {
          const registry = this.componentRegistry.get(id);
          if (registry?.domElement) {
            // Apply sync updates to DOM element
            this.applySyncUpdate(registry.domElement, position, rotation, scale);
          }
        });
      }
    });
  }

  private applySyncUpdate(
    element: HTMLElement, 
    position: [number, number, number], 
    rotation: [number, number, number], 
    scale: [number, number, number]
  ): void {
    const [x, y] = position;
    const [, , rz] = rotation;
    const [sx, sy] = scale;

    const screenX = (x + 4) / 8 * window.innerWidth;
    const screenY = (-y + 3) / 6 * window.innerHeight;

    element.style.left = `${screenX}px`;
    element.style.top = `${screenY}px`;
    element.style.transform = `translate(-50%, -50%) rotate(${rz}rad) scale(${sx}, ${sy})`;
  }

  // =============================================================================
  // CRT EFFECTS INITIALIZATION
  // =============================================================================

  private initializeCRTEffects(): void {
    // Add global CRT CSS effects
    if (typeof document !== 'undefined') {
      const style = document.createElement('style');
      style.textContent = this.generateCRTCSS();
      document.head.appendChild(style);
    }
  }

  private generateCRTCSS(): string {
    return `
      .pixel-perfect {
        image-rendering: pixelated;
        image-rendering: -moz-crisp-edges;
        image-rendering: crisp-edges;
      }

      .nes-yorha-hybrid-crt {
        position: relative;
        overflow: hidden;
      }

      .nes-yorha-hybrid-crt::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
          0deg,
          transparent 0px,
          transparent 2px,
          rgba(0, 255, 0, 0.1) 2px,
          rgba(0, 255, 0, 0.1) 4px
        );
        pointer-events: none;
        z-index: 1000;
        animation: crtFlicker 0.15s infinite linear alternate;
      }

      @keyframes crtFlicker {
        0% { opacity: 1; }
        100% { opacity: 0.98; }
      }
    `;
  }

  // =============================================================================
  // CLEANUP AND UTILITY METHODS
  // =============================================================================

  private async evictOldestComponent(): Promise<void> {
    let oldestTime = Date.now();
    let oldestId = '';

    for (const [id, registry] of this.componentRegistry) {
      if (registry.lastInteraction < oldestTime) {
        oldestTime = registry.lastInteraction;
        oldestId = id;
      }
    }

    if (oldestId) {
      await this.unregisterComponent(oldestId);
      console.log(`🗑️ Evicted oldest component: ${oldestId}`);
    }
  }

  async unregisterComponent(id: string): Promise<void> {
    const registry = this.componentRegistry.get(id);
    if (!registry) return;

    // Clean up DOM element
    if (registry.domElement && registry.domElement.parentNode) {
      registry.domElement.parentNode.removeChild(registry.domElement);
    }

    // Dispose 3D component
    registry.component.dispose();

    // Clear cache entries
    for (const cacheKey of registry.cacheKeys) {
      await nesCacheOrchestrator.clearRegion('CHR_ROM');
    }

    this.componentRegistry.delete(id);
    console.log(`🗑️ Unregistered hybrid component: ${id}`);
  }

  getStats() {
    return {
      registeredComponents: this.componentRegistry.size,
      interactionHistory: this.interactionHistory.length,
      cacheStats: nesCacheOrchestrator.getMemoryStats(),
      config: this.config,
      predictions: Array.from(this.componentRegistry.values()).reduce((total, reg) => 
        total + reg.predictedStates.length, 0)
    };
  }

  async shutdown(): Promise<void> {
    // Clean up all components
    for (const id of this.componentRegistry.keys()) {
      await this.unregisterComponent(id);
    }

    // Terminate sync worker
    if (this.syncWorker) {
      this.syncWorker.terminate();
    }

    // Shutdown cache orchestrator
    await nesCacheOrchestrator.shutdown();

    this.isInitialized = false;
    console.log('🎮 NES + YoRHa Integration Service shut down');
  }
}

// Export singleton instance
export const nesYorhaIntegration = new NESYoRHaIntegrationService();