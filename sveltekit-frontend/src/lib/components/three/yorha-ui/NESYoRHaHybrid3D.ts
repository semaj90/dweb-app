/**
 * NES.css + YoRHa 3D Hybrid Component System
 * Revolutionary fusion of 8-bit retro styling with advanced 3D GPU interfaces
 *
 * Features:
 * - NES.css styling for DOM overlay elements
 * - YoRHa 3D components for immersive 3D interfaces
 * - NES-style memory caching for instant state switching
 * - Canvas-to-DOM synchronization for hybrid layouts
 */

import * as THREE from 'three';
import { YoRHa3DComponent, YORHA_COLORS, type YoRHaStyle } from './YoRHaUI3D';
import { nesCacheOrchestrator } from '$lib/services/nes-cache-orchestrator';
import type { InteractiveCanvasState } from '$lib/types/canvas';

// NES + YoRHa Color Palette Fusion
export const NES_YORHA_PALETTE = {
  // NES 8-bit colors mapped to YoRHa aesthetic
  nesBlack: 0x0f0f0f,        // Pure NES black
  nesWhite: 0xfcfcfc,        // NES white
  nesGray: 0x7c7c7c,         // NES gray
  nesLightGray: 0xbcbcbc,    // NES light gray

  // YoRHa colors with NES pixel-perfect quantization
  yorhaGold: YORHA_COLORS.accent.gold,
  yorhaBeige: YORHA_COLORS.primary.beige,
  yorhaBlack: YORHA_COLORS.primary.black,

  // Hybrid colors (quantized YoRHa palette)
  hybridAccent: 0xd4af00,    // Quantized gold
  hybridBackground: 0xd4c500, // Quantized beige
  hybridBorder: 0x0a0a00,    // Quantized black

  // NES-style status colors
  nesSuccess: 0x00d800,      // Bright green
  nesWarning: 0xfc9838,      // Orange
  nesError: 0xf83800,        // Red
  nesInfo: 0x3cbcfc,         // Blue
} as const;

export interface NESYoRHaHybridStyle extends YoRHaStyle {
  // NES.css integration
  nesCssClass?: string;
  nesContainer?: 'with-title' | 'is-rounded' | 'is-dark' | 'is-centered';
  nesButton?: 'is-primary' | 'is-success' | 'is-warning' | 'is-error' | 'is-disabled';

  // Hybrid rendering mode
  renderMode?: '2d-overlay' | '3d-embedded' | 'hybrid-sync';
  domOverlay?: HTMLElement;

  // NES-style pixelation
  pixelPerfect?: boolean;
  pixelScale?: number;

  // Retro effects
  crtEffect?: boolean;
  scanlines?: boolean;
  ghosting?: boolean;

  // Animation style
  animationStyle?: 'nes-8bit' | 'yorha-smooth' | 'hybrid-morphing';
}

export interface DOMSyncData {
  domElement: HTMLElement;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  scale: THREE.Vector3;
  opacity: number;
  nesCssClasses: string[];
  syncFrequency: number;
}

export class NESYoRHaHybrid3D extends YoRHa3DComponent {
  protected hybridStyle: NESYoRHaHybridStyle;
  protected domOverlay: HTMLElement | null = null;
  protected domSyncData: DOMSyncData | null = null;
  protected pixelCanvas: HTMLCanvasElement | null = null;
  protected crtShader: THREE.ShaderMaterial | null = null;
  protected nesStateCache: Map<string, InteractiveCanvasState> = new Map();
  protected syncAnimationFrame: number | null = null;

  constructor(hybridStyle: NESYoRHaHybridStyle = {}) {
    // Merge NES + YoRHa default styles
    const mergedStyle = {
      backgroundColor: NES_YORHA_PALETTE.yorhaBeige,
      borderColor: NES_YORHA_PALETTE.nesBlack,
      borderWidth: 0.1, // Thicker borders for NES aesthetic
      borderRadius: 0,   // Sharp corners like NES
      pixelPerfect: true,
      renderMode: 'hybrid-sync',
      animationStyle: 'hybrid-morphing',
      ...hybridStyle
    } as NESYoRHaHybridStyle;

    super(mergedStyle);
    this.hybridStyle = mergedStyle;

    this.initializeHybridSystem();
    this.setupNESCaching();
    this.createDOMOverlay();
  }

  protected createGeometry(): void {
    // Create pixelated geometry for NES aesthetic
    const width = this.hybridStyle.width || 2;
    const height = this.hybridStyle.height || 1;
    const depth = this.hybridStyle.depth || 0.1;

    if (this.hybridStyle.pixelPerfect) {
      // Create low-poly geometry with sharp edges
      this.geometry = new THREE.BoxGeometry(width, height, depth, 1, 1, 1);

      // Modify vertices for pixelated effect
      this.pixelateGeometry();
    } else {
      this.geometry = new THREE.BoxGeometry(width, height, depth);
    }
  }

  protected createMaterial(): void {
    const materialProps: THREE.MeshStandardMaterialParameters = {
      color: this.hybridStyle.backgroundColor,
      opacity: this.hybridStyle.opacity || 1,
      transparent: (this.hybridStyle.opacity || 1) < 1,
      metalness: 0, // No metallic for 8-bit look
      roughness: 1, // Completely rough for matte finish
    };

    // Apply NES-style material enhancements
    if (this.hybridStyle.crtEffect) {
      this.material = this.createCRTMaterial(materialProps);
    } else {
      this.material = new THREE.MeshBasicMaterial(materialProps); // Basic material for flat shading
    }

    // Apply scanlines if requested
    if (this.hybridStyle.scanlines) {
      this.addScanlineEffect();
    }
  }

  private createCRTMaterial(baseProps: THREE.MeshStandardMaterialParameters): THREE.ShaderMaterial {
    this.crtShader = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        resolution: { value: new THREE.Vector2(800, 600) },
        baseColor: { value: new THREE.Color(baseProps.color) },
        scanlineIntensity: { value: 0.8 },
        curvature: { value: 2.0 },
        brightness: { value: 1.2 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec2 resolution;
        uniform vec3 baseColor;
        uniform float scanlineIntensity;
        uniform float curvature;
        uniform float brightness;
        varying vec2 vUv;

        // CRT distortion effect
        vec2 crtDistort(vec2 uv) {
          vec2 cc = uv - 0.5;
          float dist = dot(cc, cc) * curvature;
          return uv + cc * (1.0 + dist) * dist;
        }

        void main() {
          vec2 distortedUV = crtDistort(vUv);

          // Out of bounds check for CRT distortion
          if (distortedUV.x < 0.0 || distortedUV.x > 1.0 || distortedUV.y < 0.0 || distortedUV.y > 1.0) {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
          }

          // Scanlines effect
          float scanline = sin(distortedUV.y * resolution.y * 2.0) * scanlineIntensity;

          // RGB separation for CRT effect
          float r = baseColor.r + sin(time + distortedUV.x * 10.0) * 0.02;
          float g = baseColor.g + sin(time + distortedUV.x * 10.0 + 2.094) * 0.02;
          float b = baseColor.b + sin(time + distortedUV.x * 10.0 + 4.188) * 0.02;

          vec3 color = vec3(r, g, b) * brightness;
          color *= (1.0 - scanline * 0.3);

          // Add slight noise
          float noise = fract(sin(dot(distortedUV + time * 0.001, vec2(12.9898, 78.233))) * 43758.5453);
          color += noise * 0.05;

          gl_FragColor = vec4(color, 1.0);
        }
      `
    });

    return this.crtShader;
  }

  private addScanlineEffect(): void {
    // Add scanline post-processing effect
    const scanlineGeometry = new THREE.PlaneGeometry(
      (this.hybridStyle.width || 2) * 1.1,
      (this.hybridStyle.height || 1) * 1.1
    );

    const scanlineMaterial = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: {
        time: { value: 0 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        varying vec2 vUv;

        void main() {
          float scanlines = sin(vUv.y * 100.0 + time * 2.0) * 0.04;
          gl_FragColor = vec4(0.0, 0.0, 0.0, scanlines);
        }
      `
    });

    const scanlineMesh = new THREE.Mesh(scanlineGeometry, scanlineMaterial);
    scanlineMesh.position.z = 0.001; // Slightly in front
    this.add(scanlineMesh);

    // Animate scanlines
    this.addCustomAnimation('scanlines', (deltaTime) => {
      (scanlineMaterial.uniforms.time as any).value += deltaTime;
    });
  }

  private pixelateGeometry(): void {
    if (!this.geometry) return;

    // Apply vertex quantization for pixelated look
    const positions = this.geometry.attributes.position;
    const pixelSize = this.hybridStyle.pixelScale || 0.1;

    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);

      // Quantize positions to pixel grid
      positions.setX(i, Math.round(x / pixelSize) * pixelSize);
      positions.setY(i, Math.round(y / pixelSize) * pixelSize);
      positions.setZ(i, Math.round(z / pixelSize) * pixelSize);
    }

    positions.needsUpdate = true;
    this.geometry.computeVertexNormals();
  }

  private async initializeHybridSystem(): Promise<void> {
    // Initialize NES-style state caching
    await nesCacheOrchestrator.start();

    // Set up hybrid rendering pipeline
    this.setupHybridRendering();

    console.log('🎮 NES + YoRHa Hybrid 3D Component initialized');
  }

  private setupHybridRendering(): void {
    switch (this.hybridStyle.renderMode) {
      case '2d-overlay':
        this.setupDOMOverlay();
        break;
      case '3d-embedded':
        this.setupEmbedded3D();
        break;
      case 'hybrid-sync':
        this.setupHybridSync();
        break;
    }
  }

  private setupDOMOverlay(): void {
    // Create DOM elements that overlay the 3D scene
    if (typeof window !== 'undefined') {
      this.domOverlay = document.createElement('div');
      this.domOverlay.className = `nes-container ${this.hybridStyle.nesContainer || 'with-title'}`;

      // Add NES.css button if specified
      if (this.hybridStyle.nesButton) {
        const button = document.createElement('button');
        button.className = `nes-btn ${this.hybridStyle.nesButton}`;
        button.textContent = '3D Component';
        this.domOverlay.appendChild(button);
      }

      // Position overlay to match 3D position
      this.syncDOMPosition();
    }
  }

  private setupEmbedded3D(): void {
    // Embed 3D content directly in DOM layout
    this.scale.multiplyScalar(0.5); // Smaller scale for embedding
    this.position.z = -0.5; // Pull back from camera
  }

  private setupHybridSync(): void {
    // Synchronize DOM and 3D elements
    this.setupDOMOverlay();
    this.startDOMSyncLoop();
  }

  private createDOMOverlay(): void {
    if (this.hybridStyle.renderMode === '2d-overlay' || this.hybridStyle.renderMode === 'hybrid-sync') {
      this.setupDOMOverlay();
    }
  }

  private setupNESCaching(): void {
    // Cache component states in NES-style memory regions
    this.cacheCurrentState();

    // Set up predictive caching
    this.setupPredictiveCaching();
  }

  private async cacheCurrentState(): Promise<void> {
    const stateId = `hybrid_${this.hybridStyle.variant || 'default'}_${Date.now()}`;

    const canvasState: InteractiveCanvasState = {
      id: stateId,
      nodes: [],
      connections: [],
      viewport: { x: 0, y: 0, zoom: 1 },
      animation: 'hybrid_component',
      frame: 0,
      fabricJSON: this.serializeToFabricJSON(),
      metadata: {
        renderMode: this.hybridStyle.renderMode,
        nesCssClass: this.hybridStyle.nesCssClass,
        yorhaVariant: this.hybridStyle.variant,
        cacheRegion: 'CHR_ROM'
      }
    };

    // Cache using NES orchestrator
    await nesCacheOrchestrator.cacheCanvasStateAsSprite(
      'hybrid_component',
      [canvasState],
      {
        priority: 2,
        compression: true
      }
    );

    this.nesStateCache.set(stateId, canvasState);
  }

  private serializeToFabricJSON(): string {
    // Convert Three.js component to Fabric.js-compatible format
    const fabricData = {
      version: '5.3.0',
      objects: [{
        type: 'yorha-3d-component',
        left: this.position.x * 100,
        top: this.position.y * 100,
        width: (this.hybridStyle.width || 2) * 100,
        height: (this.hybridStyle.height || 1) * 100,
        fill: `#${this.hybridStyle.backgroundColor?.toString(16) || 'd4c5a9'}`,
        stroke: `#${this.hybridStyle.borderColor?.toString(16) || '0a0a0a'}`,
        strokeWidth: (this.hybridStyle.borderWidth || 0.1) * 100,
        nesStyle: {
          cssClass: this.hybridStyle.nesCssClass,
          container: this.hybridStyle.nesContainer,
          pixelPerfect: this.hybridStyle.pixelPerfect
        }
      }]
    };
    return JSON.stringify(fabricData);
  }

  private setupPredictiveCaching(): void {
    // Pre-cache likely next states based on user interaction patterns
    const likelyVariants = ['primary', 'secondary', 'accent', 'hover', 'active'];

    likelyVariants.forEach(async (variant) => {
      const predictiveState: InteractiveCanvasState = {
        id: `hybrid_${variant}_predicted`,
        nodes: [],
        connections: [],
        viewport: { x: 0, y: 0, zoom: 1 },
        animation: 'hybrid_component',
        frame: 0,
        fabricJSON: JSON.stringify(this.generateVariantFabricJSON(variant)),
        metadata: {
          renderMode: this.hybridStyle.renderMode,
          predictive: true,
          variant
        }
      };

      this.nesStateCache.set(predictiveState.id, predictiveState);
    });
  }

  private generateVariantFabricJSON(variant: string): object {
    const colorMap = {
      primary: NES_YORHA_PALETTE.yorhaGold,
      secondary: NES_YORHA_PALETTE.nesGray,
      accent: NES_YORHA_PALETTE.hybridAccent,
      hover: NES_YORHA_PALETTE.nesLightGray,
      active: NES_YORHA_PALETTE.nesSuccess
    };

    const baseJSON = JSON.parse(this.serializeToFabricJSON());
    if (baseJSON.objects?.[0]) {
      baseJSON.objects[0].fill = `#${(colorMap[variant as keyof typeof colorMap] || NES_YORHA_PALETTE.yorhaBeige).toString(16)}`;
    }

    return baseJSON;
  }

  private syncDOMPosition(): void {
    if (!this.domOverlay) return;

    // Convert 3D world position to screen coordinates
    const vector = this.position.clone();
    vector.project(this.getCamera());

    const x = (vector.x * 0.5 + 0.5) * window.innerWidth;
    const y = (vector.y * -0.5 + 0.5) * window.innerHeight;

    this.domOverlay.style.position = 'fixed';
    this.domOverlay.style.left = `${x}px`;
    this.domOverlay.style.top = `${y}px`;
    this.domOverlay.style.transform = 'translate(-50%, -50%)';
    this.domOverlay.style.pointerEvents = 'auto';
    this.domOverlay.style.zIndex = '1000';
  }

  private getCamera(): THREE.Camera {
    // Find camera in scene hierarchy (simplified)
    let current = this.parent;
    while (current && !(current instanceof THREE.Scene)) {
      current = current.parent;
    }

    // Return a default camera if not found
    return new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  }

  private startDOMSyncLoop(): void {
    const syncLoop = () => {
      if (this.domOverlay && this.hybridStyle.renderMode === 'hybrid-sync') {
        this.syncDOMPosition();
        this.syncAnimationFrame = requestAnimationFrame(syncLoop);
      }
    };

    syncLoop();
  }

  // =============================================================================
  // NES-STYLE ANIMATION METHODS
  // =============================================================================

  public async switchToNESState(stateId: string): Promise<void> {
    // Instant state switching like NES sprite frames
    const cachedState = await nesCacheOrchestrator.loadSpriteSheet('hybrid_component');

    if (cachedState) {
      const state = cachedState.find(s => s.id && s.id === stateId);
      if (state) {
        this.applyNESState(state);
        console.log(`🎮 Switched to NES state: ${stateId}`);
      }
    }
  }

  private applyNESState(state: InteractiveCanvasState): void {
    if (!state.fabricJSON) return;
    const fabricData = JSON.parse(state.fabricJSON);
    if (fabricData.objects?.[0]) {
      const obj = fabricData.objects[0];

      // Apply position
      this.position.set(obj.left / 100, obj.top / 100, this.position.z);

      // Apply color
      if (obj.fill && this.mesh.material instanceof THREE.MeshBasicMaterial) {
        this.mesh.material.color.setHex(parseInt(obj.fill.replace('#', ''), 16));
      }

      // Apply scale
      this.scale.set(obj.width / 200, obj.height / 100, 1);

      // Update DOM overlay if it exists
      if (this.domOverlay && obj.nesStyle?.cssClass) {
        this.domOverlay.className = `nes-container ${obj.nesStyle.cssClass}`;
      }
    }
  }

  public createNESAnimation(frames: string[], duration: number = 100): void {
    // Create NES-style frame-based animation
    let currentFrame = 0;

    const frameLoop = () => {
      const frameId = frames[currentFrame];
      this.switchToNESState(frameId);

      currentFrame = (currentFrame + 1) % frames.length;

      setTimeout(frameLoop, duration);
    };

    frameLoop();
  }

  // =============================================================================
  // HYBRID INTERACTION METHODS
  // =============================================================================

  protected onHover(): void {
    super.onHover();

    // Add NES-style hover effects
    if (this.domOverlay) {
      this.domOverlay.style.transform = 'translate(-50%, -50%) scale(1.05)';
      this.domOverlay.style.filter = 'brightness(1.2)';
    }

    // Switch to hover state using NES caching
    this.switchToNESState(`hybrid_hover_${Date.now()}`);
  }

  protected onUnhover(): void {
    super.onUnhover();

    if (this.domOverlay) {
      this.domOverlay.style.transform = 'translate(-50%, -50%) scale(1.0)';
      this.domOverlay.style.filter = 'brightness(1.0)';
    }
  }

  protected onClick(): void {
    super.onClick();

    // NES-style click animation
    this.playNESClickAnimation();

    // Update DOM overlay
    if (this.domOverlay) {
      this.domOverlay.style.animation = 'nesClick 0.2s ease-in-out';
    }
  }

  private playNESClickAnimation(): void {
    // Quick scale animation like NES button press
    const originalScale = this.scale.clone();

    this.scale.multiplyScalar(0.95);

    setTimeout(() => {
      this.scale.copy(originalScale);
    }, 100);
  }

  // =============================================================================
  // CLEANUP AND DISPOSAL
  // =============================================================================

  public dispose(): void {
    super.dispose();

    // Clean up DOM overlay
    if (this.domOverlay && this.domOverlay.parentNode) {
      this.domOverlay.parentNode.removeChild(this.domOverlay);
    }

    // Cancel sync loop
    if (this.syncAnimationFrame) {
      cancelAnimationFrame(this.syncAnimationFrame);
    }

    // Clean up NES cache
    this.nesStateCache.clear();

    // Clean up pixel canvas
    if (this.pixelCanvas) {
      this.pixelCanvas.remove();
    }

    console.log('🎮 NES + YoRHa Hybrid component disposed');
  }
}

// =============================================================================
// FACTORY FUNCTIONS FOR COMMON COMPONENTS
// =============================================================================

export function createNESButton(options: {
  text: string;
  variant?: 'is-primary' | 'is-success' | 'is-warning' | 'is-error';
  size?: 'small' | 'normal' | 'large';
  onClick?: () => void;
}): NESYoRHaHybrid3D {
  return new NESYoRHaHybrid3D({
    width: options.size === 'large' ? 3 : options.size === 'small' ? 1.5 : 2,
    height: options.size === 'large' ? 1 : options.size === 'small' ? 0.5 : 0.8,
    nesButton: options.variant || 'is-primary',
    variant: 'default',
    renderMode: 'hybrid-sync',
    pixelPerfect: true,
    crtEffect: true,
    scanlines: true,
    animationStyle: 'hybrid-morphing'
  });
}

export function createNESContainer(options: {
  title?: string;
  rounded?: boolean;
  dark?: boolean;
  children?: NESYoRHaHybrid3D[];
}): NESYoRHaHybrid3D {
  return new NESYoRHaHybrid3D({
    width: 4,
    height: 3,
    nesContainer: options.rounded
      ? 'is-rounded'
      : options.dark
        ? 'is-dark'
        : 'with-title',
    variant: 'outlined',
    renderMode: '2d-overlay',
    backgroundColor: options.dark
      ? NES_YORHA_PALETTE.nesBlack
      : NES_YORHA_PALETTE.yorhaBeige,
    pixelPerfect: true
  });
}

export function createNESProgressBar(options: {
  value: number;
  max?: number;
  variant?: 'is-primary' | 'is-success' | 'is-warning' | 'is-error' | 'is-pattern';
}): NESYoRHaHybrid3D {
  return new NESYoRHaHybrid3D({
    width: 3,
    height: 0.3,
    variant: 'filled',
    renderMode: 'hybrid-sync',
    backgroundColor: NES_YORHA_PALETTE.yorhaGold,
    pixelPerfect: true,
    animation: {
      type: 'pulse',
      duration: 1000,
      loop: true
    }
  });
}