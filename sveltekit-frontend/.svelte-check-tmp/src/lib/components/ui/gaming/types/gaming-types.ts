/**
 * Gaming UI Component Types and Interfaces
 * Progressive evolution from 8-bit to N64-style 3D
 */

export type GamingEra = '8bit' | '16bit' | 'n64' | 'auto';

export interface NESColorPalette {
  // Core NES hardware colors
  black: string;
  white: string;
  darkGray: string;
  lightGray: string;
  red: string;
  blue: string;
  green: string;
  yellow: string;
  // Extended palette for UI components
  success: string;
  warning: string;
  error: string;
  info: string;
}

export interface SNESColorPalette extends NESColorPalette {
  // Enhanced 16-bit colors
  purple: string;
  orange: string;
  cyan: string;
  magenta: string;
  pink: string;
  lime: string;
  // Gradient support
  primaryGradient: string[];
  secondaryGradient: string[];
}

export interface N64RenderingOptions {
  // 3D rendering options
  enableAntiAliasing: boolean;
  enableTextureFiltering: boolean;
  enableMipMapping: boolean;
  polygonCount: 'low' | 'medium' | 'high' | 'ultra';
  // Fog and depth effects
  enableFog: boolean;
  fogColor: string;
  fogDensity: number;
  // Z-buffer and depth
  enableZBuffer: boolean;
  depthTesting: boolean;
  
  // Enhanced texture filtering options
  textureQuality?: 'draft' | 'standard' | 'high' | 'ultra';
  anisotropicLevel?: number; // 1, 2, 4, 8, 16
  enableBilinearFiltering?: boolean;
  enableTrilinearFiltering?: boolean;
  
  // Advanced filtering techniques
  enableSuperSampling?: boolean;
  superSampleRate?: number; // 1.25, 1.5, 2, 4
  enableTemporalFiltering?: boolean;
  enableAdaptiveSampling?: boolean;
  
  // Performance optimization
  lodBias?: number; // Level of detail bias (-1 to 1)
  maxMipLevel?: number;
  enableTextureCompression?: boolean;
}

export interface GamingComponentProps {
  // Common props for all gaming components
  era?: GamingEra;
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  size?: 'small' | 'medium' | 'large' | 'xl';
  disabled?: boolean;
  loading?: boolean;
  
  // Visual enhancements
  pixelPerfect?: boolean;
  enableScanlines?: boolean;
  enableCRTEffect?: boolean;
  enableGlitchEffect?: boolean;
  
  // Animation preferences
  animationStyle?: 'instant' | 'smooth' | 'retro-bounce' | 'glitch-transition';
  
  // 3D specific (N64 era)
  renderOptions?: N64RenderingOptions;
  
  // Event handlers
  onClick?: () => void;
  onHover?: () => void;
  onFocus?: () => void;
}

export interface ProgressiveGamingConfig {
  // Global configuration
  defaultEra: GamingEra;
  enableAutoEvolution: boolean;
  performanceThreshold: number; // Milliseconds before downgrading
  
  // Era-specific settings
  nesSettings: {
    strictPalette: boolean;
    enableScanlines: boolean;
    pixelScale: number;
  };
  
  snesSettings: {
    enableGradients: boolean;
    enableModeViitColors: boolean;
    layerCount: number;
  };
  
  n64Settings: N64RenderingOptions & {
    enableRealTimeReflections: boolean;
    textureQuality: 'draft' | 'standard' | 'high' | 'ultra';
  };
  
  // Integration settings
  yorhaIntegration: boolean;
  bitsUICompatibility: boolean;
}

export interface GamepadState {
  // NES controller mapping
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  a: boolean;
  b: boolean;
  start: boolean;
  select: boolean;
  
  // SNES additional buttons
  x?: boolean;
  y?: boolean;
  l?: boolean;
  r?: boolean;
  
  // N64 controller additions
  cUp?: boolean;
  cDown?: boolean;
  cLeft?: boolean;
  cRight?: boolean;
  z?: boolean;
  analogStick?: { x: number; y: number };
}

export interface GamingThemeState {
  currentEra: GamingEra;
  availableEras: GamingEra[];
  isTransitioning: boolean;
  transitionDuration: number;
  performanceLevel: 'low' | 'medium' | 'high';
}

export interface RetroEffect {
  name: string;
  enabled: boolean;
  intensity: number; // 0-1
  config: Record<string, any>;
}

export interface GamingAudioConfig {
  enableChiptune: boolean;
  sampleRate: '8khz' | '22khz' | '44khz';
  channels: 'mono' | 'stereo';
  enableReverb: boolean;
  
  // Era-specific audio settings
  nesChannels: 4; // PPU limitation
  snesChannels: 8; // SPC700 capability
  n64EnableSpatial: boolean;
}

// Component-specific types
export interface NESButtonConfig {
  pressDepth: number; // pixels
  pressAnimation: 'instant' | 'bounce';
  soundEffect?: string;
}

export interface SNESButtonConfig extends NESButtonConfig {
  enableGradient: boolean;
  layerEffects: string[];
}

export interface N64ButtonConfig extends SNESButtonConfig {
  enable3D: boolean;
  meshComplexity: 'low' | 'medium' | 'high';
  materialType: 'basic' | 'phong' | 'pbr';
  enableNormalMapping: boolean;
}