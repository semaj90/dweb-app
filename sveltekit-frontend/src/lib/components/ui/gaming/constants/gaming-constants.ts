/**
 * Gaming UI Constants and Color Palettes
 * Authentic retro gaming colors and presets
 */

import type { NESColorPalette, SNESColorPalette, N64RenderingOptions } from '../types/gaming-types.js';

// NES Color Palette (hardware accurate)
export const NES_COLOR_PALETTE: NESColorPalette = {
  // Core NES PPU colors
  black: '#0f0f0f',      // Darkest black available
  white: '#fcfcfc',      // Pure white
  darkGray: '#7c7c7c',   // 50% gray
  lightGray: '#bcbcbc',  // 75% gray
  red: '#f83800',        // Bright red
  blue: '#3cbcfc',       // Bright blue  
  green: '#92cc41',      // Bright green
  yellow: '#f7d51d',     // Bright yellow
  
  // Status colors (NES-compatible)
  success: '#92cc41',    // Green
  warning: '#f7d51d',    // Yellow
  error: '#f83800',      // Red
  info: '#3cbcfc'        // Blue
};

// SNES Color Palette (enhanced 16-bit)
export const SNES_COLOR_PALETTE: SNESColorPalette = {
  ...NES_COLOR_PALETTE,
  
  // Additional 16-bit colors
  purple: '#8b41fc',
  orange: '#fc9838', 
  cyan: '#38fcfc',
  magenta: '#fc38fc',
  pink: '#fc9cfc',
  lime: '#9cfc38',
  
  // Gradient definitions for SNES-style depth
  primaryGradient: ['#3cbcfc', '#0084ff', '#0050cc'],
  secondaryGradient: ['#f7d51d', '#cc8800', '#996600']
};

// N64 Texture and Rendering Presets (Enhanced)
export const N64_TEXTURE_PRESETS = {
  // Low-poly optimized settings
  lowPoly: {
    enableAntiAliasing: false,
    enableTextureFiltering: false,
    enableMipMapping: false,
    polygonCount: 'low' as const,
    enableFog: true,
    fogColor: '#404040',
    fogDensity: 0.1,
    enableZBuffer: true,
    depthTesting: true,
    // Enhanced filtering options
    textureQuality: 'draft' as const,
    anisotropicLevel: 1,
    enableBilinearFiltering: false,
    enableTrilinearFiltering: false
  } as N64RenderingOptions,
  
  // Balanced quality for most components
  balanced: {
    enableAntiAliasing: true,
    enableTextureFiltering: true,
    enableMipMapping: false,
    polygonCount: 'medium' as const,
    enableFog: true,
    fogColor: '#202020',
    fogDensity: 0.05,
    enableZBuffer: true,
    depthTesting: true,
    // Enhanced filtering options
    textureQuality: 'standard' as const,
    anisotropicLevel: 4,
    enableBilinearFiltering: true,
    enableTrilinearFiltering: false
  } as N64RenderingOptions,
  
  // High quality for hero components
  highQuality: {
    enableAntiAliasing: true,
    enableTextureFiltering: true,
    enableMipMapping: true,
    polygonCount: 'high' as const,
    enableFog: true,
    fogColor: '#101010',
    fogDensity: 0.02,
    enableZBuffer: true,
    depthTesting: true,
    // Enhanced filtering options
    textureQuality: 'ultra' as const,
    anisotropicLevel: 16,
    enableBilinearFiltering: true,
    enableTrilinearFiltering: true
  } as N64RenderingOptions,
  
  // New ultra-enhanced preset for modern hardware
  ultraEnhanced: {
    enableAntiAliasing: true,
    enableTextureFiltering: true,
    enableMipMapping: true,
    polygonCount: 'ultra' as const,
    enableFog: true,
    fogColor: '#000000',
    fogDensity: 0.01,
    enableZBuffer: true,
    depthTesting: true,
    // Advanced filtering options
    textureQuality: 'ultra' as const,
    anisotropicLevel: 16,
    enableBilinearFiltering: true,
    enableTrilinearFiltering: true,
    enableSuperSampling: true,
    superSampleRate: 2,
    enableTemporalFiltering: true,
    enableAdaptiveSampling: true
  } as N64RenderingOptions
};

// Responsive gaming breakpoints
export const GAMING_BREAKPOINTS = {
  // Screen sizes that impact gaming component rendering
  handheld: '(max-width: 320px)',     // Game Boy size
  nes: '(max-width: 480px)',          // NES TV resolution equivalent
  snes: '(max-width: 768px)',         // SNES enhanced resolution
  n64: '(min-width: 769px)',          // N64 and above
  
  // Performance-based breakpoints
  lowPerformance: '(max-device-memory: 2)',
  mediumPerformance: '(max-device-memory: 4)',
  highPerformance: '(min-device-memory: 5)'
};

// Gaming era specifications
export const GAMING_ERA_SPECS = {
  '8bit': {
    maxColors: 25,        // NES on-screen limit
    totalColors: 64,      // NES total palette
    resolution: { width: 256, height: 240 },
    refreshRate: 60,
    audioChannels: 4,
    memoryKB: 2,
    cpuMhz: 1.79
  },
  
  '16bit': {
    maxColors: 256,       // SNES enhanced
    totalColors: 32768,   // SNES total palette
    resolution: { width: 512, height: 448 },
    refreshRate: 60,
    audioChannels: 8,
    memoryKB: 128,
    cpuMhz: 3.58
  },
  
  'n64': {
    maxColors: 16777216,  // 24-bit color
    totalColors: 16777216,
    resolution: { width: 640, height: 480 },
    refreshRate: 60,
    audioChannels: 64,
    memoryMB: 4,
    cpuMhz: 93.75
  }
};

// CSS Custom Properties for theming
export const GAMING_CSS_VARS = {
  // NES Era
  '--nes-black': NES_COLOR_PALETTE.black,
  '--nes-white': NES_COLOR_PALETTE.white,
  '--nes-dark-gray': NES_COLOR_PALETTE.darkGray,
  '--nes-light-gray': NES_COLOR_PALETTE.lightGray,
  '--nes-red': NES_COLOR_PALETTE.red,
  '--nes-blue': NES_COLOR_PALETTE.blue,
  '--nes-green': NES_COLOR_PALETTE.green,
  '--nes-yellow': NES_COLOR_PALETTE.yellow,
  
  // SNES Era additions
  '--snes-purple': SNES_COLOR_PALETTE.purple,
  '--snes-orange': SNES_COLOR_PALETTE.orange,
  '--snes-cyan': SNES_COLOR_PALETTE.cyan,
  '--snes-magenta': SNES_COLOR_PALETTE.magenta,
  '--snes-pink': SNES_COLOR_PALETTE.pink,
  '--snes-lime': SNES_COLOR_PALETTE.lime,
  
  // Component sizing (pixel-perfect)
  '--gaming-unit': '8px',          // Base 8px grid
  '--nes-border-width': '2px',     // NES standard border
  '--snes-border-width': '1px',    // SNES refined border
  '--n64-border-radius': '4px',    // N64 rounded corners
  
  // Typography
  '--gaming-font-8bit': '"Press Start 2P", monospace',
  '--gaming-font-16bit': '"Orbitron", sans-serif',
  '--gaming-font-n64': '"Rajdhani", sans-serif',
  
  // Animation timings
  '--gaming-transition-instant': '0ms',
  '--gaming-transition-fast': '100ms',
  '--gaming-transition-normal': '200ms',
  '--gaming-transition-slow': '400ms'
};

// Component size mappings
export const GAMING_COMPONENT_SIZES = {
  small: {
    padding: 'calc(var(--gaming-unit) * 1)',
    fontSize: '10px',
    minHeight: 'calc(var(--gaming-unit) * 4)'
  },
  
  medium: {
    padding: 'calc(var(--gaming-unit) * 2)', 
    fontSize: '12px',
    minHeight: 'calc(var(--gaming-unit) * 6)'
  },
  
  large: {
    padding: 'calc(var(--gaming-unit) * 3)',
    fontSize: '14px', 
    minHeight: 'calc(var(--gaming-unit) * 8)'
  },
  
  xl: {
    padding: 'calc(var(--gaming-unit) * 4)',
    fontSize: '16px',
    minHeight: 'calc(var(--gaming-unit) * 10)'
  }
};

// Retro effect presets
export const RETRO_EFFECTS = {
  scanlines: {
    name: 'Scanlines',
    cssFilter: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.3) 2px, rgba(0,0,0,.3) 4px)',
    performance: 'low'
  },
  
  crt: {
    name: 'CRT Monitor',
    cssFilter: 'contrast(1.2) brightness(1.1) saturate(1.3)',
    borderRadius: '10px',
    boxShadow: 'inset 0 0 0 4px rgba(0,0,0,0.3)',
    performance: 'medium'
  },
  
  pixelate: {
    name: 'Pixel Perfect',
    imageRendering: 'pixelated',
    msInterpolationMode: 'nearest-neighbor',
    performance: 'low'
  },
  
  glitch: {
    name: 'Digital Glitch',
    animation: 'glitch 2s infinite',
    performance: 'high'
  }
};

// Gaming sound effect mappings
export const GAMING_SOUND_EFFECTS = {
  '8bit': {
    buttonPress: '/sounds/8bit/button-press.wav',
    menuMove: '/sounds/8bit/menu-move.wav',
    error: '/sounds/8bit/error.wav',
    success: '/sounds/8bit/success.wav'
  },
  
  '16bit': {
    buttonPress: '/sounds/16bit/button-press.wav', 
    menuMove: '/sounds/16bit/menu-move.wav',
    error: '/sounds/16bit/error.wav',
    success: '/sounds/16bit/success.wav'
  },
  
  'n64': {
    buttonPress: '/sounds/n64/button-press.wav',
    menuMove: '/sounds/n64/menu-move.wav', 
    error: '/sounds/n64/error.wav',
    success: '/sounds/n64/success.wav'
  }
};