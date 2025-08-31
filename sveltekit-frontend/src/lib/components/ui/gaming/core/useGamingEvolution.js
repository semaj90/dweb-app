/**
 * useGamingEvolution Hook
 * Provides reactive access to gaming evolution state and functions
 */

import { getContext } from 'svelte';
import { derived } from 'svelte/store';

export function useGamingEvolution() {
  // Get context from ProgressiveGamingProvider
  const gamingState = getContext('gaming-state');
  const gamingConfig = getContext('gaming-config');
  const gamingFunctions = getContext('gaming-functions');
  const getManager = getContext('gaming-manager');

  if (!gamingState || !gamingConfig || !gamingFunctions) {
    throw new Error('useGamingEvolution must be used within a ProgressiveGamingProvider');
  }

  // Derived reactive values
  const currentEra = derived(gamingState, $state => $state.currentEra);
  const isTransitioning = derived(gamingState, $state => $state.isTransitioning);
  const performanceLevel = derived(gamingState, $state => $state.performanceLevel);
  const availableEras = derived(gamingState, $state => $state.availableEras);

  // Era detection utilities
  const is8Bit = derived(currentEra, $era => $era === '8bit');
  const is16Bit = derived(currentEra, $era => $era === '16bit');
  const isN64 = derived(currentEra, $era => $era === 'n64');

  // Performance utilities
  const isHighPerformance = derived(performanceLevel, $level => $level === 'high');
  const isMediumPerformance = derived(performanceLevel, $level => $level === 'medium');
  const isLowPerformance = derived(performanceLevel, $level => $level === 'low');

  // Configuration utilities
  const enabledFeatures = derived(gamingConfig, $config => ({
    autoEvolution: $config.enableAutoEvolution,
    yorhaIntegration: $config.yorhaIntegration,
    bitsUICompatibility: $config.bitsUICompatibility,
    scanlines: $config.nesSettings?.enableScanlines || false,
    gradients: $config.snesSettings?.enableGradients || false,
    antiAliasing: $config.n64Settings?.enableAntiAliasing || false
  }));

  // Era capabilities
  const eraCapabilities = derived(currentEra, $era => {
    switch ($era) {
      case '8bit':
        return {
          maxColors: 25,
          totalColors: 64,
          resolution: { width: 256, height: 240 },
          audioChannels: 4,
          supportsGradients: false,
          supports3D: false,
          supportsAntiAliasing: false
        };
      case '16bit':
        return {
          maxColors: 256,
          totalColors: 32768,
          resolution: { width: 512, height: 448 },
          audioChannels: 8,
          supportsGradients: true,
          supports3D: false,
          supportsAntiAliasing: false
        };
      case 'n64':
        return {
          maxColors: 16777216,
          totalColors: 16777216,
          resolution: { width: 640, height: 480 },
          audioChannels: 64,
          supportsGradients: true,
          supports3D: true,
          supportsAntiAliasing: true
        };
      default:
        return null;
    }
  });

  // Utility functions
  const canUseFeature = (feature) => {
    const manager = getManager?.();
    if (!manager) return false;

    const capabilities = manager.getCapabilities();
    const state = manager.getCurrentState();

    switch (feature) {
      case 'webgl':
        return capabilities?.webgl || false;
      case 'webgpu':
        return capabilities?.webgpu || false;
      case '3d':
        return state.currentEra === 'n64' && (capabilities?.webgl || capabilities?.webgpu);
      case 'gradients':
        return ['16bit', 'n64'].includes(state.currentEra);
      case 'antialiasing':
        return state.currentEra === 'n64' && state.performanceLevel !== 'low';
      case 'particles':
        return state.currentEra === 'n64' && state.performanceLevel === 'high';
      default:
        return false;
    }
  };

  const getOptimalSettings = () => {
    const manager = getManager?.();
    if (!manager) return null;

    const state = manager.getCurrentState();
    const capabilities = manager.getCapabilities();

    return {
      era: state.currentEra,
      enableEffects: state.performanceLevel !== 'low',
      enableAnimations: state.performanceLevel === 'high',
      enableSounds: capabilities?.gpu !== 'basic',
      pixelPerfect: state.currentEra === '8bit',
      enableScanlines: state.currentEra === '8bit' && state.performanceLevel !== 'low',
      enableGradients: ['16bit', 'n64'].includes(state.currentEra),
      enable3D: state.currentEra === 'n64' && (capabilities?.webgl || capabilities?.webgpu),
      enableParticles: state.currentEra === 'n64' && state.performanceLevel === 'high'
    };
  };

  const getComponentProps = (baseProps = {}) => {
    const settings = getOptimalSettings();
    if (!settings) return baseProps;

    return {
      era: settings.era,
      pixelPerfect: settings.pixelPerfect,
      enableScanlines: settings.enableScanlines && baseProps.enableScanlines !== false,
      enableCRTEffect: settings.enableEffects && baseProps.enableCRTEffect,
      enableGlitchEffect: settings.enableEffects && baseProps.enableGlitchEffect,
      animationStyle: settings.enableAnimations ? 'smooth' : 'instant',
      enableSound: settings.enableSounds && baseProps.enableSound !== false,
      enableParticles: settings.enableParticles && baseProps.enableParticles,
      ...baseProps
    };
  };

  // Performance monitoring
  const performanceMetrics = derived(gamingState, $state => {
    const manager = getManager?.();
    if (!manager) return null;

    const capabilities = manager.getCapabilities();
    return {
      currentLevel: $state.performanceLevel,
      memoryUsage: capabilities?.memory || 0,
      gpuType: capabilities?.gpu || 'unknown',
      webglSupport: capabilities?.webgl || false,
      webgpuSupport: capabilities?.webgpu || false,
      screenSize: capabilities?.screenSize || { width: 0, height: 0 },
      pixelRatio: capabilities?.pixelRatio || 1
    };
  });

  return {
    // State stores
    state: gamingState,
    config: gamingConfig,
    currentEra,
    isTransitioning,
    performanceLevel,
    availableEras,

    // Era detection
    is8Bit,
    is16Bit,
    isN64,

    // Performance detection
    isHighPerformance,
    isMediumPerformance,
    isLowPerformance,

    // Configuration
    enabledFeatures,
    eraCapabilities,
    performanceMetrics,

    // Functions
    setEra: gamingFunctions.setEra,
    upgradeEra: gamingFunctions.upgradeEra,
    downgradeEra: gamingFunctions.downgradeEra,
    updateConfig: gamingFunctions.updateConfig,

    // Utilities
    canUseFeature,
    getOptimalSettings,
    getComponentProps,
    
    // Manager access
    getManager
  };
}