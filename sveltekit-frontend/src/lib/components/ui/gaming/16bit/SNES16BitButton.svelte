<!--
  SNES 16-Bit Enhanced Button Component
  Advanced 16-bit styling with gradients and enhanced visuals
  
  Features:
  - Extended SNES color palette
  - Gradient effects and layering
  - Improved animations
  - Mode 7-inspired transformations
  - Enhanced audio capabilities
-->
<script lang="ts">
  import { Button as BitsButton } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  import type { GamingComponentProps } from '../types/gaming-types.js';
  import { SNES_COLOR_PALETTE } from '../constants/gaming-constants.js';

  interface Props extends GamingComponentProps {
    // Button specific props
    type?: 'button' | 'submit' | 'reset';
    form?: string;
    name?: string;
    value?: string;
    
    // SNES-specific styling
    gradientDirection?: 'horizontal' | 'vertical' | 'diagonal' | 'radial';
    enableLayerEffects?: boolean;
    enableMode7?: boolean;
    plasmaEffect?: boolean;
    
    // Enhanced audio
    enableEnhancedSound?: boolean;
    soundChannel?: number; // SNES had 8 audio channels
    
    // Content
    children?: any;
    class?: string;
  }

  let {
    era = '16bit',
    variant = 'primary',
    size = 'medium',
    disabled = false,
    loading = false,
    pixelPerfect = false, // SNES had smoother graphics
    enableScanlines = false,
    enableCRTEffect = false,
    animationStyle = 'smooth',
    
    type = 'button',
    form,
    name,
    value,
    
    gradientDirection = 'vertical',
    enableLayerEffects = true,
    enableMode7 = false,
    plasmaEffect = false,
    
    enableEnhancedSound = false,
    soundChannel = 1,
    
    children,
    class: className = '',
    
    onClick,
    onHover,
    onFocus
  }: Props = $props();

  const dispatch = createEventDispatcher();

  let isPressed = $state(false);
  let isHovered = $state(false);
let audioContext = $state<AudioContext | null >(null);
let buttonElement = $state<HTMLButtonElement | null >(null);

  // Create 16-bit enhanced button sound with multiple channels
  const playEnhancedButtonSound = async () => {
    if (!enableEnhancedSound) return;
    
    try {
      if (!audioContext) {
        audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      
      // Create multiple oscillators for richer sound (SNES had 8 channels)
      const oscillators: OscillatorNode[] = [];
      const gainNodes: GainNode[] = [];
      
      // Main tone
      const mainOsc = audioContext.createOscillator();
      const mainGain = audioContext.createGain();
      mainOsc.connect(mainGain);
      mainGain.connect(audioContext.destination);
      
      mainOsc.type = 'square';
      mainOsc.frequency.setValueAtTime(800, audioContext.currentTime);
      mainOsc.frequency.exponentialRampToValueAtTime(600, audioContext.currentTime + 0.1);
      mainGain.gain.setValueAtTime(0.3, audioContext.currentTime);
      mainGain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.15);
      
      // Harmony (5th)
      const harmonyOsc = audioContext.createOscillator();
      const harmonyGain = audioContext.createGain();
      harmonyOsc.connect(harmonyGain);
      harmonyGain.connect(audioContext.destination);
      
      harmonyOsc.type = 'triangle';
      harmonyOsc.frequency.setValueAtTime(1200, audioContext.currentTime);
      harmonyOsc.frequency.exponentialRampToValueAtTime(900, audioContext.currentTime + 0.1);
      harmonyGain.gain.setValueAtTime(0.15, audioContext.currentTime);
      harmonyGain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.12);
      
      oscillators.push(mainOsc, harmonyOsc);
      gainNodes.push(mainGain, harmonyGain);
      
      // Start all oscillators
      oscillators.forEach(osc => osc.start());
      oscillators.forEach(osc => osc.stop(audioContext!.currentTime + 0.2));
      
    } catch (error) {
      console.warn('Could not play enhanced button sound:', error);
    }
  };

  const handleClick = async () => {
    if (disabled || loading) return;
    
    isPressed = true;
    await playEnhancedButtonSound();
    
    setTimeout(() => {
      isPressed = false;
    }, 120);
    
    onClick?.();
    dispatch('click');
  };

  const handleHover = () => {
    if (disabled) return;
    isHovered = true;
    onHover?.();
    dispatch('hover');
  };

  const handleUnhover = () => {
    isHovered = false;
  };

  const handleFocus = () => {
    if (disabled) return;
    onFocus?.();
    dispatch('focus');
  };

  // Get SNES gradient based on variant
  const getVariantGradient = (variant: string, direction: string) => {
    const baseColors = {
      'primary': SNES_COLOR_PALETTE.primaryGradient,
      'secondary': [SNES_COLOR_PALETTE.lightGray, SNES_COLOR_PALETTE.darkGray, SNES_COLOR_PALETTE.black],
      'success': [SNES_COLOR_PALETTE.lime, SNES_COLOR_PALETTE.green, '#4a7c23'],
      'warning': [SNES_COLOR_PALETTE.yellow, SNES_COLOR_PALETTE.orange, '#cc6600'],
      'error': [SNES_COLOR_PALETTE.red, '#cc2800', '#800000'],
      'info': [SNES_COLOR_PALETTE.cyan, SNES_COLOR_PALETTE.blue, '#0050cc']
    };
    
    const colors = baseColors[variant as keyof typeof baseColors] || baseColors.primary;
    
    const directionMap = {
      'horizontal': 'to right',
      'vertical': 'to bottom',
      'diagonal': 'to bottom right', 
      'radial': 'radial-gradient(circle'
    };
    
    const gradientType = direction === 'radial' ? 'radial-gradient(circle, ' : 'linear-gradient(' + directionMap[direction as keyof typeof directionMap] + ', ';
    
    return gradientType + colors.join(', ') + ')';
  };

  const getSizeStyles = (size: string) => {
    const sizeMap = {
      small: { padding: '10px 16px', fontSize: '11px', minHeight: '36px' },
      medium: { padding: '14px 20px', fontSize: '13px', minHeight: '44px' },
      large: { padding: '18px 24px', fontSize: '15px', minHeight: '52px' },
      xl: { padding: '22px 28px', fontSize: '17px', minHeight: '60px' }
    };
    return sizeMap[size as keyof typeof sizeMap] || sizeMap.medium;
  };

  $: sizeStyles = getSizeStyles(size);
  $: variantGradient = getVariantGradient(variant, gradientDirection);
  $: mode7Transform = enableMode7 && isPressed ? 'perspective(100px) rotateX(5deg) scale(0.95)' : 
                      enableMode7 && isHovered ? 'perspective(200px) rotateX(-2deg) scale(1.02)' : 'none';
</script>

<BitsButton.Root
  bind:el={buttonElement}
  {type}
  {disabled}
  {form}
  {name}
  {value}
  on:on:on:click={handleClick}
  on:on:on:mouseenter={handleHover}
  on:on:on:mouseleave={handleUnhover}
  on:focus={handleFocus}
  class="snes-16bit-button {className} {enableLayerEffects ? 'layer-effects' : ''} {enableMode7 ? 'mode7' : ''} {plasmaEffect ? 'plasma' : ''}"
  style="
    --button-gradient: {variantGradient};
    --button-padding: {sizeStyles.padding};
    --button-font-size: {sizeStyles.fontSize};
    --button-min-height: {sizeStyles.minHeight};
    --mode7-transform: {mode7Transform};
  "
>
  {#if loading}
    <div class="loading-spinner" role="status" aria-label="Loading">
      <div class="enhanced-spinner"></div>
    </div>
  {:else}
    {@render children?.()}
  {/if}
</BitsButton.Root>

<style>
  :global(.snes-16bit-button) {
    /* Base SNES button styling */
    font-family: 'Orbitron', 'Arial', sans-serif;
    background: var(--button-gradient);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 3px;
    padding: var(--button-padding);
    font-size: var(--button-font-size);
    min-height: var(--button-min-height);
    font-weight: 500;
    
    /* Enhanced rendering (smoother than 8-bit) */
    image-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    
    /* Enhanced 3D effect */
    box-shadow: 
      0 2px 0px rgba(0, 0, 0, 0.3),
      inset 0 1px 0px rgba(255, 255, 255, 0.4),
      inset 0 -1px 0px rgba(0, 0, 0, 0.2);
    
    transform: var(--mode7-transform, none);
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    
    /* Remove default button styles */
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    outline: none;
    
    /* Prevent text selection */
    -webkit-user-select: none;
    -moz-user-select: none;
    user-select: none;
    
    /* Cursor */
    cursor: pointer;
    
    /* Text styling */
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    
    /* Flexbox for content alignment */
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    
    /* Layer effects support */
    position: relative;
    overflow: hidden;
  }

  /* Layer effects (pseudo-Mode 7 layering) */
  :global(.snes-16bit-button.layer-effects::before) {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, 
      rgba(255, 255, 255, 0.2) 0%,
      transparent 50%,
      rgba(0, 0, 0, 0.1) 100%
    );
    pointer-events: none;
    opacity: 0.6;
  }

  /* Plasma effect */
  :global(.snes-16bit-button.plasma) {
    animation: plasmaShift 3s ease-in-out infinite alternate;
    background-size: 200% 200%;
  }

  @keyframes plasmaShift {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
  }

  /* Mode 7 perspective effects */
  :global(.snes-16bit-button.mode7) {
    transform-style: preserve-3d;
  }

  /* Hover effects */
  :global(.snes-16bit-button:not(:disabled):hover) {
    border-color: rgba(255, 255, 255, 0.5);
    box-shadow: 
      0 3px 0px rgba(0, 0, 0, 0.3),
      inset 0 1px 0px rgba(255, 255, 255, 0.6),
      inset 0 -1px 0px rgba(0, 0, 0, 0.1),
      0 4px 8px rgba(0, 0, 0, 0.2);
    
    filter: brightness(1.1) saturate(1.1);
  }

  /* Active/Pressed state */
  :global(.snes-16bit-button:not(:disabled):active) {
    box-shadow: 
      0 1px 0px rgba(0, 0, 0, 0.3),
      inset 0 1px 0px rgba(255, 255, 255, 0.3),
      inset 0 2px 4px rgba(0, 0, 0, 0.3);
  }

  /* Disabled state */
  :global(.snes-16bit-button:disabled) {
    background: linear-gradient(to bottom, #7c7c7c, #5c5c5c, #3c3c3c);
    color: #bcbcbc;
    cursor: not-allowed;
    opacity: 0.7;
    transform: none !important;
    box-shadow: 
      0 1px 0px rgba(0, 0, 0, 0.2),
      inset 0 1px 0px rgba(255, 255, 255, 0.1);
  }

  /* Focus styles for accessibility */
  :global(.snes-16bit-button:focus-visible) {
    outline: 2px solid #ffffff;
    outline-offset: 2px;
  }

  /* Enhanced loading spinner */
  .loading-spinner {
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .enhanced-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-right: 2px solid currentColor;
    border-bottom: 1px solid rgba(255, 255, 255, 0.3);
    border-left: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    animation: enhancedSpin 0.8s ease-in-out infinite;
  }

  @keyframes enhancedSpin {
    0% { transform: rotate(0deg); border-radius: 50%; }
    50% { transform: rotate(180deg); border-radius: 30%; }
    100% { transform: rotate(360deg); border-radius: 50%; }
  }

  /* Variant-specific enhancements */
  :global(.snes-16bit-button.variant-primary) {
    background: linear-gradient(to bottom, #5cb3ff, #3cbcfc, #0084ff);
  }

  :global(.snes-16bit-button.variant-success) {
    background: linear-gradient(to bottom, #9cfc38, #92cc41, #4a7c23);
  }

  :global(.snes-16bit-button.variant-warning) {
    background: linear-gradient(to bottom, #f7d51d, #fc9838, #cc6600);
    color: #000000;
    text-shadow: 0 1px 1px rgba(255, 255, 255, 0.5);
  }

  :global(.snes-16bit-button.variant-error) {
    background: linear-gradient(to bottom, #fc5c5c, #f83800, #cc2800);
  }

  /* Scanlines effect (optional) */
  :global(.snes-16bit-button.enable-scanlines::after) {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 1.5px,
      rgba(0, 0, 0, 0.05) 1.5px,
      rgba(0, 0, 0, 0.05) 3px
    );
    pointer-events: none;
  }

  /* CRT effect (enhanced for 16-bit) */
  :global(.snes-16bit-button.enable-crt) {
    filter: contrast(1.1) brightness(1.05) saturate(1.2);
    border-radius: 3px;
    box-shadow: 
      inset 0 0 0 1px rgba(255, 255, 255, 0.1),
      0 2px 0px rgba(0, 0, 0, 0.3),
      0 0 6px rgba(0, 0, 0, 0.2);
  }

  /* Smooth animation style */
  :global(.snes-16bit-button.smooth) {
    transition: all 200ms cubic-bezier(0.25, 0.46, 0.45, 0.94);
  }

  :global(.snes-16bit-button.smooth:hover) {
    animation: smoothHover 0.4s ease-in-out;
  }

  @keyframes smoothHover {
    0%, 100% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-1px) scale(1.01); }
  }

  /* Enhanced glitch transition */
  :global(.snes-16bit-button.glitch-transition:hover) {
    animation: enhancedGlitch 0.3s ease-in-out;
  }

  @keyframes enhancedGlitch {
    0% { transform: translateY(0px); filter: hue-rotate(0deg); }
    20% { transform: translateY(-0.5px) translateX(0.5px); filter: hue-rotate(90deg); }
    40% { transform: translateY(0.5px) translateX(-0.5px); filter: hue-rotate(180deg); }
    60% { transform: translateY(-0.5px) translateX(0.5px); filter: hue-rotate(270deg); }
    80% { transform: translateY(0.5px) translateX(-0.5px); filter: hue-rotate(360deg); }
    100% { transform: translateY(0px); filter: hue-rotate(0deg); }
  }

  /* Mobile optimizations */
  @media (max-width: 480px) {
    :global(.snes-16bit-button) {
      min-height: 44px; /* iOS touch target minimum */
      font-size: 11px;
    }
    
    /* Disable complex effects on mobile for performance */
    :global(.snes-16bit-button.plasma) {
      animation: none;
      background-size: 100% 100%;
    }
    
    :global(.snes-16bit-button.mode7) {
      transform: none !important;
    }
  }

  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    :global(.snes-16bit-button) {
      animation: none;
      transition: opacity 150ms ease;
    }
    
    :global(.snes-16bit-button.plasma) {
      animation: none;
    }
    
    :global(.enhanced-spinner) {
      animation: none;
      border: 2px solid currentColor;
    }
  }

  /* High contrast mode */
  @media (prefers-contrast: high) {
    :global(.snes-16bit-button) {
      border-width: 2px;
      border-color: currentColor;
      text-shadow: none;
    }
  }

  /* Dark mode adjustments */
  @media (prefers-color-scheme: dark) {
    :global(.snes-16bit-button) {
      box-shadow: 
        0 2px 0px rgba(255, 255, 255, 0.1),
        inset 0 1px 0px rgba(255, 255, 255, 0.2),
        inset 0 -1px 0px rgba(0, 0, 0, 0.4);
    }
  }
</style>