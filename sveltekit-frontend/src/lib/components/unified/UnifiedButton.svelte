<!--
  UnifiedButton.svelte
  
  Phase 14 - Unified UI Kit Component
  Perfect integration of bits-ui v2 + Melt Svelte 5 + UnoCSS
  Features:
  - GPU animations with WebGL confidence glow effects  
  - Legal AI context integration
  - NES-style pixelated transitions
  - Memory-efficient animations (2KB budget)
-->

<script lang="ts">
  import { createButton, melt } from 'melt';
  import { fly, fade } from 'svelte/transition';
  import { spring } from 'svelte/motion';
  import { onMount } from 'svelte';
  
  // Props with Svelte 5 bindable support
  interface Props {
    variant?: 'primary' | 'secondary' | 'legal' | 'evidence' | 'case' | 'ghost';
    size?: 'sm' | 'md' | 'lg' | 'xl';
    disabled?: boolean;
    loading?: boolean;
    children?: import('svelte').Snippet;
    
    // Legal AI Context
    legalContext?: {
      confidence?: number;
      caseType?: 'contract' | 'evidence' | 'brief' | 'citation';
      aiSuggested?: boolean;
      riskLevel?: 'low' | 'medium' | 'high';
    };
    
    // GPU Animation Settings
    gpuEffects?: boolean;
    glowIntensity?: number;
    pixelated?: boolean;
    nesStyle?: boolean;
    
    // Event handlers
    onclick?: (event: MouseEvent) => void;
    class?: string;
  }

  let {
    variant = 'primary',
    size = 'md',
    disabled = false,
    loading = false,
    children,
    legalContext,
    gpuEffects = true,
    glowIntensity = 0.8,
    pixelated = false,
    nesStyle = false,
    onclick,
    class: className = '',
    ...restProps
  }: Props = $props();

  // Melt UI button
  const {
    elements: { root },
    states: { pressed }
  } = createButton({
    disabled: disabled || loading
  });

  // GPU Animation State
let canvas = $state<HTMLCanvasElement;
  let gl: WebGLRenderingContext | null >(null);
  let animationFrame: number;
  let isHovered = $state(false);
  let isPressed = $state(false);
  
  // Legal AI confidence animation
  const confidence = spring(legalContext?.confidence || 0, {
    stiffness: 0.3,
    damping: 0.8
  });

  // Memory-efficient animation state (NES constraints: 2KB)
  let animationState = $state({
    glowPhase: 0,
    pulseIntensity: 0,
    lastFrame: 0,
    memoryUsed: 0
  });

  onMount(() => {
    if (gpuEffects && canvas) {
      initWebGL();
      startAnimation();
    }
    
    // Update confidence when legalContext changes
    $effect(() => {
      if (legalContext?.confidence !== undefined) {
        confidence.set(legalContext.confidence);
      }
    });

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      cleanupWebGL();
    };
  });

  function initWebGL() {
    if (!canvas) return;
    
    gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      console.warn('WebGL not supported, falling back to CSS animations');
      return;
    }

    // Create minimal shader for legal confidence glow
    const vertexShaderSource = `
      attribute vec4 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = a_position;
        v_texCoord = a_texCoord;
      }
    `;

    const fragmentShaderSource = `
      precision mediump float;
      uniform float u_confidence;
      uniform float u_time;
      uniform float u_glow;
      varying vec2 v_texCoord;
      
      void main() {
        vec2 center = vec2(0.5, 0.5);
        float distance = length(v_texCoord - center);
        
        // Legal confidence glow effect
        float glow = u_confidence * u_glow * (1.0 - distance);
        glow *= (sin(u_time * 2.0) * 0.3 + 0.7); // Subtle pulse
        
        // Color based on legal context
        vec3 color = vec3(0.0, 0.8, 0.2); // Legal green
        if (u_confidence > 0.8) {
          color = vec3(0.0, 0.9, 0.1); // High confidence
        } else if (u_confidence < 0.4) {
          color = vec3(0.9, 0.5, 0.0); // Low confidence warning
        }
        
        gl_FragColor = vec4(color * glow, glow * 0.6);
      }
    `;

    // Compile shaders (memory-efficient)
    const program = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (!program) return;

    // Set up minimal geometry
    const vertices = new Float32Array([
      -1, -1, 0, 0,
       1, -1, 1, 0,
      -1,  1, 0, 1,
       1,  1, 1, 1
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    gl.useProgram(program);
    
    // Set up attributes
    const positionLocation = gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
    
    gl.enableVertexAttribArray(positionLocation);
    gl.enableVertexAttribArray(texCoordLocation);
    
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 16, 8);

    // Store uniforms for animation
    gl.confidenceUniform = gl.getUniformLocation(program, 'u_confidence');
    gl.timeUniform = gl.getUniformLocation(program, 'u_time');
    gl.glowUniform = gl.getUniformLocation(program, 'u_glow');
  }

  function createShaderProgram(vertexSource: string, fragmentSource: string) {
    if (!gl) return null;

    const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
    
    if (!vertexShader || !fragmentShader) return null;

    const program = gl.createProgram()!;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Shader program linking failed:', gl.getProgramInfoLog(program));
      return null;
    }

    return program;
  }

  function compileShader(type: number, source: string) {
    if (!gl) return null;

    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compilation failed:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  function startAnimation() {
    if (!gl) return;

    function animate(currentTime: number) {
      if (!gl || !canvas) return;

      const deltaTime = currentTime - animationState.lastFrame;
      animationState.lastFrame = currentTime;

      // Update animation state (memory efficient)
      animationState.glowPhase += deltaTime * 0.001;
      animationState.pulseIntensity = isHovered ? 1.0 : 0.3;

      // Render WebGL effect
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      // Set uniforms
      gl.uniform1f(gl.confidenceUniform, $confidence);
      gl.uniform1f(gl.timeUniform, animationState.glowPhase);
      gl.uniform1f(gl.glowUniform, glowIntensity * animationState.pulseIntensity);

      // Draw
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      animationFrame = requestAnimationFrame(animate);
    }

    animationFrame = requestAnimationFrame(animate);
  }

  function cleanupWebGL() {
    if (gl) {
      // Cleanup WebGL resources
      gl = null;
    }
  }

  function handleClick(event: MouseEvent) {
    if (disabled || loading) return;
    
    isPressed = true;
    setTimeout(() => isPressed = false, 150);
    
    onclick?.(event);
  }

  function handleMouseEnter() {
    isHovered = true;
  }

  function handleMouseLeave() {
    isHovered = false;
  }

  // Dynamic classes based on props
  let baseClasses = $derived([);
    // Base button styles
    'relative inline-flex items-center justify-center',
    'font-medium transition-all duration-200',
    'focus:outline-none focus:ring-2 focus:ring-offset-2',
    
    // Size variants
    size === 'sm' ? 'px-3 py-1.5 text-sm rounded' :
    size === 'md' ? 'px-4 py-2 text-sm rounded-md' :
    size === 'lg' ? 'px-6 py-3 text-base rounded-lg' :
    size === 'xl' ? 'px-8 py-4 text-lg rounded-xl' : '',
    
    // Variant styles
    variant === 'primary' ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500' :
    variant === 'secondary' ? 'bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500' :
    variant === 'legal' ? 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500' :
    variant === 'evidence' ? 'bg-amber-600 text-white hover:bg-amber-700 focus:ring-amber-500' :
    variant === 'case' ? 'bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-500' :
    variant === 'ghost' ? 'text-gray-700 hover:bg-gray-100 focus:ring-gray-500' : '',
    
    // Legal context styling
    legalContext?.riskLevel === 'high' ? 'border-2 border-red-400' :
    legalContext?.riskLevel === 'medium' ? 'border-2 border-yellow-400' :
    legalContext?.aiSuggested ? 'border-2 border-blue-400' : '',
    
    // NES/Pixel styling
    nesStyle ? 'font-mono text-xs tracking-wider' : '',
    pixelated ? 'image-rendering-pixelated' : '',
    
    // State classes
    disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
    loading ? 'cursor-wait' : '',
    isPressed ? 'scale-95' : '',
    
    class
  ].filter(Boolean).join(' ');

  // Legal confidence indicator
  let confidenceColor = $derived($confidence > 0.8 ? 'text-green-500' :);
    $confidence > 0.5 ? 'text-yellow-500' : 'text-red-500';
</script>

<!-- Button with GPU animation overlay -->
<div class="relative inline-block">
  <!-- WebGL Canvas for GPU effects -->
  {#if gpuEffects}
    <canvas 
      bind:this={canvas}
      class="absolute inset-0 pointer-events-none rounded-inherit"
      width="100"
      height="40"
      style="mix-blend-mode: screen; opacity: 0.8;"
    />
  {/if}

  <!-- Main button -->
  <button
    <!-- <!-- use:melt={$root}
    class={baseClasses}
    on:onclick={handleClick}
    on:on:mouseenter={handleMouseEnter}
    on:on:mouseleave={handleMouseLeave}
    {...restProps}
  >
    <!-- Loading spinner -->
    {#if loading}
      <div 
        class="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
        transitifade={{ duration: 200 }}
      />
    {/if}

    <!-- Button content -->
    <span class="relative z-10 flex items-center gap-2">
      {#if children}
        {@render children()}
      {/if}
      
      <!-- Legal AI confidence indicator -->
      {#if legalContext?.confidence !== undefined}
        <span 
          class="ml-2 text-xs {confidenceColor} font-mono"
          title="AI Confidence: {Math.round($confidence * 100)}%"
        >
          {Math.round($confidence * 100)}%
        </span>
      {/if}
      
      <!-- Legal context icon -->
      {#if legalContext?.aiSuggested}
        <div 
          class="ml-1 h-2 w-2 rounded-full bg-blue-400 animate-pulse"
          title="AI Suggested"
          transitifly={{ y: -10, duration: 300 }}
        />
      {/if}
    </span>

    <!-- Legal risk indicator -->
    {#if legalContext?.riskLevel === 'high'}
      <div 
        class="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-red-500 animate-ping"
        transition:fade
      />
    {/if}
  </button>

  <!-- NES-style shadow effect -->
  {#if nesStyle}
    <div 
      class="absolute inset-0 translate-x-0.5 translate-y-0.5 -z-10 bg-black/20 rounded-inherit"
      style="image-rendering: pixelated;"
    />
  {/if}
</div>

<style>
  .image-rendering-pixelated {
    image-rendering: -moz-crisp-edges;
    image-rendering: -webkit-crisp-edges;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
  }
  
  .rounded-inherit {
    border-radius: inherit;
  }

  /* GPU animation performance optimizations */
  canvas {
    will-change: transform;
    transform: translateZ(0);
  }
  
  /* NES-style font rendering */
  .font-mono {
    font-family: 'Courier New', 'Monaco', monospace;
    font-feature-settings: normal;
  }
</style>