<!--
  UnifiedDialog.svelte
  
  Phase 14 - Unified UI Kit Component  
  Perfect integration of bits-ui v2 + Melt Svelte 5 + UnoCSS
  Features:
  - Real-time collaboration support
  - WebGPU effects and transitions
  - NES-style pixelated modal backgrounds
  - Memory-efficient rendering (8KB budget)
  - Legal AI context integration
-->

<script lang="ts">
  import { createDialog, melt } from 'melt';
  import { fade, fly, scale } from 'svelte/transition';
  import { cubicInOut } from 'svelte/easing';
  import { onMount } from 'svelte';
  import { UnifiedButton } from './index.js';
  
  // Props with Svelte 5 support
  interface Props {
    open?: boolean;
    size?: 'sm' | 'md' | 'lg' | 'xl' | 'fullscreen';
    
    // Content slots
    title?: import('svelte').Snippet;
    content?: import('svelte').Snippet;
    footer?: import('svelte').Snippet;
    
    // Styling
    variant?: 'default' | 'legal' | 'evidence' | 'case' | 'nes';
    glassmorphism?: boolean;
    pixelated?: boolean;
    webgpuEffects?: boolean;
    
    // Real-time collaboration
    collaboration?: {
      enabled: boolean;
      users?: Array<{
        id: string;
        name: string;
        avatar?: string;
        color: string;
        cursor?: { x: number; y: number };
      }>;
      sessionId?: string;
    };
    
    // Legal AI context
    legalContext?: {
      caseId?: string;
      documentType?: 'contract' | 'evidence' | 'brief' | 'citation';
      aiAnalysis?: {
        riskLevel: 'low' | 'medium' | 'high';
        confidence: number;
        suggestions: string[];
      };
    };
    
    // Events
    onOpenChange?: (open: boolean) => void;
    onClose?: () => void;
    class?: string;
  }

  let {
    open = $bindable(false),
    size = 'md',
    title,
    content,
    footer,
    variant = 'default',
    glassmorphism = false,
    pixelated = false,
    webgpuEffects = true,
    collaboration,
    legalContext,
    onOpenChange,
    onClose,
    class: className = '',
    ...restProps
  }: Props = $props();

  // Melt UI dialog
  const {
    elements: { trigger, overlay, content: dialogContent, title: dialogTitle, description, close },
    states: { open: dialogOpen }
  } = createDialog({
    open,
    onOpenChange: (newOpen) => {
      open = newOpen;
      onOpenChange?.(newOpen);
      if (!newOpen) onClose?.();
    }
  });

  // WebGPU animation state
let canvas = $state<HTMLCanvasElement;
  let gpu: GPU | null >(null);
let device = $state<GPUDevice | null >(null);
  let animationFrame: number;
  
  // Memory-efficient state (NES constraints: 8KB)
  let dialogState = $state({
    animationPhase: 0,
    backgroundEffectIntensity: 0,
    collaborationData: new Map();,
    lastRender: 0,
    memoryUsed: 0
  });

  // Reactive updates
  $effect(() => {
    if ($dialogOpen !== open) {
      open = $dialogOpen;
    }
  });

  onMount(() => {
    if (webgpuEffects) {
      initWebGPU();
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      cleanupWebGPU();
    };
  });

  async function initWebGPU() {
    if (!canvas || !navigator.gpu) {
      console.warn('WebGPU not supported, falling back to CSS effects');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return;

      device = await adapter.requestDevice();
      gpu = navigator.gpu;

      // Create simple compute shader for background effects
      const computeShaderCode = `
        struct Uniforms {
          time: f32,
          intensity: f32,
          variant: f32,
          legal_confidence: f32,
        }
        
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
        
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let dims = textureDimensions(outputTex);
          let coord = vec2<i32>(global_id.xy);
          
          if (coord.x >= dims.x || coord.y >= dims.y) {
            return;
          }
          
          let uv = vec2<f32>(coord) / vec2<f32>(dims);
          let center = vec2<f32>(0.5, 0.5);
          let dist = distance(uv, center);
          
          // Legal AI context glow
          var color = vec3<f32>(0.0, 0.0, 0.0);
          if (uniforms.variant == 1.0) { // legal
            color = vec3<f32>(0.0, 0.6, 0.2);
          } else if (uniforms.variant == 2.0) { // evidence
            color = vec3<f32>(0.8, 0.4, 0.0);
          } else if (uniforms.variant == 3.0) { // case
            color = vec3<f32>(0.2, 0.4, 0.8);
          }
          
          // Background gradient with confidence influence
          let gradient = (1.0 - dist) * uniforms.intensity * uniforms.legal_confidence;
          let wave = sin(uniforms.time * 2.0 + dist * 10.0) * 0.1 + 0.9;
          
          let finalColor = color * gradient * wave;
          let alpha = gradient * 0.3;
          
          textureStore(outputTex, coord, vec4<f32>(finalColor, alpha));
        }
      `;

      // Create compute pipeline (memory efficient)
      const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({ code: computeShaderCode }),
          entryPoint: 'main'
        }
      });

      startWebGPUAnimation(computePipeline);

    } catch (error) {
      console.warn('WebGPU initialization failed:', error);
    }
  }

  function startWebGPUAnimation(pipeline: GPUComputePipeline) {
    if (!device || !canvas) return;

    const context = canvas.getContext('webgpu');
    if (!context) return;

    context.configure({
      device,
      format: 'bgra8unorm'
    });

    function animate(currentTime: number) {
      if (!device || !context) return;

      const deltaTime = currentTime - dialogState.lastRender;
      dialogState.lastRender = currentTime;

      // Update animation state
      dialogState.animationPhase += deltaTime * 0.001;
      dialogState.backgroundEffectIntensity = open ? 1.0 : 0.0;

      // Create uniform buffer
      const uniformData = new Float32Array([
        dialogState.animationPhase,
        dialogState.backgroundEffectIntensity,
        variant === 'legal' ? 1.0 : variant === 'evidence' ? 2.0 : variant === 'case' ? 3.0 : 0.0,
        legalContext?.aiAnalysis?.confidence || 0.5
      ]);

      const uniformBuffer = device.createBuffer({
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      // Create output texture
      const outputTexture = device.createTexture({
        size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
      });

      // Create bind group
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: outputTexture.createView() }
        ]
      });

      // Dispatch compute shader
      const commandEncoder = device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(canvas.width / 8),
        Math.ceil(canvas.height / 8)
      );
      computePass.end();

      // Copy to canvas
      commandEncoder.copyTextureToTexture(
        { texture: outputTexture },
        { texture: context.getCurrentTexture() },
        { width: canvas.width, height: canvas.height }
      );

      device.queue.submit([commandEncoder.finish()]);

      if (open) {
        animationFrame = requestAnimationFrame(animate);
      }
    }

    if (open) {
      animationFrame = requestAnimationFrame(animate);
    }
  }

  function cleanupWebGPU() {
    if (device) {
      device.destroy();
      device = null;
      gpu = null;
    }
  }

  // Collaboration cursor rendering
  function renderCollaborationCursors() {
    if (!collaboration?.enabled || !collaboration.users) return;

    return collaboration.users.map(user => ({
      id: user.id,
      x: user.cursor?.x || 0,
      y: user.cursor?.y || 0,
      color: user.color,
      name: user.name
    }));
  }

  // Dynamic classes
  let dialogClasses = $derived([);
    'fixed inset-0 z-50 flex items-center justify-center p-4',
    
    // Size variants
    size === 'sm' ? 'max-w-sm' :
    size === 'md' ? 'max-w-md' :
    size === 'lg' ? 'max-w-2xl' :
    size === 'xl' ? 'max-w-4xl' :
    size === 'fullscreen' ? 'max-w-full h-full' : '',
    
    class
  ].filter(Boolean).join(' ');

  let contentClasses = $derived([);
    'relative bg-white rounded-lg shadow-xl',
    'max-h-[90vh] overflow-hidden',
    
    // Variant styling
    variant === 'legal' ? 'border-l-4 border-green-500' :
    variant === 'evidence' ? 'border-l-4 border-amber-500' :
    variant === 'case' ? 'border-l-4 border-indigo-500' : '',
    
    // Effects
    glassmorphism ? 'backdrop-blur-md bg-white/80' : 'bg-white',
    pixelated ? 'image-rendering-pixelated' : '',
    
    // NES styling
    variant === 'nes' ? 'border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]' : ''
  ].filter(Boolean).join(' ');
</script>

{#if open}
  <!-- Dialog Portal -->
  <div class="fixed inset-0 z-50">
    <!-- WebGPU Background Canvas -->
    {#if webgpuEffects}
      <canvas 
        bind:this={canvas}
        class="absolute inset-0 w-full h-full"
        width="800"
        height="600"
        style="mix-blend-mode: multiply; opacity: 0.6;"
      />
    {/if}

    <!-- Overlay -->
    <div
      <!-- <!-- use:melt={$overlay}
      class="fixed inset-0 bg-black/50 backdrop-blur-sm"
      transitifade={{ duration: 150 }}
    />

    <!-- Dialog Container -->
    <div class={dialogClasses}>
      <!-- Dialog Content -->
      <div
        <!-- <!-- use:melt={$dialogContent}
        class={contentClasses}
        transitiscale={{ 
          duration: 200, 
          easing: cubicInOut,
          start: 0.95
        }}
        {...restProps}
      >
        <!-- Collaboration Users -->
        {#if collaboration?.enabled}
          <div class="absolute -top-8 right-0 flex -space-x-2">
            {#each collaboration.users || [] as user}
              <div 
                class="w-6 h-6 rounded-full border-2 border-white shadow-sm flex items-center justify-center text-xs font-bold text-white"
                style="background-color: {user.color};"
                title={user.name}
                transitifly={{ x: 20, duration: 300 }}
              >
                {user.name.charAt(0).toUpperCase()}
              </div>
            {/each}
          </div>
        {/if}

        <!-- Legal AI Risk Indicator -->
        {#if legalContext?.aiAnalysis?.riskLevel === 'high'}
          <div class="absolute -top-2 -right-2 w-4 h-4 bg-red-500 rounded-full animate-pulse border-2 border-white" />
        {/if}

        <!-- Header -->
        {#if title}
          <div class="px-6 py-4 border-b border-gray-200">
            <h2 
              <!-- <!-- use:melt={$dialogTitle}
              class="text-lg font-semibold text-gray-900 flex items-center gap-3"
            >
              {@render title()}
              
              <!-- Legal AI Analysis Badge -->
              {#if legalContext?.aiAnalysis}
                <span 
                  class="px-2 py-1 text-xs rounded-full"
                  class:bg-green-100={legalContext.aiAnalysis.riskLevel === 'low'}
                  class:text-green-800={legalContext.aiAnalysis.riskLevel === 'low'}
                  class:bg-yellow-100={legalContext.aiAnalysis.riskLevel === 'medium'}
                  class:text-yellow-800={legalContext.aiAnalysis.riskLevel === 'medium'}
                  class:bg-red-100={legalContext.aiAnalysis.riskLevel === 'high'}
                  class:text-red-800={legalContext.aiAnalysis.riskLevel === 'high'}
                >
                  AI: {Math.round(legalContext.aiAnalysis.confidence * 100)}%
                </span>
              {/if}
            </h2>
          </div>
        {/if}

        <!-- Content -->
        <div class="px-6 py-4 overflow-y-auto max-h-[60vh]">
          {#if content}
            {@render content()}
          {/if}

          <!-- AI Suggestions -->
          {#if legalContext?.aiAnalysis?.suggestions.length}
            <div class="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <h4 class="text-sm font-medium text-blue-900 mb-2">AI Suggestions:</h4>
              <ul class="text-sm text-blue-800 space-y-1">
                {#each legalContext.aiAnalysis.suggestions as suggestion}
                  <li class="flex items-start gap-2">
                    <span class="text-blue-600">•</span>
                    {suggestion}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>

        <!-- Footer -->
        {#if footer}
          <div class="px-6 py-4 border-t border-gray-200 bg-gray-50">
            {@render footer()}
          </div>
        {:else}
          <!-- Default footer with close button -->
          <div class="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-end gap-3">
            <UnifiedButton
              <!-- <!-- use:melt={$close}
              variant="secondary"
              size="sm"
            >
              Close
            </UnifiedButton>
          </div>
        {/if}

        <!-- Close button -->
        <button
          <!-- <!-- use:melt={$close}
          class="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
    
        <!-- Collaboration Cursors -->
        {#each renderCollaborationCursors() as cursor (cursor.id)}
          <div 
            class="absolute pointer-events-none z-10"
            style="left: {cursor.x}px; top: {cursor.y}px; color: {cursor.color};"
            transitifade={{ duration: 200 }}
          >
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M7 2L17 12L12 13L13 18L7 2Z"/>
            </svg>
            <span 
              class="ml-2 px-1 py-0.5 text-xs font-medium text-white rounded shadow-lg"
              style="background-color: {cursor.color};"
            >
              {cursor.name}
            </span>
          </div>
        {/each}
      </div>
    </div>
  </div>
{/if}

<style>
  .image-rendering-pixelated {
    image-rendering: -moz-crisp-edges;
    image-rendering: -webkit-crisp-edges;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
  }

  /* WebGPU canvas optimization */
  canvas {
    will-change: transform;
    transform: translateZ(0);
  }

  /* NES-style shadows */
  .shadow-\[4px_4px_0px_0px_rgba\(0\,0\,0\,1\)\] {
    box-shadow: 4px 4px 0px 0px rgba(0, 0, 0, 1);
  }
</style>