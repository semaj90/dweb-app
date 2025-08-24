/**
 * GPU-Accelerated Animations System - Phase 14
 * 
 * Advanced animation system with:
 * - WebGL shader-based legal confidence glow effects
 * - Priority-based animations with memory management  
 * - NES-style pixelated transitions
 * - Legal AI context-aware visual effects
 */

interface AnimationContext {
  element: HTMLElement;
  id: string;
  priority: number;
  type: 'confidence' | 'risk' | 'transition' | 'pulse' | 'nes-pixel';
  duration: number;
  legalContext?: {
    confidence: number;
    riskLevel: 'low' | 'medium' | 'high';
    aiSuggested: boolean;
    documentType: 'contract' | 'evidence' | 'brief' | 'citation';
  };
  nesStyle?: boolean;
  pixelated?: boolean;
  shader?: WebGLProgram;
}

// NES-inspired memory constraints for animations
const ANIMATION_MEMORY = {
  MAX_ACTIVE_ANIMATIONS: 8, // Like NES sprite limit
  SHADER_CACHE_SIZE: 4096, // 4KB for compiled shaders
  FRAME_BUFFER_SIZE: 2048, // 2KB for frame data
  TOTAL_BUDGET: 8192 // 8KB total animation memory
} as const;

export class GPUAnimationEngine {
  private gl: WebGLRenderingContext | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private shaderCache: Map<string, WebGLProgram> = new Map();
  private activeAnimations: Map<string, AnimationContext> = new Map();
  private animationFrame: number = 0;
  private memoryUsed = 0;
  
  // Shader sources
  private vertexShaderSources = {
    confidence: `
      attribute vec4 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      uniform float u_time;
      uniform float u_confidence;
      
      void main() {
        vec4 position = a_position;
        
        // Slight shake for low confidence
        if (u_confidence < 0.5) {
          float shake = sin(u_time * 20.0) * 0.005 * (0.5 - u_confidence);
          position.x += shake;
          position.y += shake * 0.5;
        }
        
        gl_Position = position;
        v_texCoord = a_texCoord;
      }
    `,
    
    nesPixel: `
      attribute vec4 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      uniform float u_time;
      uniform float u_pixelSize;
      
      void main() {
        vec4 position = a_position;
        
        // Pixelate position for NES effect
        position.x = floor(position.x * u_pixelSize) / u_pixelSize;
        position.y = floor(position.y * u_pixelSize) / u_pixelSize;
        
        gl_Position = position;
        v_texCoord = a_texCoord;
      }
    `,
    
    transition: `
      attribute vec4 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      uniform float u_time;
      uniform float u_progress;
      
      void main() {
        vec4 position = a_position;
        
        // Scale based on transition progress
        position.xy *= (1.0 - u_progress * 0.1);
        
        gl_Position = position;
        v_texCoord = a_texCoord;
      }
    `
  };

  private fragmentShaderSources = {
    confidence: `
      precision mediump float;
      varying vec2 v_texCoord;
      uniform float u_time;
      uniform float u_confidence;
      uniform float u_riskLevel; // 0=low, 1=medium, 2=high
      
      void main() {
        vec2 center = vec2(0.5, 0.5);
        float distance = length(v_texCoord - center);
        
        // Base glow intensity based on confidence
        float glow = u_confidence * (1.0 - distance);
        
        // Pulse effect
        float pulse = sin(u_time * 3.0) * 0.3 + 0.7;
        glow *= pulse;
        
        // Risk-based color
        vec3 color = vec3(0.0, 0.8, 0.2); // Default green
        if (u_riskLevel >= 2.0) {
          color = vec3(0.9, 0.2, 0.1); // High risk red
        } else if (u_riskLevel >= 1.0) {
          color = vec3(0.9, 0.6, 0.1); // Medium risk yellow
        }
        
        // Legal confidence glow
        float alpha = glow * 0.6;
        gl_FragColor = vec4(color * glow, alpha);
      }
    `,
    
    nesPixel: `
      precision mediump float;
      varying vec2 v_texCoord;
      uniform float u_time;
      uniform float u_pixelSize;
      uniform sampler2D u_texture;
      
      // NES-style color palette
      vec3 nesColor(vec3 color) {
        // Quantize to NES-like palette
        color = floor(color * 4.0) / 4.0;
        
        // Apply NES color constraints
        if (color.r > 0.8) color.r = 1.0;
        else if (color.r > 0.4) color.r = 0.6;
        else if (color.r > 0.2) color.r = 0.3;
        else color.r = 0.0;
        
        // Similar quantization for G and B
        if (color.g > 0.8) color.g = 1.0;
        else if (color.g > 0.4) color.g = 0.6;
        else if (color.g > 0.2) color.g = 0.3;
        else color.g = 0.0;
        
        if (color.b > 0.8) color.b = 1.0;
        else if (color.b > 0.4) color.b = 0.6;
        else if (color.b > 0.2) color.b = 0.3;
        else color.b = 0.0;
        
        return color;
      }
      
      void main() {
        // Pixelate texture coordinates
        vec2 pixelCoord = floor(v_texCoord * u_pixelSize) / u_pixelSize;
        vec3 color = texture2D(u_texture, pixelCoord).rgb;
        
        // Apply NES-style color quantization
        color = nesColor(color);
        
        gl_FragColor = vec4(color, 1.0);
      }
    `,
    
    transition: `
      precision mediump float;
      varying vec2 v_texCoord;
      uniform float u_time;
      uniform float u_progress;
      uniform float u_transitionType; // 0=fade, 1=slide, 2=scale
      
      void main() {
        vec2 uv = v_texCoord;
        vec3 color = vec3(0.0);
        float alpha = 1.0;
        
        if (u_transitionType < 0.5) {
          // Fade transition
          alpha = 1.0 - u_progress;
          color = vec3(0.2, 0.6, 1.0); // Legal blue
        } else if (u_transitionType < 1.5) {
          // Slide transition
          uv.x += u_progress;
          if (uv.x > 1.0) discard;
          color = vec3(0.0, 0.8, 0.3); // Legal green
        } else {
          // Scale transition with glow
          float distance = length(uv - vec2(0.5));
          float glow = (1.0 - distance) * (1.0 - u_progress);
          color = vec3(0.8, 0.4, 0.0) * glow; // Evidence amber
          alpha = glow;
        }
        
        gl_FragColor = vec4(color, alpha);
      }
    `
  };

  async initialize(canvas: HTMLCanvasElement): Promise<boolean> {
    this.canvas = canvas;
    
    try {
      this.gl = canvas.getContext('webgl', {
        alpha: true,
        antialias: false, // Disable for pixel-perfect effects
        depth: false,
        preserveDrawingBuffer: true
      });

      if (!this.gl) {
        console.warn('WebGL not available for GPU animations');
        return false;
      }

      // Configure WebGL for optimal performance
      this.gl.enable(this.gl.BLEND);
      this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
      this.gl.pixelStorei(this.gl.UNPACK_ALIGNMENT, 1);

      // Pre-compile common shaders
      await this.precompileShaders();
      
      console.log('✅ GPU Animation Engine initialized');
      return true;

    } catch (error) {
      console.error('❌ GPU Animation Engine initialization failed:', error);
      return false;
    }
  }

  private async precompileShaders(): Promise<void> {
    if (!this.gl) return;

    const shaderTypes = ['confidence', 'nesPixel', 'transition'];
    
    for (const type of shaderTypes) {
      try {
        const program = await this.createShaderProgram(type);
        if (program) {
          this.shaderCache.set(type, program);
          this.memoryUsed += 512; // Estimate 512 bytes per shader program
        }
      } catch (error) {
        console.warn(`Failed to precompile ${type} shader:`, error);
      }
    }
  }

  private async createShaderProgram(type: string): Promise<WebGLProgram | null> {
    if (!this.gl) return null;

    const vertexSource = this.vertexShaderSources[type as keyof typeof this.vertexShaderSources];
    const fragmentSource = this.fragmentShaderSources[type as keyof typeof this.fragmentShaderSources];

    if (!vertexSource || !fragmentSource) return null;

    const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentSource);

    if (!vertexShader || !fragmentShader) return null;

    const program = this.gl.createProgram();
    if (!program) return null;

    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error('Shader program linking failed:', this.gl.getProgramInfoLog(program));
      this.gl.deleteProgram(program);
      return null;
    }

    return program;
  }

  private compileShader(type: number, source: string): WebGLShader | null {
    if (!this.gl) return null;

    const shader = this.gl.createShader(type);
    if (!shader) return null;

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compilation failed:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  createAnimation(
    element: HTMLElement,
    type: AnimationContext['type'],
    options: {
      duration?: number;
      priority?: number;
      legalContext?: AnimationContext['legalContext'];
      nesStyle?: boolean;
      pixelated?: boolean;
    } = {}
  ): string {
    const id = `anim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Check memory constraints
    if (this.activeAnimations.size >= ANIMATION_MEMORY.MAX_ACTIVE_ANIMATIONS) {
      this.garbageCollectAnimations();
    }

    if (this.memoryUsed > ANIMATION_MEMORY.TOTAL_BUDGET * 0.9) {
      console.warn('⚠️ Animation memory budget exceeded, skipping animation');
      return '';
    }

    const animation: AnimationContext = {
      element,
      id,
      type,
      priority: options.priority || 1,
      duration: options.duration || 1000,
      legalContext: options.legalContext,
      nesStyle: options.nesStyle || false,
      pixelated: options.pixelated || false,
      shader: this.shaderCache.get(type)
    };

    this.activeAnimations.set(id, animation);
    this.memoryUsed += 256; // Estimate 256 bytes per animation

    // Start animation if this is the first one
    if (this.activeAnimations.size === 1) {
      this.startRenderLoop();
    }

    return id;
  }

  private garbageCollectAnimations(): void {
    const animations = Array.from(this.activeAnimations.entries());
    
    // Sort by priority (low priority first)
    animations.sort((a, b) => a[1].priority - b[1].priority);
    
    // Remove lowest priority animations
    const toRemove = Math.ceil(animations.length * 0.3);
    for (let i = 0; i < toRemove; i++) {
      const [id, animation] = animations[i];
      this.stopAnimation(id);
    }
  }

  private startRenderLoop(): void {
    if (this.animationFrame) return;

    const render = (currentTime: number) => {
      if (this.activeAnimations.size === 0) {
        this.animationFrame = 0;
        return;
      }

      this.renderFrame(currentTime);
      this.animationFrame = requestAnimationFrame(render);
    };

    this.animationFrame = requestAnimationFrame(render);
  }

  private renderFrame(currentTime: number): void {
    if (!this.gl || !this.canvas) return;

    // Clear canvas
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    // Render each active animation
    for (const [id, animation] of this.activeAnimations) {
      try {
        this.renderAnimation(animation, currentTime);
      } catch (error) {
        console.warn(`Animation ${id} render failed:`, error);
        this.stopAnimation(id);
      }
    }
  }

  private renderAnimation(animation: AnimationContext, currentTime: number): void {
    if (!this.gl || !animation.shader) return;

    this.gl.useProgram(animation.shader);

    // Set common uniforms
    const timeUniform = this.gl.getUniformLocation(animation.shader, 'u_time');
    if (timeUniform) {
      this.gl.uniform1f(timeUniform, currentTime * 0.001);
    }

    // Set animation-specific uniforms
    switch (animation.type) {
      case 'confidence':
        this.renderConfidenceAnimation(animation);
        break;
      case 'risk':
        this.renderRiskAnimation(animation);
        break;
      case 'nes-pixel':
        this.renderNESPixelAnimation(animation);
        break;
      case 'transition':
        this.renderTransitionAnimation(animation, currentTime);
        break;
    }

    // Draw quad
    this.drawQuad();
  }

  private renderConfidenceAnimation(animation: AnimationContext): void {
    if (!this.gl || !animation.shader || !animation.legalContext) return;

    const confidenceUniform = this.gl.getUniformLocation(animation.shader, 'u_confidence');
    const riskUniform = this.gl.getUniformLocation(animation.shader, 'u_riskLevel');

    if (confidenceUniform) {
      this.gl.uniform1f(confidenceUniform, animation.legalContext.confidence);
    }

    if (riskUniform) {
      const riskValue = animation.legalContext.riskLevel === 'high' ? 2.0 : 
                       animation.legalContext.riskLevel === 'medium' ? 1.0 : 0.0;
      this.gl.uniform1f(riskUniform, riskValue);
    }

    // Apply visual effect to DOM element
    const opacity = 0.3 + (animation.legalContext.confidence * 0.7);
    const glowColor = animation.legalContext.riskLevel === 'high' ? 'rgba(255, 50, 50, 0.6)' :
                     animation.legalContext.riskLevel === 'medium' ? 'rgba(255, 200, 50, 0.6)' :
                     'rgba(50, 255, 100, 0.6)';

    animation.element.style.boxShadow = `0 0 ${10 + animation.legalContext.confidence * 20}px ${glowColor}`;
    animation.element.style.opacity = opacity.toString();
  }

  private renderNESPixelAnimation(animation: AnimationContext): void {
    if (!this.gl || !animation.shader) return;

    const pixelSizeUniform = this.gl.getUniformLocation(animation.shader, 'u_pixelSize');
    if (pixelSizeUniform) {
      this.gl.uniform1f(pixelSizeUniform, animation.pixelated ? 32.0 : 64.0);
    }

    // Apply pixelated effect to DOM element
    animation.element.style.imageRendering = 'pixelated';
    animation.element.style.imageRendering = 'crisp-edges';
  }

  private renderRiskAnimation(animation: AnimationContext): void {
    if (!animation.legalContext) return;

    const intensity = animation.legalContext.riskLevel === 'high' ? 1.0 :
                     animation.legalContext.riskLevel === 'medium' ? 0.6 : 0.3;

    // Pulsing border for risk indication
    const borderWidth = Math.floor(2 + intensity * 3);
    const borderColor = animation.legalContext.riskLevel === 'high' ? '#ef4444' :
                       animation.legalContext.riskLevel === 'medium' ? '#f59e0b' : '#10b981';

    animation.element.style.border = `${borderWidth}px solid ${borderColor}`;
    animation.element.style.animation = `pulse ${1000 / intensity}ms infinite`;
  }

  private renderTransitionAnimation(animation: AnimationContext, currentTime: number): void {
    if (!this.gl || !animation.shader) return;

    const progress = Math.min((currentTime % animation.duration) / animation.duration, 1.0);
    const progressUniform = this.gl.getUniformLocation(animation.shader, 'u_progress');
    
    if (progressUniform) {
      this.gl.uniform1f(progressUniform, progress);
    }

    // Apply CSS transform for hardware acceleration
    const scale = 1.0 - (progress * 0.1);
    animation.element.style.transform = `scale(${scale})`;
  }

  private drawQuad(): void {
    if (!this.gl) return;

    // Create simple quad vertices
    const vertices = new Float32Array([
      -1, -1,  0, 0,
       1, -1,  1, 0,
      -1,  1,  0, 1,
       1,  1,  1, 1
    ]);

    const buffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

    // Set up attributes
    this.gl.enableVertexAttribArray(0);
    this.gl.enableVertexAttribArray(1);
    this.gl.vertexAttribPointer(0, 2, this.gl.FLOAT, false, 16, 0);
    this.gl.vertexAttribPointer(1, 2, this.gl.FLOAT, false, 16, 8);

    // Draw
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    // Clean up
    this.gl.deleteBuffer(buffer);
  }

  stopAnimation(id: string): void {
    const animation = this.activeAnimations.get(id);
    if (!animation) return;

    // Reset element styles
    animation.element.style.boxShadow = '';
    animation.element.style.opacity = '';
    animation.element.style.transform = '';
    animation.element.style.border = '';
    animation.element.style.animation = '';
    animation.element.style.imageRendering = '';

    this.activeAnimations.delete(id);
    this.memoryUsed -= 256;

    // Stop render loop if no animations
    if (this.activeAnimations.size === 0 && this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = 0;
    }
  }

  getMemoryStats() {
    return {
      used: this.memoryUsed,
      budget: ANIMATION_MEMORY.TOTAL_BUDGET,
      utilization: this.memoryUsed / ANIMATION_MEMORY.TOTAL_BUDGET,
      activeAnimations: this.activeAnimations.size,
      maxAnimations: ANIMATION_MEMORY.MAX_ACTIVE_ANIMATIONS,
      cachedShaders: this.shaderCache.size
    };
  }

  destroy(): void {
    // Stop all animations
    for (const id of this.activeAnimations.keys()) {
      this.stopAnimation(id);
    }

    // Clean up WebGL resources
    if (this.gl) {
      for (const program of this.shaderCache.values()) {
        this.gl.deleteProgram(program);
      }
    }

    this.shaderCache.clear();
    this.memoryUsed = 0;

    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = 0;
    }
  }
}

// Export singleton instance
export const gpuAnimations = new GPUAnimationEngine();