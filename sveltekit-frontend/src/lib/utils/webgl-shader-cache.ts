
/**
 * WebGL Shader Cache and Optimization System
 * High-performance shader compilation and caching for legal AI visualizations
 */

import { writable, type Writable } from 'svelte/store';
// Orphaned content: import type { ComprehensiveCachingArchitecture
// Minimal placeholder interface to prevent type errors if not imported from elsewhere
interface ComprehensiveCachingArchitecture {
  set(key: string, value: any, options?: { ttl?: number; tags?: string[]; layers?: string[] }): Promise<void>;
}

// Shader definitions for legal AI visualizations
export const LEGAL_AI_SHADERS = {
  // Attention weight visualization
  attentionHeatmap: {
    vertex: `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      attribute float a_attention;

      uniform mat4 u_matrix;
      uniform float u_time;
      uniform float u_scale;

      varying vec2 v_texCoord;
      varying float v_attention;
      varying float v_pulse;

      void main() {
        v_texCoord = a_texCoord;
        v_attention = a_attention;

        // Pulsing effect for high attention areas
        v_pulse = sin(u_time * 3.0 + a_attention * 10.0) * 0.1 + 0.9;

        // Scale vertices based on attention weight
        vec2 scaledPosition = a_position * (1.0 + a_attention * u_scale);

        gl_Position = u_matrix * vec4(scaledPosition, 0.0, 1.0);
      }
    `,
    fragment: `
      precision mediump float;

      varying vec2 v_texCoord;
      varying float v_attention;
      varying float v_pulse;

      uniform float u_time;
      uniform vec3 u_lowColor;
      uniform vec3 u_highColor;
      uniform float u_intensity;

      void main() {
        // Create attention heatmap colors
        vec3 color = mix(u_lowColor, u_highColor, v_attention);

        // Add pulsing effect
        color *= v_pulse;

        // Add temporal shimmer for active areas
        float shimmer = sin(u_time * 8.0 + v_texCoord.x * 20.0 + v_texCoord.y * 15.0) * 0.05 + 0.95;
        color *= shimmer;

        // Apply intensity
        color *= u_intensity;

        // Soft edges for better visual appeal
        float edge = smoothstep(0.0, 0.1, v_attention) * smoothstep(1.0, 0.9, v_attention);

        gl_FragColor = vec4(color, v_attention * edge);
      }
    `
  },

  // Document similarity network
  documentNetwork: {
    vertex: `
      attribute vec3 a_position;
      attribute vec3 a_color;
      attribute float a_similarity;
      attribute float a_pageRank;

      uniform mat4 u_matrix;
      uniform float u_time;
      uniform float u_nodeSize;

      varying vec3 v_color;
      varying float v_similarity;
      varying float v_pageRank;
      varying float v_glow;

      void main() {
        v_color = a_color;
        v_similarity = a_similarity;
        v_pageRank = a_pageRank;

        // Animate nodes based on PageRank
        v_glow = sin(u_time + a_pageRank * 5.0) * 0.3 + 0.7;

        // Scale nodes by PageRank score
        float nodeScale = u_nodeSize * (1.0 + a_pageRank * 2.0);

        gl_Position = u_matrix * vec4(a_position, 1.0);
        gl_PointSize = nodeScale * v_glow;
      }
    `,
    fragment: `
      precision mediump float;

      varying vec3 v_color;
      varying float v_similarity;
      varying float v_pageRank;
      varying float v_glow;

      uniform float u_alpha;

      void main() {
        // Create circular points
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);

        if (dist > 0.5) {
          discard;
        }

        // Soft edges
        float alpha = smoothstep(0.5, 0.3, dist);

        // Enhance important nodes
        vec3 color = v_color * (1.0 + v_pageRank);

        // Add glow effect
        color += vec3(v_glow * 0.2);

        gl_FragColor = vec4(color, alpha * u_alpha * v_glow);
      }
    `
  },

  // Legal document text flow
  textFlow: {
    vertex: `
      attribute vec2 a_position;
      attribute vec2 a_velocity;
      attribute float a_relevance;
      attribute float a_age;

      uniform mat4 u_matrix;
      uniform float u_time;
      uniform float u_flowSpeed;

      varying float v_relevance;
      varying float v_age;
      varying vec2 v_velocity;

      void main() {
        v_relevance = a_relevance;
        v_age = a_age;
        v_velocity = a_velocity;

        // Animate particles along flow lines
        vec2 flowOffset = a_velocity * u_time * u_flowSpeed;
        vec2 position = a_position + flowOffset;

        // Fade old particles
        float fade = 1.0 - smoothstep(0.0, 1.0, a_age);

        gl_Position = u_matrix * vec4(position, 0.0, 1.0);
        gl_PointSize = 2.0 + a_relevance * 4.0 * fade;
      }
    `,
    fragment: `
      precision mediump float;

      varying float v_relevance;
      varying float v_age;
      varying vec2 v_velocity;

      uniform vec3 u_relevantColor;
      uniform vec3 u_irrelevantColor;

      void main() {
        // Create particle shape
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);

        if (dist > 0.5) {
          discard;
        }

        // Color based on relevance
        vec3 color = mix(u_irrelevantColor, u_relevantColor, v_relevance);

        // Fade with age
        float alpha = (1.0 - v_age) * (1.0 - dist * 2.0);

        gl_FragColor = vec4(color, alpha);
      }
    `
  },

  // Evidence timeline
  evidenceTimeline: {
    vertex: `
      attribute vec2 a_position;
      attribute float a_timestamp;
      attribute float a_importance;
      attribute vec3 a_evidenceColor;

      uniform mat4 u_matrix;
      uniform float u_currentTime;
      uniform float u_timeRange;

      varying float v_importance;
      varying vec3 v_evidenceColor;
      varying float v_timeDistance;

      void main() {
        v_importance = a_importance;
        v_evidenceColor = a_evidenceColor;

        // Calculate time distance from current time
        v_timeDistance = abs(a_timestamp - u_currentTime) / u_timeRange;

        // Scale based on importance and time relevance
        float scale = 1.0 + a_importance * (1.0 - v_timeDistance);

        gl_Position = u_matrix * vec4(a_position * scale, 0.0, 1.0);
      }
    `,
    fragment: `
      precision mediump float;

      varying float v_importance;
      varying vec3 v_evidenceColor;
      varying float v_timeDistance;

      uniform float u_alpha;

      void main() {
        // Fade based on time distance
        float timeFade = 1.0 - smoothstep(0.0, 1.0, v_timeDistance);

        // Enhance important evidence
        vec3 color = v_evidenceColor * (1.0 + v_importance * 0.5);

        // Add urgency glow for recent important evidence
        if (v_importance > 0.7 && v_timeDistance < 0.2) {
          color += vec3(0.3, 0.1, 0.0);
        }

        float alpha = u_alpha * timeFade * (0.5 + v_importance * 0.5);

        gl_FragColor = vec4(color, alpha);
      }
    `
  }
};

export interface ShaderProgram {
  id: string;
  name: string;
  program: WebGLProgram;
  uniforms: Map<string, WebGLUniformLocation>;
  attributes: Map<string, number>;
  compilationTime: number;
  lastUsed: number;
  useCount: number;
}

export interface ShaderCacheMetrics {
  totalShaders: number;
  compiledShaders: number;
  cacheHits: number;
  cacheMisses: number;
  totalCompilationTime: number;
  averageCompilationTime: number;
  memoryUsage: number;
}

/**
 * WebGL Shader Cache Manager
 * Optimized shader compilation and caching for legal AI visualizations
 */
export class WebGLShaderCache {
  private shaderPrograms = new Map<string, ShaderProgram>();
  private compilationQueue: Array<{ id: string; resolve: Function; reject: Function }> = [];
  private isCompiling = false;

  // Performance tracking
  private metrics: Writable<ShaderCacheMetrics>;
  private cacheHits = 0;
  private cacheMisses = 0;

  // Integration with comprehensive caching
  private cacheArchitecture?: ComprehensiveCachingArchitecture;

  constructor(
    private gl: WebGLRenderingContext | WebGL2RenderingContext,
    cacheArchitecture?: ComprehensiveCachingArchitecture
  ) {
    this.cacheArchitecture = cacheArchitecture;
    this.metrics = writable(this.getInitialMetrics());

    // Pre-compile common shaders
    this.precompileCommonShaders();
  }

  /**
   * Get or compile shader program
   */
  public async getShaderProgram(id: string): Promise<ShaderProgram> {
    // Check cache first
    const cached = this.shaderPrograms.get(id);
    if (cached) {
      cached.lastUsed = Date.now();
      cached.useCount++;
      this.cacheHits++;
      this.updateMetrics();
      return cached;
    }

    this.cacheMisses++;

    // Check if shader definition exists
    const shaderDef = this.getShaderDefinition(id);
    if (!shaderDef) {
      throw new Error(`Shader definition not found: ${id}`);
    }

    // Compile shader
    return this.compileShader(id, shaderDef.vertex, shaderDef.fragment);
  }

  /**
   * Compile shader program
   */
  public async compileShader(
    id: string,
    vertexSource: string,
    fragmentSource: string
  ): Promise<ShaderProgram> {

    // Queue compilation to avoid blocking
    return new Promise((resolve, reject) => {
      this.compilationQueue.push({ id, resolve, reject });
      this.processCompilationQueue();
    });
  }

  /**
   * Process shader compilation queue
   */
  private async processCompilationQueue(): Promise<void> {
    if (this.isCompiling || this.compilationQueue.length === 0) {
      return;
    }

    this.isCompiling = true;

    while (this.compilationQueue.length > 0) {
      const { id, resolve, reject } = this.compilationQueue.shift()!;

      try {
        const shaderDef = this.getShaderDefinition(id);
        if (!shaderDef) {
          reject(new Error(`Shader definition not found: ${id}`));
          continue;
        }

        const startTime = Date.now();

        // Compile vertex shader
        const vertexShader = this.compileShaderStage(
          this.gl.VERTEX_SHADER,
          shaderDef.vertex
        );

        // Compile fragment shader
        const fragmentShader = this.compileShaderStage(
          this.gl.FRAGMENT_SHADER,
          shaderDef.fragment
        );

        // Create and link program
        const program = this.gl.createProgram();
        if (!program) {
          throw new Error('Failed to create shader program');
        }

        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);

        // Check linking status
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
          const error = this.gl.getProgramInfoLog(program);
          this.gl.deleteProgram(program);
          throw new Error(`Shader program linking failed: ${error}`);
        }

        // Clean up individual shaders
        this.gl.deleteShader(vertexShader);
        this.gl.deleteShader(fragmentShader);

        const compilationTime = Date.now() - startTime;

        // Extract uniforms and attributes
        const uniforms = this.extractUniforms(program);
        const attributes = this.extractAttributes(program);

        const shaderProgram: ShaderProgram = {
          id,
          name: id,
          program,
          uniforms,
          attributes,
          compilationTime,
          lastUsed: Date.now(),
          useCount: 1
        };

        // Cache the compiled shader
        this.shaderPrograms.set(id, shaderProgram);

        // Cache in comprehensive caching system
        if (this.cacheArchitecture) {
          await this.cacheArchitecture.set(`shader_${id}`, {
            id,
            compilationTime,
            uniforms: Array.from(uniforms.keys()),
            attributes: Array.from(attributes.keys())
          }, {
            ttl: 3600000, // 1 hour
            tags: ['webgl-shader', 'legal-ai'],
            layers: ['loki', 'redis']
          });
        }

        console.log(`✨ Compiled shader '${id}' in ${compilationTime}ms`);

        this.updateMetrics();
        resolve(shaderProgram);

      } catch (error: any) {
        console.error(`Shader compilation failed for '${id}':`, error);
        reject(error);
      }
    }

    this.isCompiling = false;
  }

  /**
   * Compile individual shader stage
   */
  private compileShaderStage(type: number, source: string): WebGLShader {
    const shader = this.gl.createShader(type);
    if (!shader) {
      throw new Error('Failed to create shader');
    }

    // Add precision qualifiers for fragment shaders if missing
    if (type === this.gl.FRAGMENT_SHADER && !source.includes('precision')) {
      source = 'precision mediump float;\n' + source;
    }

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      const error = this.gl.getShaderInfoLog(shader);
      this.gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${error}`);
    }

    return shader;
  }

  /**
   * Extract uniform locations from compiled program
   */
  private extractUniforms(program: WebGLProgram): Map<string, WebGLUniformLocation> {
    const uniforms = new Map<string, WebGLUniformLocation>();

    const uniformCount = this.gl.getProgramParameter(program, this.gl.ACTIVE_UNIFORMS);

    for (let i = 0; i < uniformCount; i++) {
      const uniform = this.gl.getActiveUniform(program, i);
      if (uniform) {
        const location = this.gl.getUniformLocation(program, uniform.name);
        if (location !== null) {
          uniforms.set(uniform.name, location);
        }
      }
    }

    return uniforms;
  }

  /**
   * Extract attribute locations from compiled program
   */
  private extractAttributes(program: WebGLProgram): Map<string, number> {
    const attributes = new Map<string, number>();

    const attributeCount = this.gl.getProgramParameter(program, this.gl.ACTIVE_ATTRIBUTES);

    for (let i = 0; i < attributeCount; i++) {
      const attribute = this.gl.getActiveAttrib(program, i);
      if (attribute) {
        const location = this.gl.getAttribLocation(program, attribute.name);
        if (location !== -1) {
          attributes.set(attribute.name, location);
        }
      }
    }

    return attributes;
  }

  /**
   * Get shader definition by ID
   */
  private getShaderDefinition(id: string): { vertex: string; fragment: string } | null {
    const shaderName = id.replace('legal-ai-', '') as keyof typeof LEGAL_AI_SHADERS;
    return LEGAL_AI_SHADERS[shaderName] || null;
  }

  /**
   * Pre-compile commonly used shaders
   */
  private async precompileCommonShaders(): Promise<void> {
    const commonShaders = [
      'legal-ai-attentionHeatmap',
      'legal-ai-documentNetwork'
    ];

    console.log('🔄 Pre-compiling common shaders...');

    for (const shaderId of commonShaders) {
      try {
        await this.getShaderProgram(shaderId);
      } catch (error: any) {
        console.warn(`Failed to pre-compile shader ${shaderId}:`, error);
      }
    }

    console.log('✅ Common shaders pre-compiled');
  }

  /**
   * Create optimized vertex buffer for legal AI visualizations
   */
  public createVertexBuffer(data: Float32Array, usage: number = this.gl.STATIC_DRAW): WebGLBuffer {
    const buffer = this.gl.createBuffer();
    if (!buffer) {
      throw new Error('Failed to create vertex buffer');
    }

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, data, usage);

    return buffer;
  }

  /**
   * Create optimized index buffer
   */
  public createIndexBuffer(data: Uint16Array | Uint32Array): WebGLBuffer {
    const buffer = this.gl.createBuffer();
    if (!buffer) {
      throw new Error('Failed to create index buffer');
    }

    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, data, this.gl.STATIC_DRAW);

    return buffer;
  }

  /**
   * Optimized uniform setting with caching
   */
  public setUniforms(program: ShaderProgram, uniforms: Record<string, any>): void {
    this.gl.useProgram(program.program);

    for (const [name, value] of Object.entries(uniforms)) {
      const location = program.uniforms.get(name);
      if (location === undefined) {
        console.warn(`Uniform '${name}' not found in shader '${program.id}'`);
        continue;
      }

      // Set uniform based on type
      if (typeof value === 'number') {
        this.gl.uniform1f(location, value);
      } else if (Array.isArray(value)) {
        switch (value.length) {
          case 2:
            this.gl.uniform2fv(location, value);
            break;
          case 3:
            this.gl.uniform3fv(location, value);
            break;
          case 4:
            this.gl.uniform4fv(location, value);
            break;
          case 9:
            this.gl.uniformMatrix3fv(location, false, value);
            break;
          case 16:
            this.gl.uniformMatrix4fv(location, false, value);
            break;
          default:
            console.warn(`Unsupported uniform array length: ${value.length}`);
        }
      }
    }
  }

  /**
   * Setup vertex attributes for legal AI visualizations
   */
  public setupVertexAttributes(
    program: ShaderProgram,
    attributes: Record<string, { buffer: WebGLBuffer; size: number; type?: number; normalized?: boolean; stride?: number; offset?: number }>
  ): void {

    for (const [name, config] of Object.entries(attributes)) {
      const location = program.attributes.get(name);
      if (location === undefined) {
        console.warn(`Attribute '${name}' not found in shader '${program.id}'`);
        continue;
      }

      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, config.buffer);
      this.gl.enableVertexAttribArray(location);
      this.gl.vertexAttribPointer(
        location,
        config.size,
        config.type || this.gl.FLOAT,
        config.normalized || false,
        config.stride || 0,
        config.offset || 0
      );
    }
  }

  /**
   * Render legal AI visualization
   */
  public render(
    program: ShaderProgram,
    uniforms: Record<string, any>,
    attributes: Record<string, any>,
    drawMode: number = this.gl.TRIANGLES,
    count?: number,
    indexBuffer?: WebGLBuffer
  ): void {

    // Use shader program
    this.gl.useProgram(program.program);

    // Set uniforms
    this.setUniforms(program, uniforms);

    // Setup vertex attributes
    this.setupVertexAttributes(program, attributes);

    // Draw
    if (indexBuffer) {
      this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      this.gl.drawElements(drawMode, count || 0, this.gl.UNSIGNED_SHORT, 0);
    } else {
      this.gl.drawArrays(drawMode, 0, count || 0);
    }

    // Update usage stats
    program.lastUsed = Date.now();
    program.useCount++;
  }

  /**
   * Update performance metrics
   */
  private updateMetrics(): void {
    const totalCompilationTime = Array.from(this.shaderPrograms.values())
      .reduce((sum, shader) => sum + shader.compilationTime, 0);

    const metrics: ShaderCacheMetrics = {
      totalShaders: this.shaderPrograms.size,
      compiledShaders: this.shaderPrograms.size,
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      totalCompilationTime,
      averageCompilationTime: this.shaderPrograms.size > 0 ?
        totalCompilationTime / this.shaderPrograms.size : 0,
      memoryUsage: this.estimateMemoryUsage()
    };

    this.metrics.set(metrics);
  }

  /**
   * Estimate memory usage of shader cache
   */
  private estimateMemoryUsage(): number {
    // Rough estimate: each shader program uses ~10KB
    return this.shaderPrograms.size * 10 * 1024;
  }

  /**
   * Get initial metrics
   */
  private getInitialMetrics(): ShaderCacheMetrics {
    return {
      totalShaders: 0,
      compiledShaders: 0,
      cacheHits: 0,
      cacheMisses: 0,
      totalCompilationTime: 0,
      averageCompilationTime: 0,
      memoryUsage: 0
    };
  }

  /**
   * Clean up shader cache
   */
  public cleanup(): void {
    for (const shader of this.shaderPrograms.values()) {
      this.gl.deleteProgram(shader.program);
    }
    this.shaderPrograms.clear();
    this.updateMetrics();
  }

  // Public getters
}

/**
 * Factory function for creating WebGL shader cache
 */
export function createWebGLShaderCache(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  cacheArchitecture?: ComprehensiveCachingArchitecture
): WebGLShaderCache {
  return new WebGLShaderCache(gl, cacheArchitecture);
}