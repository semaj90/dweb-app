
/**
 * WebGL2 Shader Cache with NVIDIA Optimizations
 * Optimized for legal AI canvas operations with sprite-based rendering
 */

export interface ShaderCacheConfig {
  enableNVIDIAOptimizations: boolean;
  cacheSize: number;
  persistToDisk: boolean;
}

export interface ShaderProgram {
  id: string;
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation>;
  attributes: Record<string, number>;
  lastUsed: number;
  complexity: number;
}

export interface SpriteTransforms {
  webgl?: {
    matrix: Float32Array;
    opacity: number;
    blend: string;
  };
  css3d?: string;
}

export class ShaderCache {
  private gl: WebGL2RenderingContext;
  private config: ShaderCacheConfig;
  private programs: Map<string, ShaderProgram> = new Map();
  private persistentCache: Map<string, string> = new Map();

  // NVIDIA-optimized shaders for sprite operations
  private readonly spriteVertexShader = `#version 300 es
    precision highp float;
    
    // Vertex attributes
    in vec2 a_position;
    in vec2 a_texCoord;
    
    // Uniforms for sprite transforms
    uniform mat4 u_transform;
    uniform mat4 u_view;
    uniform mat4 u_projection;
    uniform float u_opacity;
    
    // Outputs to fragment shader
    out vec2 v_texCoord;
    out float v_opacity;
    
    void main() {
      // Apply sprite transformation matrix
      vec4 worldPosition = u_transform * vec4(a_position, 0.0, 1.0);
      vec4 viewPosition = u_view * worldPosition;
      gl_Position = u_projection * viewPosition;
      
      // Pass texture coordinates and opacity
      v_texCoord = a_texCoord;
      v_opacity = u_opacity;
    }
  `;

  private readonly spriteFragmentShader = `#version 300 es
    precision highp float;
    
    // Inputs from vertex shader
    in vec2 v_texCoord;
    in float v_opacity;
    
    // Uniforms
    uniform sampler2D u_texture;
    uniform vec4 u_tintColor;
    uniform float u_brightness;
    uniform float u_contrast;
    
    // Output
    out vec4 fragColor;
    
    void main() {
      // Sample sprite texture
      vec4 texColor = texture(u_texture, v_texCoord);
      
      // Apply legal AI specific color adjustments
      texColor.rgb = mix(texColor.rgb, u_tintColor.rgb, u_tintColor.a);
      texColor.rgb = ((texColor.rgb - 0.5) * u_contrast) + 0.5 + u_brightness;
      
      // Apply opacity
      texColor.a *= v_opacity;
      
      fragColor = texColor;
    }
  `;

  constructor(gl: WebGL2RenderingContext, config: ShaderCacheConfig) {
    this.gl = gl;
    this.config = config;
    this.initializeCache();
    this.precompileCommonShaders();
  }

  private initializeCache(): void {
    if (this.config.persistToDisk) {
      try {
        const cached = localStorage.getItem("neural-sprite-shaders");
        if (cached) {
          const parsed = JSON.parse(cached);
          this.persistentCache = new Map(parsed);
        }
      } catch (error: any) {
        console.warn("Failed to load shader cache from storage:", error);
      }
    }
  }

  private precompileCommonShaders(): void {
    // Pre-compile essential sprite shaders for NVIDIA optimization
    this.createProgram(
      "sprite-basic",
      this.spriteVertexShader,
      this.spriteFragmentShader,
    );

    // Legal document specific shader (high contrast, sharp text)
    const legalFragmentShader = this.spriteFragmentShader.replace(
      "texColor.rgb = ((texColor.rgb - 0.5) * u_contrast) + 0.5 + u_brightness;",
      "texColor.rgb = clamp(((texColor.rgb - 0.5) * 1.5) + 0.5, 0.0, 1.0);", // Higher contrast for legal docs
    );

    this.createProgram(
      "sprite-legal",
      this.spriteVertexShader,
      legalFragmentShader,
    );

    // Evidence annotation shader (with highlighting)
    const evidenceFragmentShader = `#version 300 es
      precision highp float;
      
      in vec2 v_texCoord;
      in float v_opacity;
      
      uniform sampler2D u_texture;
      uniform vec4 u_highlightColor;
      uniform float u_highlightIntensity;
      
      out vec4 fragColor;
      
      void main() {
        vec4 texColor = texture(u_texture, v_texCoord);
        
        // Add evidence highlighting with yellow tint
        vec3 highlight = mix(texColor.rgb, u_highlightColor.rgb, u_highlightIntensity);
        texColor.rgb = highlight;
        texColor.a *= v_opacity;
        
        fragColor = texColor;
      }
    `;

    this.createProgram(
      "sprite-evidence",
      this.spriteVertexShader,
      evidenceFragmentShader,
    );
  }

  public createProgram(
    id: string,
    vertexSource: string,
    fragmentSource: string,
  ): WebGLProgram | null {
    // Check cache first
    if (this.programs.has(id)) {
      const cached = this.programs.get(id)!;
      cached.lastUsed = Date.now();
      return cached.program;
    }

    // Check persistent cache
    const cacheKey = this.generateCacheKey(vertexSource, fragmentSource);
    if (this.persistentCache.has(cacheKey)) {
      // TODO: WebGL doesn't support direct program serialization
      // This would require WebGL program binary extension
      console.log(`Shader cache hit for ${id}`);
    }

    const vertexShader = this.compileShader(
      vertexSource,
      this.gl.VERTEX_SHADER,
    );
    const fragmentShader = this.compileShader(
      fragmentSource,
      this.gl.FRAGMENT_SHADER,
    );

    if (!vertexShader || !fragmentShader) {
      return null;
    }

    const program = this.gl.createProgram();
    if (!program) {
      return null;
    }

    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);

    // NVIDIA optimization: Bind attribute locations before linking
    this.gl.bindAttribLocation(program, 0, "a_position");
    this.gl.bindAttribLocation(program, 1, "a_texCoord");

    this.gl.linkProgram(program);

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error(
        "Shader program linking failed:",
        this.gl.getProgramInfoLog(program),
      );
      this.gl.deleteProgram(program);
      return null;
    }

    // Cache uniform and attribute locations
    const uniforms: Record<string, WebGLUniformLocation> = {};
    const attributes: Record<string, number> = {};

    const uniformCount = this.gl.getProgramParameter(
      program,
      this.gl.ACTIVE_UNIFORMS,
    );
    for (let i = 0; i < uniformCount; i++) {
      const info = this.gl.getActiveUniform(program, i);
      if (info) {
        const location = this.gl.getUniformLocation(program, info.name);
        if (location) {
          uniforms[info.name] = location;
        }
      }
    }

    const attributeCount = this.gl.getProgramParameter(
      program,
      this.gl.ACTIVE_ATTRIBUTES,
    );
    for (let i = 0; i < attributeCount; i++) {
      const info = this.gl.getActiveAttrib(program, i);
      if (info) {
        attributes[info.name] = this.gl.getAttribLocation(program, info.name);
      }
    }

    const shaderProgram: ShaderProgram = {
      id,
      program,
      uniforms,
      attributes,
      lastUsed: Date.now(),
      complexity: this.calculateComplexity(vertexSource, fragmentSource),
    };

    this.programs.set(id, shaderProgram);
    this.evictOldPrograms();

    // Clean up individual shaders
    this.gl.deleteShader(vertexShader);
    this.gl.deleteShader(fragmentShader);

    return program;
  }

  public async precompileForSprite(sprite: any): Promise<void> {
    // Determine optimal shader based on sprite metadata
    let shaderId = "sprite-basic";

    if (sprite.metadata?.triggers?.includes("legal_document")) {
      shaderId = "sprite-legal";
    } else if (sprite.metadata?.triggers?.includes("evidence_annotation")) {
      shaderId = "sprite-evidence";
    }

    // Ensure shader is compiled and warmed up
    const program = this.programs.get(shaderId);
    if (program) {
      // Warm up the GPU by using the program briefly
      this.gl.useProgram(program.program);
      program.lastUsed = Date.now();
    }
  }

  public applyTransforms(
    transforms: NonNullable<SpriteTransforms["webgl"]>,
  ): void {
    const currentProgram = this.getCurrentProgram();
    if (!currentProgram) return;

    // Apply matrix transform
    const matrixLocation = currentProgram.uniforms["u_transform"];
    if (matrixLocation) {
      this.gl.uniformMatrix4fv(matrixLocation, false, transforms.matrix);
    }

    // Apply opacity
    const opacityLocation = currentProgram.uniforms["u_opacity"];
    if (opacityLocation) {
      this.gl.uniform1f(opacityLocation, transforms.opacity);
    }

    // Set blend mode
    if (transforms.blend === "multiply") {
      this.gl.blendFunc(this.gl.DST_COLOR, this.gl.ZERO);
    } else if (transforms.blend === "screen") {
      this.gl.blendFunc(this.gl.ONE_MINUS_DST_COLOR, this.gl.ONE);
    } else {
      this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    }
  }

  private compileShader(source: string, type: number): WebGLShader | null {
    const shader = this.gl.createShader(type);
    if (!shader) {
      return null;
    }

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error(
        "Shader compilation failed:",
        this.gl.getShaderInfoLog(shader),
      );
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  private getCurrentProgram(): ShaderProgram | null {
    // Find the currently active program
    const programs = Array.from(this.programs.values());
    for (let i = 0; i < programs.length; i++) {
      const program = programs[i];
      if (this.gl.getParameter(this.gl.CURRENT_PROGRAM) === program.program) {
        return program;
      }
    }
    return null;
  }

  private generateCacheKey(
    vertexSource: string,
    fragmentSource: string,
  ): string {
    // Simple hash for cache key
    return btoa(vertexSource + fragmentSource).substring(0, 16);
  }

  private calculateComplexity(
    vertexSource: string,
    fragmentSource: string,
  ): number {
    // Simple complexity calculation based on shader operations
    const operations = (vertexSource + fragmentSource).match(
      /(\*|\/|\+|\-|texture|mix|clamp)/g,
    );
    return operations ? operations.length : 1;
  }

  private evictOldPrograms(): void {
    if (this.programs.size <= this.config.cacheSize) {
      return;
    }

    // Sort by last used time and remove oldest
    const sorted = Array.from(this.programs.entries()).sort(
      ([, a], [, b]) => a.lastUsed - b.lastUsed,
    );

    const toRemove = sorted.slice(0, sorted.length - this.config.cacheSize);

    for (const [id, program] of toRemove) {
      this.gl.deleteProgram(program.program);
      this.programs.delete(id);
    }
  }

  public destroy(): void {
    // Clean up all programs
    const programs = Array.from(this.programs.values());
    for (let i = 0; i < programs.length; i++) {
      this.gl.deleteProgram(programs[i].program);
    }
    this.programs.clear();

    // Save persistent cache
    if (this.config.persistToDisk) {
      try {
        localStorage.setItem(
          "neural-sprite-shaders",
          JSON.stringify(Array.from(this.persistentCache.entries())),
        );
      } catch (error: any) {
        console.warn("Failed to save shader cache to storage:", error);
      }
    }
  }

  public getStats(): {
    programCount: number;
    cacheHits: number;
    memoryUsage: number;
  } {
    return {
      programCount: this.programs.size,
      cacheHits: this.persistentCache.size,
      memoryUsage: this.programs.size * 1024, // Rough estimate
    };
  }
}
