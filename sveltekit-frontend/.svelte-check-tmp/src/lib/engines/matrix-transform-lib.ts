
/**
 * Lightweight Matrix Transform Library (~10KB)
 * Optimized for CSS transforms and WebGL integration
 * Inspired by gl-matrix but focused on sprite transformations
 */

export interface MatrixTransformConfig {
  enableGPUAcceleration: boolean;
  optimizeForCSS: boolean;
  cacheTransforms: boolean;
}

export interface Transform2D {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number; // radians
  skewX: number;
  skewY: number;
}

export interface TransformResult {
  css3d?: string;
  webgl?: Float32Array;
  matrix2d?: number[];
}

export class MatrixTransformLib {
  private config: MatrixTransformConfig;
  private transformCache: Map<string, TransformResult> = new Map();

  // Pre-computed constants for performance
  private static readonly DEG_TO_RAD = Math.PI / 180;
  private static readonly RAD_TO_DEG = 180 / Math.PI;

  constructor(config: MatrixTransformConfig) {
    this.config = config;
  }

  /**
   * Generate CSS transforms from sprite JSON state
   */
  public generateCSSTransforms(spriteJsonState: string): TransformResult {
    const cacheKey = this.getCacheKey(spriteJsonState);

    if (this.config.cacheTransforms && this.transformCache.has(cacheKey)) {
      return this.transformCache.get(cacheKey)!;
    }

    try {
      const spriteData = JSON.parse(spriteJsonState);
      const transform = this.extractTransformFromSprite(spriteData);
      const result = this.computeTransforms(transform);

      if (this.config.cacheTransforms) {
        this.transformCache.set(cacheKey, result);
        this.evictOldTransforms();
      }

      return result;
    } catch (error: any) {
      console.warn("Failed to generate transforms from sprite:", error);
      return {};
    }
  }

  /**
   * Create transform from individual components
   */
  public createTransform(
    x = 0,
    y = 0,
    scaleX = 1,
    scaleY = 1,
    rotation = 0,
    skewX = 0,
    skewY = 0,
  ): TransformResult {
    const transform: Transform2D = {
      x,
      y,
      scaleX,
      scaleY,
      rotation,
      skewX,
      skewY,
    };
    return this.computeTransforms(transform);
  }

  /**
   * Multiply two 2D transformation matrices
   */
  public multiplyMatrices(a: number[], b: number[]): number[] {
    // 2D transformation matrix multiplication (3x3 homogeneous coordinates)
    // [ a b c ]   [ g h i ]
    // [ d e f ] × [ j k l ]
    // [ 0 0 1 ]   [ 0 0 1 ]

    return [
      a[0] * b[0] + a[1] * b[3], // a11
      a[0] * b[1] + a[1] * b[4], // a12
      a[0] * b[2] + a[1] * b[5] + a[2], // a13
      a[3] * b[0] + a[4] * b[3], // a21
      a[3] * b[1] + a[4] * b[4], // a22
      a[3] * b[2] + a[4] * b[5] + a[5], // a23
    ];
  }

  /**
   * Convert 2D matrix to CSS transform string
   */
  public matrixToCSS(matrix: number[]): string {
    if (this.config.optimizeForCSS) {
      // Use CSS3 matrix() function for hardware acceleration
      return `matrix(${matrix.map((v) => this.roundToPrecision(v, 6)).join(",")})`;
    }

    // Fallback to individual transform functions
    const { x, y, scaleX, scaleY, rotation } = this.matrixToTransform(matrix);
    const parts: string[] = [];

    if (x !== 0 || y !== 0) {
      parts.push(
        `translate(${this.roundToPrecision(x, 2)}px, ${this.roundToPrecision(y, 2)}px)`,
      );
    }

    if (scaleX !== 1 || scaleY !== 1) {
      parts.push(
        `scale(${this.roundToPrecision(scaleX, 4)}, ${this.roundToPrecision(scaleY, 4)})`,
      );
    }

    if (rotation !== 0) {
      parts.push(
        `rotate(${this.roundToPrecision(rotation * MatrixTransformLib.RAD_TO_DEG, 2)}deg)`,
      );
    }

    return parts.join(" ");
  }

  /**
   * Convert 2D matrix to WebGL-compatible 4x4 matrix
   */
  public matrixToWebGL(matrix: number[]): Float32Array {
    // Convert 2D transformation matrix to 4x4 WebGL matrix
    return new Float32Array([
      matrix[0],
      matrix[3],
      0,
      0, // Column 1
      matrix[1],
      matrix[4],
      0,
      0, // Column 2
      0,
      0,
      1,
      0, // Column 3
      matrix[2],
      matrix[5],
      0,
      1, // Column 4
    ]);
  }

  /**
   * Interpolate between two transforms (for animations)
   */
  public interpolateTransforms(
    from: Transform2D,
    to: Transform2D,
    t: number, // 0 to 1
  ): TransformResult {
    t = Math.max(0, Math.min(1, t)); // Clamp to 0-1

    const interpolated: Transform2D = {
      x: this.lerp(from.x, to.x, t),
      y: this.lerp(from.y, to.y, t),
      scaleX: this.lerp(from.scaleX, to.scaleX, t),
      scaleY: this.lerp(from.scaleY, to.scaleY, t),
      rotation: this.lerpAngle(from.rotation, to.rotation, t),
      skewX: this.lerp(from.skewX, to.skewX, t),
      skewY: this.lerp(from.skewY, to.skewY, t),
    };

    return this.computeTransforms(interpolated);
  }

  private extractTransformFromSprite(spriteData: any): Transform2D {
    // Extract transformation data from Fabric.js canvas JSON
    const objects = spriteData.objects || [];

    if (objects.length === 0) {
      return {
        x: 0,
        y: 0,
        scaleX: 1,
        scaleY: 1,
        rotation: 0,
        skewX: 0,
        skewY: 0,
      };
    }

    // For legal documents, often we want the transform of the primary object
    const primaryObject = objects[0];

    return {
      x: primaryObject.left || 0,
      y: primaryObject.top || 0,
      scaleX: primaryObject.scaleX || 1,
      scaleY: primaryObject.scaleY || 1,
      rotation: (primaryObject.angle || 0) * MatrixTransformLib.DEG_TO_RAD,
      skewX: primaryObject.skewX || 0,
      skewY: primaryObject.skewY || 0,
    };
  }

  private computeTransforms(transform: Transform2D): TransformResult {
    const matrix = this.transformTo2DMatrix(transform);
    const result: TransformResult = {
      matrix2d: matrix,
    };

    if (this.config.optimizeForCSS) {
      result.css3d = this.matrixToCSS(matrix);
    }

    if (this.config.enableGPUAcceleration) {
      result.webgl = this.matrixToWebGL(matrix);
    }

    return result;
  }

  private transformTo2DMatrix(transform: Transform2D): number[] {
    const { x, y, scaleX, scaleY, rotation, skewX, skewY } = transform;

    // Create individual transformation matrices
    const translateMatrix = [1, 0, x, 0, 1, y];
    const scaleMatrix = [scaleX, 0, 0, 0, scaleY, 0];
    const rotationMatrix = [
      Math.cos(rotation),
      -Math.sin(rotation),
      0,
      Math.sin(rotation),
      Math.cos(rotation),
      0,
    ];
    const skewMatrix = [1, Math.tan(skewY), 0, Math.tan(skewX), 1, 0];

    // Multiply matrices in correct order: Translate * Rotate * Scale * Skew
    let result = this.multiplyMatrices(scaleMatrix, skewMatrix);
    result = this.multiplyMatrices(rotationMatrix, result);
    result = this.multiplyMatrices(translateMatrix, result);

    return result;
  }

  private matrixToTransform(matrix: number[]): Transform2D {
    // Decompose 2D transformation matrix back to components
    const [a, b, c, d, e, f] = matrix;

    // Extract translation
    const x = c;
    const y = f;

    // Extract scale and rotation
    const scaleX = Math.sqrt(a * a + b * b);
    const scaleY = Math.sqrt(d * d + e * e);
    const rotation = Math.atan2(b, a);

    // Extract skew
    const skewX = Math.atan2(a * d + b * e, scaleX * scaleX);
    const skewY = 0; // Simplified for most use cases

    return { x, y, scaleX, scaleY, rotation, skewX, skewY };
  }

  private lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }

  private lerpAngle(a: number, b: number, t: number): number {
    // Handle angle interpolation (shortest path)
    let diff = b - a;
    if (diff > Math.PI) {
      diff -= 2 * Math.PI;
    } else if (diff < -Math.PI) {
      diff += 2 * Math.PI;
    }
    return a + diff * t;
  }

  private roundToPrecision(value: number, decimals: number): number {
    const factor = Math.pow(10, decimals);
    return Math.round(value * factor) / factor;
  }

  private getCacheKey(data: string): string {
    // Simple hash for cache key (djb2 algorithm)
    let hash = 5381;
    for (let i = 0; i < data.length; i++) {
      hash = (hash << 5) + hash + data.charCodeAt(i);
    }
    return hash.toString(16);
  }

  private evictOldTransforms(): void {
    // Keep cache size under control (max 100 entries)
    if (this.transformCache.size > 100) {
      const keysToDelete = Array.from(this.transformCache.keys()).slice(0, 20);
      keysToDelete.forEach((key) => this.transformCache.delete(key));
    }
  }

  /**
   * Utility methods for common transformations
   */
  public static createTranslation(x: number, y: number): number[] {
    return [1, 0, x, 0, 1, y];
  }

  public static createScale(scaleX: number, scaleY: number = scaleX): number[] {
    return [scaleX, 0, 0, 0, scaleY, 0];
  }

  public static createRotation(angle: number): number[] {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return [cos, -sin, 0, sin, cos, 0];
  }

  public static createSkew(skewX: number, skewY: number): number[] {
    return [1, Math.tan(skewY), 0, Math.tan(skewX), 1, 0];
  }

  /**
   * Get performance statistics
   */
  public getStats(): {
    cacheSize: number;
    cacheEnabled: boolean;
    gpuAcceleration: boolean;
    cssOptimization: boolean;
  } {
    return {
      cacheSize: this.transformCache.size,
      cacheEnabled: this.config.cacheTransforms,
      gpuAcceleration: this.config.enableGPUAcceleration,
      cssOptimization: this.config.optimizeForCSS,
    };
  }

  /**
   * Clear transform cache
   */
  public clearCache(): void {
    this.transformCache.clear();
  }
}
