/**
 * Shader Manager - Handles WGSL shader loading and caching
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { Logger } from './logger';

interface ShaderCache {
  [name: string]: string;
}

interface ShaderConfig {
  name: string;
  path: string;
  entryPoint: string;
  workgroupSize: [number, number, number];
}

export class ShaderManager {
  private cache: ShaderCache = {};
  private logger: Logger;
  private shaderConfigs: ShaderConfig[];

  constructor(logger: Logger) {
    this.logger = logger;
    this.shaderConfigs = [
      {
        name: 'embedding_processor',
        path: '../../lib/shaders/embedding_processor.wgsl',
        entryPoint: 'main',
        workgroupSize: [8, 8, 1]
      },
      {
        name: 'kmeans_clustering',
        path: '../../lib/shaders/kmeans_clustering.wgsl',
        entryPoint: 'main',
        workgroupSize: [64, 1, 1]
      },
      {
        name: 'cosine_similarity',
        path: '../../lib/shaders/cosine_similarity.wgsl',
        entryPoint: 'main',
        workgroupSize: [16, 16, 1]
      },
      {
        name: 'boost_transform',
        path: '../../lib/shaders/boost_transform.wgsl',
        entryPoint: 'main',
        workgroupSize: [64, 1, 1]
      }
    ];
  }

  async initialize(): Promise<void> {
    this.logger.info('🎨 Loading WGSL shaders...');

    for (const config of this.shaderConfigs) {
      try {
        await this.loadShader(config);
        this.logger.debug(`✅ Loaded shader: ${config.name}`);
      } catch (error) {
        this.logger.error(`❌ Failed to load shader ${config.name}:`, error);
        // Create fallback shader if main shader fails
        this.createFallbackShader(config.name);
      }
    }

    this.logger.info(`✅ Loaded ${Object.keys(this.cache).length} shaders`);
  }

  private async loadShader(config: ShaderConfig): Promise<void> {
    const shaderPath = join(__dirname, config.path);
    
    try {
      const shaderCode = readFileSync(shaderPath, 'utf-8');
      this.cache[config.name] = shaderCode;
    } catch (error) {
      // If file doesn't exist, create a basic shader
      if ((error as any).code === 'ENOENT') {
        this.logger.warn(`Shader file not found: ${shaderPath}, creating basic shader`);
        this.cache[config.name] = this.createBasicShader(config.name, config.workgroupSize);
      } else {
        throw error;
      }
    }
  }

  getShader(name: string): string {
    const shader = this.cache[name];
    if (!shader) {
      this.logger.warn(`Shader ${name} not found, creating fallback`);
      return this.createFallbackShader(name);
    }
    return shader;
  }

  private createBasicShader(name: string, workgroupSize: [number, number, number]): string {
    switch (name) {
      case 'embedding_processor':
        return this.createEmbeddingProcessorShader(workgroupSize);
      case 'kmeans_clustering':
        return this.createKMeansShader(workgroupSize);
      case 'cosine_similarity':
        return this.createCosineSimilarityShader(workgroupSize);
      case 'boost_transform':
        return this.createBoostTransformShader(workgroupSize);
      default:
        return this.createFallbackShader(name);
    }
  }

  private createEmbeddingProcessorShader(workgroupSize: [number, number, number]): string {
    return `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_items = arrayLength(&input_data);
    
    if (index >= total_items) {
        return;
    }
    
    // Simple embedding processing - normalize input
    let input_value = input_data[index];
    let normalized = tanh(input_value * 0.1);
    output_data[index] = normalized;
}
`;
  }

  private createKMeansShader(workgroupSize: [number, number, number]): string {
    return `
@group(0) @binding(0) var<storage, read> input_embeddings: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_assignments: array<i32>;
@group(0) @binding(2) var<storage, read> cluster_centers: array<f32>;

const DIMENSIONS: u32 = 384u;
const NUM_CLUSTERS: u32 = 8u;

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;
    let num_points = arrayLength(&input_embeddings) / DIMENSIONS;
    
    if (point_idx >= num_points) {
        return;
    }
    
    var min_distance = 1e10;
    var best_cluster = 0i;
    
    // Find closest cluster
    for (var cluster_idx = 0u; cluster_idx < NUM_CLUSTERS; cluster_idx++) {
        var distance = 0.0;
        
        // Compute Euclidean distance
        for (var dim = 0u; dim < DIMENSIONS; dim++) {
            let point_val = input_embeddings[point_idx * DIMENSIONS + dim];
            let center_val = cluster_centers[cluster_idx * DIMENSIONS + dim];
            let diff = point_val - center_val;
            distance += diff * diff;
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            best_cluster = i32(cluster_idx);
        }
    }
    
    output_assignments[point_idx] = best_cluster;
}
`;
  }

  private createCosineSimilarityShader(workgroupSize: [number, number, number]): string {
    return `
@group(0) @binding(0) var<storage, read> embeddings_a: array<f32>;
@group(0) @binding(1) var<storage, read> embeddings_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> similarity_scores: array<f32>;

const DIMENSIONS: u32 = 384u;

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    
    let num_a = arrayLength(&embeddings_a) / DIMENSIONS;
    let num_b = arrayLength(&embeddings_b) / DIMENSIONS;
    
    if (i >= num_a || j >= num_b) {
        return;
    }
    
    var dot_product = 0.0;
    var norm_a = 0.0;
    var norm_b = 0.0;
    
    // Compute cosine similarity
    for (var dim = 0u; dim < DIMENSIONS; dim++) {
        let val_a = embeddings_a[i * DIMENSIONS + dim];
        let val_b = embeddings_b[j * DIMENSIONS + dim];
        
        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }
    
    let similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b));
    similarity_scores[i * num_b + j] = similarity;
}
`;
  }

  private createBoostTransformShader(workgroupSize: [number, number, number]): string {
    return `
@group(0) @binding(0) var<storage, read> input_embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> boost_factors: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_embeddings: array<f32>;

const DIMENSIONS: u32 = 384u;

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let embedding_idx = global_id.x;
    let dim_idx = global_id.y;
    
    let num_embeddings = arrayLength(&input_embeddings) / DIMENSIONS;
    
    if (embedding_idx >= num_embeddings || dim_idx >= DIMENSIONS) {
        return;
    }
    
    let index = embedding_idx * DIMENSIONS + dim_idx;
    let boost_idx = dim_idx % arrayLength(&boost_factors);
    
    let input_val = input_embeddings[index];
    let boost_factor = boost_factors[boost_idx];
    
    // Apply 4D boost transform
    let boosted_val = input_val * boost_factor;
    let transformed_val = tanh(boosted_val); // Apply activation
    
    output_embeddings[index] = transformed_val;
}
`;
  }

  private createFallbackShader(name: string): string {
    return `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    // Fallback: pass through input to output
    output_data[index] = input_data[index];
}
`;
  }

  reloadShader(name: string): boolean {
    try {
      const config = this.shaderConfigs.find(c => c.name === name);
      if (!config) {
        this.logger.error(`Shader config not found: ${name}`);
        return false;
      }

      this.loadShader(config);
      this.logger.info(`✅ Reloaded shader: ${name}`);
      return true;
    } catch (error) {
      this.logger.error(`❌ Failed to reload shader ${name}:`, error);
      return false;
    }
  }

  listShaders(): string[] {
    return Object.keys(this.cache);
  }

  getShaderInfo(name: string): ShaderConfig | null {
    return this.shaderConfigs.find(c => c.name === name) || null;
  }

  validateShader(shaderCode: string): boolean {
    try {
      // Basic validation - check for required keywords
      const requiredKeywords = ['@compute', '@workgroup_size', 'fn main'];
      return requiredKeywords.every(keyword => shaderCode.includes(keyword));
    } catch {
      return false;
    }
  }

  cleanup(): void {
    this.cache = {};
    this.logger.info('✅ Shader cache cleared');
  }
}