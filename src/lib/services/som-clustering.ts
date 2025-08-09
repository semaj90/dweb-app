// @ts-nocheck
/**
 * Self-Organizing Map (SOM) Implementation for Legal Document Clustering
 * Unsupervised learning for document similarity and topic discovery
 */

import type { SOMConfig, SelfOrganizingMap, DocumentCluster } from '$lib/api/enhanced-rest-architecture';
import { Redis } from 'ioredis';

export class LegalDocumentSOM implements SelfOrganizingMap {
  private neurons: number[][][] = []; // [x][y][dimensions]
  private config: SOMConfig;
  private redis: Redis;
  private trained: boolean = false;
  
  constructor(config: SOMConfig, redis: Redis) {
    this.config = config;
    this.redis = redis;
    this.initializeNeurons();
  }
  
  /**
   * Initialize SOM neural network with random weights
   */
  private initializeNeurons(): void {
    this.neurons = [];
    for (let x = 0; x < this.config.width; x++) {
      this.neurons[x] = [];
      for (let y = 0; y < this.config.height; y++) {
        this.neurons[x][y] = Array(this.config.dimensions)
          .fill(0)
          .map(() => Math.random() * 2 - 1); // Random weights between -1 and 1
      }
    }
  }
  
  /**
   * Train SOM with legal document embeddings
   */
  async train(embeddings: number[][]): Promise<void> {
    console.log(`Training SOM with ${embeddings.length} document embeddings...`);
    
    let learningRate = this.config.learningRate;
    let radius = this.config.radius;
    
    for (let iteration = 0; iteration < this.config.iterations; iteration++) {
      // Decay learning rate and radius over time
      const decayFactor = 1 - (iteration / this.config.iterations);
      const currentLearningRate = learningRate * decayFactor;
      const currentRadius = radius * decayFactor;
      
      // Shuffle embeddings for better training
      const shuffledEmbeddings = this.shuffleArray([...embeddings]);
      
      for (const embedding of shuffledEmbeddings) {
        // Find Best Matching Unit (BMU)
        const bmu = this.findBMU(embedding);
        
        // Update BMU and neighbors
        await this.updateNeighborhood(embedding, bmu, currentLearningRate, currentRadius);
      }
      
      // Save progress to Redis
      if (iteration % 100 === 0) {
        await this.saveTrainingProgress(iteration, currentLearningRate, currentRadius);
      }
    }
    
    this.trained = true;
    await this.saveToRedis();
    console.log('SOM training completed!');
  }
  
  /**
   * Find Best Matching Unit for an embedding
   */
  private findBMU(embedding: number[]): { x: number; y: number; distance: number } {
    let minDistance = Infinity;
    let bmu = { x: 0, y: 0, distance: Infinity };
    
    for (let x = 0; x < this.config.width; x++) {
      for (let y = 0; y < this.config.height; y++) {
        const distance = this.euclideanDistance(embedding, this.neurons[x][y]);
        if (distance < minDistance) {
          minDistance = distance;
          bmu = { x, y, distance };
        }
      }
    }
    
    return bmu;
  }
  
  /**
   * Update BMU and neighboring neurons
   */
  private async updateNeighborhood(
    embedding: number[],
    bmu: { x: number; y: number },
    learningRate: number,
    radius: number
  ): Promise<void> {
    for (let x = 0; x < this.config.width; x++) {
      for (let y = 0; y < this.config.height; y++) {
        const distance = Math.sqrt((x - bmu.x) ** 2 + (y - bmu.y) ** 2);
        
        if (distance <= radius) {
          // Calculate influence based on distance from BMU
          const influence = Math.exp(-(distance ** 2) / (2 * radius ** 2));
          const effectiveLearningRate = learningRate * influence;
          
          // Update neuron weights
          for (let i = 0; i < this.config.dimensions; i++) {
            const delta = embedding[i] - this.neurons[x][y][i];
            this.neurons[x][y][i] += effectiveLearningRate * delta;
          }
        }
      }
    }
  }
  
  /**
   * Cluster a new document embedding
   */
  async cluster(embedding: number[]): Promise<{ x: number; y: number; confidence: number }> {
    if (!this.trained) {
      throw new Error('SOM must be trained before clustering');
    }
    
    const bmu = this.findBMU(embedding);
    
    // Calculate confidence based on distance to BMU
    const maxDistance = Math.sqrt(this.config.dimensions); // Theoretical max
    const confidence = 1 - (bmu.distance / maxDistance);
    
    return {
      x: bmu.x,
      y: bmu.y,
      confidence: Math.max(0, Math.min(1, confidence))
    };
  }
  
  /**
   * Get neighboring neurons within radius
   */
  async getNeighborhood(x: number, y: number, radius: number): Promise<number[][]> {
    const neighbors: number[][] = [];
    
    for (let nx = 0; nx < this.config.width; nx++) {
      for (let ny = 0; ny < this.config.height; ny++) {
        const distance = Math.sqrt((nx - x) ** 2 + (ny - y) ** 2);
        if (distance <= radius) {
          neighbors.push(this.neurons[nx][ny]);
        }
      }
    }
    
    return neighbors;
  }
  
  /**
   * Generate SOM visualization data
   */
  async visualize(): Promise<{ width: number; height: number; neurons: number[][] }> {
    const visualization = {
      width: this.config.width,
      height: this.config.height,
      neurons: [] as number[][]
    };
    
    for (let x = 0; x < this.config.width; x++) {
      for (let y = 0; y < this.config.height; y++) {
        // Calculate activation level (average of weights)
        const activation = this.neurons[x][y].reduce((sum, weight) => sum + Math.abs(weight), 0) / this.config.dimensions;
        visualization.neurons.push([x, y, activation]);
      }
    }
    
    return visualization;
  }
  
  /**
   * Analyze legal document clusters on the SOM
   */
  async analyzeLegalClusters(documents: Array<{ id: string; embedding: number[]; metadata: any }>): Promise<{
    clusters: Array<{
      position: { x: number; y: number };
      documents: string[];
      legalTopics: string[];
      coherence: number;
    }>;
  }> {
    const clusterMap = new Map<string, Array<{ id: string; metadata: any }>>();
    
    // Map documents to SOM positions
    for (const doc of documents) {
      const position = await this.cluster(doc.embedding);
      const key = `${position.x},${position.y}`;
      
      if (!clusterMap.has(key)) {
        clusterMap.set(key, []);
      }
      clusterMap.get(key)!.push({ id: doc.id, metadata: doc.metadata });
    }
    
    // Analyze each cluster
    const clusters = [];
    for (const [positionKey, clusterDocs] of clusterMap.entries()) {
      const [x, y] = positionKey.split(',').map(Number);
      
      // Extract legal topics from metadata
      const legalTopics = this.extractLegalTopics(clusterDocs.map(d => d.metadata));
      
      // Calculate cluster coherence
      const coherence = await this.calculateClusterCoherence(x, y, clusterDocs.length);
      
      clusters.push({
        position: { x, y },
        documents: clusterDocs.map(d => d.id),
        legalTopics,
        coherence
      });
    }
    
    return { clusters };
  }
  
  /**
   * Extract legal topics from document metadata
   */
  private extractLegalTopics(metadataArray: any[]): string[] {
    const topicCounts = new Map<string, number>();
    
    for (const metadata of metadataArray) {
      const topics = metadata.legalTopics || metadata.keywords || [];
      for (const topic of topics) {
        topicCounts.set(topic, (topicCounts.get(topic) || 0) + 1);
      }
    }
    
    // Return topics sorted by frequency
    return Array.from(topicCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([topic]) => topic);
  }
  
  /**
   * Calculate cluster coherence based on neuron similarity
   */
  private async calculateClusterCoherence(x: number, y: number, documentCount: number): Promise<number> {
    const neighbors = await this.getNeighborhood(x, y, 1);
    const centerNeuron = this.neurons[x][y];
    
    let totalSimilarity = 0;
    for (const neighbor of neighbors) {
      totalSimilarity += this.cosineSimilarity(centerNeuron, neighbor);
    }
    
    const averageSimilarity = totalSimilarity / neighbors.length;
    const documentDensity = Math.min(documentCount / 10, 1); // Normalize to 0-1
    
    return (averageSimilarity + documentDensity) / 2;
  }
  
  // =============================================================================
  // UTILITY METHODS
  // =============================================================================
  
  private euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
  }
  
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }
  
  private shuffleArray<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }
  
  // =============================================================================
  // PERSISTENCE METHODS
  // =============================================================================
  
  private async saveTrainingProgress(iteration: number, learningRate: number, radius: number): Promise<void> {
    await this.redis.hset('som:training:progress', {
      iteration,
      learningRate,
      radius,
      timestamp: Date.now()
    });
  }
  
  private async saveToRedis(): Promise<void> {
    const serialized = {
      config: this.config,
      neurons: this.neurons,
      trained: this.trained,
      savedAt: new Date().toISOString()
    };
    
    await this.redis.set('som:model', JSON.stringify(serialized));
  }
  
  static async loadFromRedis(redis: Redis): Promise<LegalDocumentSOM | null> {
    const serialized = await redis.get('som:model');
    if (!serialized) return null;
    
    const data = JSON.parse(serialized);
    const som = new LegalDocumentSOM(data.config, redis);
    som.neurons = data.neurons;
    som.trained = data.trained;
    
    return som;
  }
}