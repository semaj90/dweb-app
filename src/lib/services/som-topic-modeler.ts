/**
 * Self-Organizing Map (SOM) Topic Modeler
 * Advanced topic modeling using Kohonen's SOM algorithm
 * Implements: GPU-accelerated training, density-based ranking, predictive caching
 */

import { writable, derived } from "svelte/store";
import { webGPUProcessor } from "./webgpu-processor";

export interface SOMConfig {
  gridWidth: number;
  gridHeight: number;
  embeddingDim: number;
  learningRate: number;
  initialRadius: number;
  iterations: number;
  useGPU: boolean;
}

export interface SOMNode {
  x: number;
  y: number;
  weights: Float32Array;
  density: number;
  topicLabel?: string;
  documentIds: string[];
  representative?: boolean;
}

export interface TopicCluster {
  id: string;
  centroid: { x: number; y: number };
  nodes: SOMNode[];
  density: number;
  documents: string[];
  keywords: string[];
  importance: number;
  relationships: string[]; // IDs of related clusters
}

export interface DocumentMapping {
  documentId: string;
  embedding: Float32Array;
  bestMatchingUnit: { x: number; y: number };
  similarity: number;
  topicCluster: string;
}

class SOMTopicModeler {
  private config: SOMConfig;
  private som = writable<SOMNode[][]>([]);
  private clusters = writable<TopicCluster[]>([]);
  private documentMappings = writable<Map<string, DocumentMapping>>(new Map());
  private isTraining = writable(false);
  private trainingProgress = writable(0);

  // High-ranking system for proactive caching
  private densityMap = writable<Float32Array>(new Float32Array());
  private importanceScores = writable<Map<string, number>>(new Map());

  constructor(config: Partial<SOMConfig> = {}) {
    this.config = {
      gridWidth: 50,
      gridHeight: 50,
      embeddingDim: 768,
      learningRate: 0.1,
      initialRadius: 25,
      iterations: 1000,
      useGPU: true,
      ...config,
    };

    this.initializeSOM();
  }

  private initializeSOM() {
    const grid: SOMNode[][] = [];

    for (let x = 0; x < this.config.gridWidth; x++) {
      grid[x] = [];
      for (let y = 0; y < this.config.gridHeight; y++) {
        // Initialize with small random weights
        const weights = new Float32Array(this.config.embeddingDim);
        for (let i = 0; i < this.config.embeddingDim; i++) {
          weights[i] = (Math.random() - 0.5) * 0.1;
        }

        grid[x][y] = {
          x,
          y,
          weights,
          density: 0,
          documentIds: [],
          representative: false,
        };
      }
    }

    this.som.set(grid);
    console.log(
      `🧠 SOM initialized: ${this.config.gridWidth}x${this.config.gridHeight}`
    );
  }

  async trainOnDocuments(
    documents: Array<{ id: string; embedding: Float32Array; text?: string }>
  ): Promise<void> {
    if (documents.length === 0) return;

    this.isTraining.set(true);
    this.trainingProgress.set(0);

    try {
      console.log(`🚀 Starting SOM training on ${documents.length} documents`);

      // Prepare embedding data for training
      const embeddings = new Float32Array(
        documents.length * this.config.embeddingDim
      );
      for (let i = 0; i < documents.length; i++) {
        embeddings.set(documents[i].embedding, i * this.config.embeddingDim);
      }

      let trainedWeights: Float32Array;

      if (this.config.useGPU && (await webGPUProcessor.initialize())) {
        // GPU-accelerated training
        console.log("🔥 Using GPU acceleration for SOM training");
        trainedWeights = await webGPUProcessor.trainSOM(
          embeddings,
          this.config.gridWidth,
          this.config.gridHeight,
          this.config.iterations
        );
      } else {
        // CPU fallback training
        console.log("💻 Using CPU for SOM training");
        trainedWeights = await this.trainSOMOnCPU(embeddings);
      }

      // Update SOM with trained weights
      await this.updateSOMWithWeights(trainedWeights);

      // Map documents to their BMUs
      await this.mapDocumentsToBMUs(documents);

      // Calculate density and importance scores
      await this.calculateDensityAndImportance();

      // Extract topic clusters
      await this.extractTopicClusters();

      // Setup proactive caching based on high-ranking areas
      this.setupProactiveCaching();

      console.log("✅ SOM training completed successfully");
    } catch (error) {
      console.error("❌ SOM training failed:", error);
      throw error;
    } finally {
      this.isTraining.set(false);
      this.trainingProgress.set(100);
    }
  }

  private async trainSOMOnCPU(embeddings: Float32Array): Promise<Float32Array> {
    const {
      gridWidth,
      gridHeight,
      embeddingDim,
      learningRate,
      initialRadius,
      iterations,
    } = this.config;
    const somSize = gridWidth * gridHeight * embeddingDim;
    const numDocs = embeddings.length / embeddingDim;

    // Initialize SOM weights
    const weights = new Float32Array(somSize);
    for (let i = 0; i < somSize; i++) {
      weights[i] = (Math.random() - 0.5) * 0.1;
    }

    for (let iter = 0; iter < iterations; iter++) {
      // Update progress
      this.trainingProgress.set((iter / iterations) * 100);

      // Decay learning rate and neighborhood radius
      const currentLearningRate = learningRate * Math.exp(-iter / iterations);
      const currentRadius = initialRadius * Math.exp(-iter / (iterations / 3));

      // Select random data point
      const dataIdx = Math.floor(Math.random() * numDocs);
      const dataStart = dataIdx * embeddingDim;

      // Find Best Matching Unit (BMU)
      let bestDistance = Infinity;
      let bmuX = 0,
        bmuY = 0;

      for (let x = 0; x < gridWidth; x++) {
        for (let y = 0; y < gridHeight; y++) {
          let distance = 0;
          const nodeStart = (x * gridHeight + y) * embeddingDim;

          for (let d = 0; d < embeddingDim; d++) {
            const diff = weights[nodeStart + d] - embeddings[dataStart + d];
            distance += diff * diff;
          }

          if (distance < bestDistance) {
            bestDistance = distance;
            bmuX = x;
            bmuY = y;
          }
        }
      }

      // Update weights in neighborhood of BMU
      for (let x = 0; x < gridWidth; x++) {
        for (let y = 0; y < gridHeight; y++) {
          const gridDistance = Math.sqrt((x - bmuX) ** 2 + (y - bmuY) ** 2);
          const influence = Math.exp(
            -(gridDistance ** 2) / (2 * currentRadius ** 2)
          );

          if (influence > 0.01) {
            // Only update if influence is significant
            const nodeStart = (x * gridHeight + y) * embeddingDim;

            for (let d = 0; d < embeddingDim; d++) {
              const delta =
                currentLearningRate *
                influence *
                (embeddings[dataStart + d] - weights[nodeStart + d]);
              weights[nodeStart + d] += delta;
            }
          }
        }
      }

      // Log progress periodically
      if (iter % 100 === 0) {
        console.log(
          `SOM training progress: ${iter}/${iterations} (${((iter / iterations) * 100).toFixed(1)}%)`
        );
      }
    }

    return weights;
  }

  private async updateSOMWithWeights(weights: Float32Array) {
    const { gridWidth, gridHeight, embeddingDim } = this.config;

    this.som.update((grid) => {
      for (let x = 0; x < gridWidth; x++) {
        for (let y = 0; y < gridHeight; y++) {
          const nodeStart = (x * gridHeight + y) * embeddingDim;
          const nodeWeights = weights.slice(
            nodeStart,
            nodeStart + embeddingDim
          );
          grid[x][y].weights = nodeWeights;
        }
      }
      return grid;
    });
  }

  private async mapDocumentsToBMUs(
    documents: Array<{ id: string; embedding: Float32Array }>
  ) {
    const mappings = new Map<string, DocumentMapping>();

    this.som.subscribe((grid) => {
      for (const doc of documents) {
        let bestDistance = Infinity;
        let bestX = 0,
          bestY = 0;

        // Find BMU for this document
        for (let x = 0; x < grid.length; x++) {
          for (let y = 0; y < grid[x].length; y++) {
            const node = grid[x][y];
            let distance = 0;

            for (let d = 0; d < this.config.embeddingDim; d++) {
              const diff = node.weights[d] - doc.embedding[d];
              distance += diff * diff;
            }

            if (distance < bestDistance) {
              bestDistance = distance;
              bestX = x;
              bestY = y;
            }
          }
        }

        // Calculate similarity (inverse of distance)
        const similarity = 1 / (1 + Math.sqrt(bestDistance));

        mappings.set(doc.id, {
          documentId: doc.id,
          embedding: doc.embedding,
          bestMatchingUnit: { x: bestX, y: bestY },
          similarity,
          topicCluster: `cluster_${bestX}_${bestY}`, // Will be updated after clustering
        });

        // Add document to the BMU node
        grid[bestX][bestY].documentIds.push(doc.id);
      }
    })();

    this.documentMappings.set(mappings);
  }

  private async calculateDensityAndImportance() {
    const { gridWidth, gridHeight } = this.config;
    const densityArray = new Float32Array(gridWidth * gridHeight);
    const importanceMap = new Map<string, number>();

    this.som.subscribe((grid) => {
      let maxDensity = 0;

      // Calculate density for each node
      for (let x = 0; x < gridWidth; x++) {
        for (let y = 0; y < gridHeight; y++) {
          const node = grid[x][y];
          node.density = node.documentIds.length;

          densityArray[x * gridHeight + y] = node.density;
          maxDensity = Math.max(maxDensity, node.density);

          // Calculate importance based on density and centrality
          const centrality = this.calculateCentrality(
            x,
            y,
            gridWidth,
            gridHeight
          );
          const importance = node.density * centrality;

          importanceMap.set(`${x}_${y}`, importance);

          // Mark high-density nodes as representatives
          if (node.density > 0) {
            node.representative = node.density >= maxDensity * 0.7;
          }
        }
      }

      // Normalize density values
      for (let i = 0; i < densityArray.length; i++) {
        densityArray[i] /= maxDensity;
      }
    })();

    this.densityMap.set(densityArray);
    this.importanceScores.set(importanceMap);
  }

  private calculateCentrality(
    x: number,
    y: number,
    width: number,
    height: number
  ): number {
    // Distance from center of grid (normalized)
    const centerX = width / 2;
    const centerY = height / 2;
    const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
    const maxDistance = Math.sqrt(centerX ** 2 + centerY ** 2);

    return 1 - distance / maxDistance; // Higher centrality for nodes closer to center
  }

  private async extractTopicClusters() {
    const clusters: TopicCluster[] = [];

    this.som.subscribe((grid) => {
      const visited = new Set<string>();
      let clusterId = 0;

      for (let x = 0; x < grid.length; x++) {
        for (let y = 0; y < grid[x].length; y++) {
          const nodeKey = `${x}_${y}`;

          if (visited.has(nodeKey) || grid[x][y].documentIds.length === 0) {
            continue;
          }

          // Find connected high-density regions using flood fill
          const clusterNodes = this.floodFillCluster(grid, x, y, visited);

          if (clusterNodes.length > 0) {
            const cluster = this.createTopicCluster(
              `cluster_${clusterId++}`,
              clusterNodes
            );
            clusters.push(cluster);
          }
        }
      }

      // Calculate relationships between clusters
      this.calculateClusterRelationships(clusters);
    })();

    this.clusters.set(clusters);
    console.log(`📊 Extracted ${clusters.length} topic clusters`);
  }

  private floodFillCluster(
    grid: SOMNode[][],
    startX: number,
    startY: number,
    visited: Set<string>
  ): SOMNode[] {
    const cluster: SOMNode[] = [];
    const queue: Array<{ x: number; y: number }> = [{ x: startX, y: startY }];
    const minDensity = 1; // Minimum density threshold for cluster inclusion

    while (queue.length > 0) {
      const { x, y } = queue.shift()!;
      const nodeKey = `${x}_${y}`;

      if (
        visited.has(nodeKey) ||
        x < 0 ||
        x >= grid.length ||
        y < 0 ||
        y >= grid[0].length ||
        grid[x][y].density < minDensity
      ) {
        continue;
      }

      visited.add(nodeKey);
      cluster.push(grid[x][y]);

      // Add neighbors to queue
      const neighbors = [
        { x: x - 1, y },
        { x: x + 1, y },
        { x, y: y - 1 },
        { x, y: y + 1 },
      ];

      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x}_${neighbor.y}`;
        if (!visited.has(neighborKey)) {
          queue.push(neighbor);
        }
      }
    }

    return cluster;
  }

  private createTopicCluster(id: string, nodes: SOMNode[]): TopicCluster {
    // Calculate centroid
    const centroidX =
      nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length;
    const centroidY =
      nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length;

    // Collect all documents in cluster
    const documents = nodes.flatMap((node) => node.documentIds);

    // Calculate cluster density and importance
    const totalDensity = nodes.reduce((sum, node) => sum + node.density, 0);
    const avgDensity = totalDensity / nodes.length;

    // Generate topic keywords (placeholder - would use actual NLP)
    const keywords = this.generateTopicKeywords(documents);

    return {
      id,
      centroid: { x: centroidX, y: centroidY },
      nodes,
      density: avgDensity,
      documents,
      keywords,
      importance: totalDensity,
      relationships: [], // Will be calculated separately
    };
  }

  private generateTopicKeywords(documentIds: string[]): string[] {
    // Placeholder implementation
    // In practice, this would analyze document content to extract key terms
    const topics = [
      "Contract Law",
      "Property Rights",
      "Evidence Analysis",
      "Litigation",
      "Intellectual Property",
      "Corporate Law",
      "Criminal Law",
      "Family Law",
      "Tax Law",
      "Environmental Law",
      "Employment Law",
      "Healthcare Law",
    ];

    return topics.slice(0, Math.min(3, Math.ceil(documentIds.length / 10)));
  }

  private calculateClusterRelationships(clusters: TopicCluster[]) {
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const cluster1 = clusters[i];
        const cluster2 = clusters[j];

        // Calculate distance between centroids
        const distance = Math.sqrt(
          (cluster1.centroid.x - cluster2.centroid.x) ** 2 +
            (cluster1.centroid.y - cluster2.centroid.y) ** 2
        );

        // If clusters are close enough, consider them related
        const maxRelationshipDistance =
          Math.min(this.config.gridWidth, this.config.gridHeight) * 0.3;

        if (distance <= maxRelationshipDistance) {
          cluster1.relationships.push(cluster2.id);
          cluster2.relationships.push(cluster1.id);
        }
      }
    }
  }

  private setupProactiveCaching() {
    // Identify high-importance areas for proactive caching
    this.importanceScores.subscribe((scores) => {
      const sortedScores = Array.from(scores.entries())
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10); // Top 10 most important areas

      // Send to caching service
      this.triggerProactiveCache(sortedScores);
    })();
  }

  private async triggerProactiveCache(importantAreas: Array<[string, number]>) {
    try {
      await fetch("/api/proactive-cache/som", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          importantAreas: importantAreas.map(([coords, importance]) => ({
            coordinates: coords,
            importance,
          })),
        }),
      });
    } catch (error) {
      console.error("Failed to trigger proactive caching:", error);
    }
  }

  // Public API
  getSOM() {
    return this.som;
  }

  getClusters() {
    return this.clusters;
  }

  getDocumentMappings() {
    return this.documentMappings;
  }

  getDensityMap() {
    return this.densityMap;
  }

  getTrainingState() {
    return {
      isTraining: this.isTraining,
      progress: this.trainingProgress,
    };
  }

  // Derived stores for insights
  getTopicInsights() {
    return derived(this.clusters, ($clusters) => ({
      totalClusters: $clusters.length,
      averageClusterSize:
        $clusters.reduce((sum, c) => sum + c.documents.length, 0) /
        $clusters.length,
      mostImportantCluster: $clusters.reduce(
        (max, cluster) => (cluster.importance > max.importance ? cluster : max),
        $clusters[0] || { importance: 0 }
      ),
      clusterDistribution: $clusters.map((c) => ({
        id: c.id,
        size: c.documents.length,
        keywords: c.keywords.slice(0, 3),
      })),
    }));
  }

  getHighRankingAreas() {
    return derived(
      [this.densityMap, this.importanceScores],
      ([$densityMap, $importanceScores]) => {
        const highRankingAreas = Array.from($importanceScores.entries())
          .filter(([, importance]) => importance > 0.7)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5);

        return highRankingAreas.map(([coords, importance]) => ({
          coordinates: coords,
          importance,
          density:
            $densityMap[
              parseInt(coords.split("_")[0]) * this.config.gridHeight +
                parseInt(coords.split("_")[1])
            ],
        }));
      }
    );
  }

  async findSimilarDocuments(
    documentId: string,
    limit: number = 10
  ): Promise<string[]> {
    const mapping = this.documentMappings.subscribe((mappings) =>
      mappings.get(documentId)
    )();
    if (!mapping) return [];

    const { x: targetX, y: targetY } = mapping.bestMatchingUnit;
    const candidates: Array<{ id: string; distance: number }> = [];

    // Search in expanding radius around the BMU
    for (let radius = 0; radius <= 5; radius++) {
      this.som.subscribe((grid) => {
        for (let dx = -radius; dx <= radius; dx++) {
          for (let dy = -radius; dy <= radius; dy++) {
            const x = targetX + dx;
            const y = targetY + dy;

            if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length) {
              const node = grid[x][y];
              const distance = Math.sqrt(dx * dx + dy * dy);

              for (const docId of node.documentIds) {
                if (docId !== documentId) {
                  candidates.push({ id: docId, distance });
                }
              }
            }
          }
        }
      })();

      if (candidates.length >= limit) break;
    }

    return candidates
      .sort((a, b) => a.distance - b.distance)
      .slice(0, limit)
      .map((c) => c.id);
  }

  // Configuration updates
  updateConfig(newConfig: Partial<SOMConfig>) {
    this.config = { ...this.config, ...newConfig };
    this.initializeSOM(); // Reinitialize with new config
  }
}

// Singleton instance
export const somTopicModeler = new SOMTopicModeler({
  gridWidth: 50,
  gridHeight: 50,
  embeddingDim: 768,
  learningRate: 0.1,
  initialRadius: 25,
  iterations: 1000,
  useGPU: true,
});
