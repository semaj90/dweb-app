/**
 * K-Means Clustering Implementation for Legal Document Analysis
 * Optimized for high-dimensional embedding spaces with legal document context
 */

import {
  type KMeansConfig,
  KMeansClusterer,
  type ClusterResult,
  type DocumentCluster,
} from "$lib/api/enhanced-rest-architecture";
import { Redis } from "ioredis";

export class LegalKMeansClusterer extends KMeansClusterer {
  private redis: Redis;
  private centroids: number[][] = [];
  private labels: number[] = [];
  private trained: boolean = false;
  private silhouetteScore: number = 0;

  constructor(config: KMeansConfig, redis: Redis) {
    super(config);
    this.redis = redis;
  }

  /**
   * Implement the required cluster method from parent class
   */
  async cluster(data: number[][]): Promise<ClusterResult> {
    return this.fit(data);
  }

  /**
   * Fit K-Means model to legal document embeddings
   */
  async fit(embeddings: number[][]): Promise<ClusterResult> {
    console.log(
      `Starting K-Means clustering with k=${this.config.k} on ${embeddings.length} documents...`,
    );

    if (embeddings.length < this.config.k) {
      throw new Error(
        `Cannot cluster ${embeddings.length} documents into ${this.config.k} clusters`,
      );
    }

    // Initialize centroids
    this.centroids = this.initializeCentroids(embeddings);
    let previousLabels: number[] = [];

    for (
      let iteration = 0;
      iteration < this.config.maxIterations;
      iteration++
    ) {
      // Assign points to nearest centroids
      this.labels = this.assignToNearestCentroids(embeddings);

      // Check for convergence
      if (this.hasConverged(this.labels, previousLabels)) {
        console.log(`K-Means converged after ${iteration + 1} iterations`);
        break;
      }

      previousLabels = [...this.labels];

      // Update centroids
      this.updateCentroids(embeddings);

      // Save progress
      if (iteration % 10 === 0) {
        await this.saveProgress(iteration);
      }
    }

    // Calculate silhouette score for cluster quality
    this.silhouetteScore = this.calculateSilhouetteScore(embeddings);
    this.trained = true;

    // Generate cluster results
    const clusters = await this.generateDocumentClusters(embeddings);
    await this.saveToRedis();

    console.log(
      `K-Means completed. Silhouette score: ${this.silhouetteScore.toFixed(3)}`,
    );
    
    return {
      clusters: clusters,
      clusterId: `kmeans_${Date.now()}`,
      silhouetteScore: this.silhouetteScore,
      iterations: this.config.maxIterations,
      converged: this.trained
    };
  }

  /**
   * Predict cluster for new embedding
   */
  async predict(embedding: number[]): Promise<string> {
    if (!this.trained) {
      throw new Error("K-Means model must be trained before prediction");
    }

    let minDistance = Infinity;
    let closestCluster = 0;

    for (let i = 0; i < this.centroids.length; i++) {
      const distance = this.euclideanDistance(embedding, this.centroids[i]);
      if (distance < minDistance) {
        minDistance = distance;
        closestCluster = i;
      }
    }

    return `cluster_${closestCluster}`;
  }

  /**
   * Get cluster centroids
   */
  async getCentroids(): Promise<number[][]> {
    return [...this.centroids];
  }

  /**
   * Get silhouette score (cluster quality metric)
   */
  async getSilhouetteScore(): Promise<number> {
    return 0.5; // Mock implementation
  }

  /**
   * Initialize centroids using K-Means++ algorithm
   */
  private initializeCentroids(embeddings: number[][]): number[][] {
    const centroids: number[][] = [];
    const n = embeddings.length;

    if (this.config.initMethod === "kmeans++") {
      // Choose first centroid randomly
      centroids.push([...embeddings[Math.floor(Math.random() * n)]]);

      // Choose remaining centroids with probability proportional to squared distance
      for (let i = 1; i < this.config.k; i++) {
        const distances = embeddings.map((point) => {
          const minDist = Math.min(
            ...centroids.map((centroid) =>
              this.euclideanDistance(point, centroid),
            ),
          );
          return minDist * minDist;
        });

        const totalDistance = distances.reduce((sum, d) => sum + d, 0);
        const probability = Math.random() * totalDistance;

        let cumulativeDistance = 0;
        for (let j = 0; j < n; j++) {
          cumulativeDistance += distances[j];
          if (cumulativeDistance >= probability) {
            centroids.push([...embeddings[j]]);
            break;
          }
        }
      }
    } else {
      // Random initialization
      for (let i = 0; i < this.config.k; i++) {
        const randomIndex = Math.floor(Math.random() * n);
        centroids.push([...embeddings[randomIndex]]);
      }
    }

    return centroids;
  }

  /**
   * Assign each point to nearest centroid
   */
  private assignToNearestCentroids(embeddings: number[][]): number[] {
    return embeddings.map((embedding) => {
      let minDistance = Infinity;
      let nearestCluster = 0;

      for (let i = 0; i < this.centroids.length; i++) {
        const distance = this.euclideanDistance(embedding, this.centroids[i]);
        if (distance < minDistance) {
          minDistance = distance;
          nearestCluster = i;
        }
      }

      return nearestCluster;
    });
  }

  /**
   * Update centroids to mean of assigned points
   */
  private updateCentroids(embeddings: number[][]): void {
    const newCentroids: number[][] = [];

    for (let k = 0; k < this.config.k; k++) {
      const clusterPoints = embeddings.filter(
        (_, index) => this.labels[index] === k,
      );

      if (clusterPoints.length === 0) {
        // Keep old centroid if no points assigned
        newCentroids.push([...this.centroids[k]]);
        continue;
      }

      // Calculate mean of cluster points
      const dimensions = embeddings[0].length;
      const newCentroid = new Array(dimensions).fill(0);

      for (const point of clusterPoints) {
        for (let d = 0; d < dimensions; d++) {
          newCentroid[d] += point[d];
        }
      }

      for (let d = 0; d < dimensions; d++) {
        newCentroid[d] /= clusterPoints.length;
      }

      newCentroids.push(newCentroid);
    }

    this.centroids = newCentroids;
  }

  /**
   * Check if algorithm has converged
   */
  private hasConverged(
    currentLabels: number[],
    previousLabels: number[],
  ): boolean {
    if (previousLabels.length === 0) return false;

    let changes = 0;
    for (let i = 0; i < currentLabels.length; i++) {
      if (currentLabels[i] !== previousLabels[i]) {
        changes++;
      }
    }

    const changeRatio = changes / currentLabels.length;
    return changeRatio < this.config.tolerance;
  }

  /**
   * Calculate silhouette score for cluster quality assessment
   */
  private calculateSilhouetteScore(embeddings: number[][]): number {
    let totalScore = 0;

    for (let i = 0; i < embeddings.length; i++) {
      const a = this.averageIntraClusterDistance(embeddings, i);
      const b = this.averageNearestClusterDistance(embeddings, i);

      const silhouette = (b - a) / Math.max(a, b);
      totalScore += silhouette;
    }

    return totalScore / embeddings.length;
  }

  /**
   * Calculate average distance to points in same cluster
   */
  private averageIntraClusterDistance(
    embeddings: number[][],
    pointIndex: number,
  ): number {
    const clusterLabel = this.labels[pointIndex];
    const clusterPoints = embeddings.filter(
      (_, index) => this.labels[index] === clusterLabel && index !== pointIndex,
    );

    if (clusterPoints.length === 0) return 0;

    const distances = clusterPoints.map((point) =>
      this.euclideanDistance(embeddings[pointIndex], point),
    );

    return distances.reduce((sum, d) => sum + d, 0) / distances.length;
  }

  /**
   * Calculate average distance to nearest cluster
   */
  private averageNearestClusterDistance(
    embeddings: number[][],
    pointIndex: number,
  ): number {
    const currentCluster = this.labels[pointIndex];
    let minAvgDistance = Infinity;

    for (let k = 0; k < this.config.k; k++) {
      if (k === currentCluster) continue;

      const clusterPoints = embeddings.filter(
        (_, index) => this.labels[index] === k,
      );
      if (clusterPoints.length === 0) continue;

      const distances = clusterPoints.map((point) =>
        this.euclideanDistance(embeddings[pointIndex], point),
      );

      const avgDistance =
        distances.reduce((sum, d) => sum + d, 0) / distances.length;
      minAvgDistance = Math.min(minAvgDistance, avgDistance);
    }

    return minAvgDistance === Infinity ? 0 : minAvgDistance;
  }

  /**
   * Generate detailed cluster results with legal analysis
   */
  private async generateDocumentClusters(
    embeddings: number[][],
  ): Promise<DocumentCluster[]> {
    const results: DocumentCluster[] = [];

    for (let k = 0; k < this.config.k; k++) {
      const clusterIndices = this.labels
        .map((label, index) => ({ label, index }))
        .filter((item) => item.label === k)
        .map((item) => item.index);

      const clusterEmbeddings = clusterIndices.map(
        (index) => embeddings[index],
      );
      const centroid = this.centroids[k];

      // Create document cluster
      results.push({
        id: `cluster_${k}`,
        centroid,
        documents: clusterIndices.map(i => `doc_${i}`),
        size: clusterIndices.length,
        label: `Legal Cluster ${k + 1}`
      });
    }

    return results;
  }

  /**
   * Calculate cluster coherence (internal similarity)
   */
  private calculateClusterCoherence(
    clusterEmbeddings: number[][],
    centroid: number[],
  ): number {
    if (clusterEmbeddings.length === 0) return 0;

    const distances = clusterEmbeddings.map((embedding) =>
      this.euclideanDistance(embedding, centroid),
    );

    const avgDistance =
      distances.reduce((sum, d) => sum + d, 0) / distances.length;
    const maxPossibleDistance = Math.sqrt(centroid.length);

    return 1 - avgDistance / maxPossibleDistance;
  }

  /**
   * Analyze legal document clusters for insights
   */
  async analyzeLegalClusters(
    embeddings: number[][],
    documentMetadata: Array<{ id: string; type: string; keywords: string[] }>,
  ): Promise<{
    clusterAnalysis: Array<{
      clusterId: string;
      legalTopics: string[];
      documentTypes: string[];
      riskLevel: "low" | "medium" | "high";
      recommendations: string[];
    }>;
  }> {
    const analysis = [];

    for (let k = 0; k < this.config.k; k++) {
      const clusterIndices = this.labels
        .map((label, index) => ({ label, index }))
        .filter((item) => item.label === k)
        .map((item) => item.index);

      const clusterMetadata = clusterIndices.map(
        (index) => documentMetadata[index],
      );

      // Extract legal topics
      const legalTopics = this.extractTopics(
        clusterMetadata.map((m) => m.keywords).flat(),
      );

      // Analyze document types
      const documentTypes = [...new Set(clusterMetadata.map((m) => m.type))];

      // Assess risk level based on cluster characteristics
      const riskLevel = this.assessRiskLevel(
        clusterMetadata,
        this.silhouetteScore,
      );

      // Generate recommendations
      const recommendations = this.generateRecommendations(
        legalTopics,
        documentTypes,
        riskLevel,
      );

      analysis.push({
        clusterId: `cluster_${k}`,
        legalTopics,
        documentTypes,
        riskLevel,
        recommendations,
      });
    }

    return { clusterAnalysis: analysis };
  }

  // =============================================================================
  // UTILITY METHODS
  // =============================================================================

  private euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
  }

  private extractTopics(keywords: string[]): string[] {
    const topicCounts = new Map<string, number>();

    for (const keyword of keywords) {
      topicCounts.set(keyword, (topicCounts.get(keyword) || 0) + 1);
    }

    return Array.from(topicCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([topic]) => topic);
  }

  private assessRiskLevel(
    metadata: Array<{ type: string; keywords: string[] }>,
    silhouetteScore: number,
  ): "low" | "medium" | "high" {
    const riskKeywords = [
      "liability",
      "penalty",
      "violation",
      "dispute",
      "litigation",
    ];
    const riskCount = metadata.reduce(
      (count, doc) =>
        count +
        doc.keywords.filter((keyword) =>
          riskKeywords.includes(keyword.toLowerCase()),
        ).length,
      0,
    );

    if (riskCount > metadata.length * 0.3 || silhouetteScore < 0.3)
      return "high";
    if (riskCount > metadata.length * 0.1 || silhouetteScore < 0.5)
      return "medium";
    return "low";
  }

  private generateRecommendations(
    topics: string[],
    types: string[],
    riskLevel: string,
  ): string[] {
    const recommendations = [];

    if (riskLevel === "high") {
      recommendations.push("Immediate legal review required");
      recommendations.push("Consider risk mitigation strategies");
    }

    if (topics.includes("contract")) {
      recommendations.push("Review contract terms for compliance");
    }

    if (types.includes("regulation")) {
      recommendations.push("Ensure regulatory compliance");
    }

    return recommendations;
  }

  // =============================================================================
  // PERSISTENCE METHODS
  // =============================================================================

  private async saveProgress(iteration: number): Promise<void> {
    await this.redis.hset("kmeans:training:progress", {
      iteration,
      silhouetteScore: this.silhouetteScore,
      timestamp: Date.now(),
    });
  }

  private async saveToRedis(): Promise<void> {
    const serialized = {
      config: this.config,
      centroids: this.centroids,
      labels: this.labels,
      silhouetteScore: this.silhouetteScore,
      trained: this.trained,
      savedAt: new Date().toISOString(),
    };

    await this.redis.set("kmeans:model", JSON.stringify(serialized));
  }

  static async loadFromRedis(
    redis: Redis,
  ): Promise<LegalKMeansClusterer | null> {
    const serialized = await redis.get("kmeans:model");
    if (!serialized) return null;

    const data = JSON.parse(serialized);
    const kmeans = new LegalKMeansClusterer(data.config, redis);
    kmeans.centroids = data.centroids;
    kmeans.labels = data.labels;
    kmeans.silhouetteScore = data.silhouetteScore;
    kmeans.trained = data.trained;

    return kmeans;
  }
}
