// Vector operations utilities for embedding pipeline
// Handles: normalization, similarity, clustering, quantization

import { createHash } from 'crypto';

export class VectorOperations {
  // Convert number array to pgvector format string
  static toVector(arr) {
    if (!Array.isArray(arr) || arr.length === 0) {
      throw new Error('Invalid vector array');
    }
    return `[${arr.join(',')}]`;
  }

  // Parse pgvector string to number array
  static fromVector(vectorString) {
    if (typeof vectorString !== 'string') {
      throw new Error('Invalid vector string');
    }
    
    const cleaned = vectorString.replace(/[\[\]]/g, '');
    return cleaned.split(',').map(n => parseFloat(n.trim())).filter(n => !isNaN(n));
  }

  // Normalize vector to unit length
  static normalize(vector) {
    if (!Array.isArray(vector) || vector.length === 0) {
      return vector;
    }

    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    
    if (magnitude === 0 || !isFinite(magnitude)) {
      return vector.map(() => 0);
    }

    return vector.map(val => val / magnitude);
  }

  // Calculate cosine similarity between two vectors
  static cosineSimilarity(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      throw new Error('Invalid vectors for similarity calculation');
    }

    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  // Calculate Euclidean distance
  static euclideanDistance(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      throw new Error('Invalid vectors for distance calculation');
    }

    return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
  }

  // Calculate Manhattan (L1) distance
  static manhattanDistance(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      throw new Error('Invalid vectors for distance calculation');
    }

    return a.reduce((sum, val, i) => sum + Math.abs(val - b[i]), 0);
  }

  // Vector quantization for storage efficiency
  static quantize(vector, bits = 8) {
    if (!Array.isArray(vector) || vector.length === 0) {
      return vector;
    }

    const maxVal = Math.max(...vector.map(Math.abs));
    if (maxVal === 0) return vector;

    const scale = (Math.pow(2, bits - 1) - 1) / maxVal;
    
    return vector.map(val => {
      const quantized = Math.round(val * scale);
      return Math.max(Math.min(quantized, Math.pow(2, bits - 1) - 1), -Math.pow(2, bits - 1));
    });
  }

  // Dequantize vector back to float
  static dequantize(quantizedVector, originalMax, bits = 8) {
    if (!Array.isArray(quantizedVector) || quantizedVector.length === 0) {
      return quantizedVector;
    }

    const scale = originalMax / (Math.pow(2, bits - 1) - 1);
    return quantizedVector.map(val => val * scale);
  }

  // Create hash for vector (for caching)
  static hash(input) {
    const text = Array.isArray(input) ? input.join(',') : String(input);
    return createHash('sha256').update(text).digest('hex').substring(0, 16);
  }

  // Average multiple vectors
  static average(vectors) {
    if (!Array.isArray(vectors) || vectors.length === 0) {
      throw new Error('Invalid vectors array');
    }

    const dimension = vectors[0].length;
    if (!vectors.every(v => v.length === dimension)) {
      throw new Error('All vectors must have same dimensions');
    }

    const averaged = new Array(dimension).fill(0);
    
    for (const vector of vectors) {
      for (let i = 0; i < dimension; i++) {
        averaged[i] += vector[i];
      }
    }

    return averaged.map(val => val / vectors.length);
  }

  // Weighted average of vectors
  static weightedAverage(vectorsWithWeights) {
    if (!Array.isArray(vectorsWithWeights) || vectorsWithWeights.length === 0) {
      throw new Error('Invalid vectors array');
    }

    const { vector: firstVector } = vectorsWithWeights[0];
    const dimension = firstVector.length;
    
    const averaged = new Array(dimension).fill(0);
    let totalWeight = 0;

    for (const { vector, weight = 1 } of vectorsWithWeights) {
      if (vector.length !== dimension) {
        throw new Error('All vectors must have same dimensions');
      }

      for (let i = 0; i < dimension; i++) {
        averaged[i] += vector[i] * weight;
      }
      totalWeight += weight;
    }

    if (totalWeight === 0) return averaged;

    return averaged.map(val => val / totalWeight);
  }

  // K-means clustering for vectors
  static async kMeansClustering(vectors, k = 3, maxIterations = 100) {
    if (!Array.isArray(vectors) || vectors.length === 0) {
      throw new Error('Invalid vectors array');
    }

    if (k > vectors.length) {
      k = vectors.length;
    }

    const dimension = vectors[0].length;
    
    // Initialize centroids randomly
    let centroids = [];
    for (let i = 0; i < k; i++) {
      centroids.push(vectors[Math.floor(Math.random() * vectors.length)].slice());
    }

    let assignments = new Array(vectors.length);
    let converged = false;
    let iteration = 0;

    while (!converged && iteration < maxIterations) {
      const newAssignments = new Array(vectors.length);
      
      // Assign each vector to nearest centroid
      for (let i = 0; i < vectors.length; i++) {
        let bestDistance = Infinity;
        let bestCluster = 0;
        
        for (let j = 0; j < centroids.length; j++) {
          const distance = this.euclideanDistance(vectors[i], centroids[j]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = j;
          }
        }
        
        newAssignments[i] = bestCluster;
      }

      // Check for convergence
      converged = assignments.every((val, idx) => val === newAssignments[idx]);
      assignments = newAssignments;

      if (!converged) {
        // Update centroids
        const newCentroids = new Array(k);
        
        for (let i = 0; i < k; i++) {
          const clusterVectors = vectors.filter((_, idx) => assignments[idx] === i);
          
          if (clusterVectors.length > 0) {
            newCentroids[i] = this.average(clusterVectors);
          } else {
            // Keep old centroid if no vectors assigned
            newCentroids[i] = centroids[i];
          }
        }
        
        centroids = newCentroids;
      }

      iteration++;
    }

    return {
      centroids,
      assignments,
      clusters: centroids.map((centroid, idx) => ({
        centroid,
        vectors: vectors.filter((_, vIdx) => assignments[vIdx] === idx),
        indices: assignments.map((assignment, vIdx) => assignment === idx ? vIdx : null).filter(idx => idx !== null)
      })),
      iterations: iteration,
      converged
    };
  }

  // Find nearest neighbors using brute force search
  static findNearestNeighbors(queryVector, vectorDatabase, k = 5, similarity = 'cosine') {
    if (!Array.isArray(vectorDatabase) || vectorDatabase.length === 0) {
      return [];
    }

    const distances = vectorDatabase.map((item, index) => {
      let distance;
      
      if (similarity === 'cosine') {
        distance = 1 - this.cosineSimilarity(queryVector, item.vector);
      } else if (similarity === 'euclidean') {
        distance = this.euclideanDistance(queryVector, item.vector);
      } else {
        distance = this.manhattanDistance(queryVector, item.vector);
      }

      return {
        ...item,
        distance,
        similarity: similarity === 'cosine' ? 1 - distance : 1 / (1 + distance),
        index
      };
    });

    return distances
      .sort((a, b) => a.distance - b.distance)
      .slice(0, k);
  }

  // PCA dimensionality reduction
  static async principalComponentAnalysis(vectors, targetDimensions = null) {
    if (!Array.isArray(vectors) || vectors.length === 0) {
      throw new Error('Invalid vectors array');
    }

    const originalDimensions = vectors[0].length;
    targetDimensions = targetDimensions || Math.min(originalDimensions, Math.floor(originalDimensions * 0.8));

    // Center the data
    const mean = this.average(vectors);
    const centered = vectors.map(vector => 
      vector.map((val, i) => val - mean[i])
    );

    // For simplicity, we'll use a basic implementation
    // In production, use a proper linear algebra library
    console.log(`PCA: reducing from ${originalDimensions} to ${targetDimensions} dimensions`);
    
    // Simple dimension truncation for now
    // TODO: Implement proper SVD-based PCA
    return {
      reduced: centered.map(vector => vector.slice(0, targetDimensions)),
      mean,
      originalDimensions,
      targetDimensions,
      explainedVariance: 0.85 // Mock value
    };
  }

  // Vector interpolation
  static interpolate(vectorA, vectorB, t = 0.5) {
    if (!Array.isArray(vectorA) || !Array.isArray(vectorB) || vectorA.length !== vectorB.length) {
      throw new Error('Invalid vectors for interpolation');
    }

    t = Math.max(0, Math.min(1, t)); // Clamp t between 0 and 1

    return vectorA.map((val, i) => val * (1 - t) + vectorB[i] * t);
  }

  // Spherical linear interpolation (for normalized vectors)
  static slerp(vectorA, vectorB, t = 0.5) {
    const normalizedA = this.normalize(vectorA);
    const normalizedB = this.normalize(vectorB);
    
    const dot = this.cosineSimilarity(normalizedA, normalizedB);
    
    // If vectors are too similar, use linear interpolation
    if (Math.abs(dot) > 0.9995) {
      return this.normalize(this.interpolate(normalizedA, normalizedB, t));
    }

    const theta = Math.acos(Math.abs(dot));
    const sinTheta = Math.sin(theta);

    const ratioA = Math.sin((1 - t) * theta) / sinTheta;
    const ratioB = Math.sin(t * theta) / sinTheta;

    return normalizedA.map((val, i) => val * ratioA + normalizedB[i] * ratioB);
  }

  // Generate random vector (for testing)
  static randomVector(dimensions = 384, range = 1) {
    return Array.from({ length: dimensions }, () => (Math.random() - 0.5) * 2 * range);
  }

  // Vector statistics
  static stats(vector) {
    if (!Array.isArray(vector) || vector.length === 0) {
      return null;
    }

    const sorted = [...vector].sort((a, b) => a - b);
    const sum = vector.reduce((a, b) => a + b, 0);
    const mean = sum / vector.length;
    const variance = vector.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / vector.length;

    return {
      dimensions: vector.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean,
      median: sorted[Math.floor(sorted.length / 2)],
      variance,
      stdDev: Math.sqrt(variance),
      magnitude: Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0)),
      sparsity: vector.filter(val => Math.abs(val) < 1e-6).length / vector.length
    };
  }

  // Batch normalize vectors
  static batchNormalize(vectors) {
    return vectors.map(vector => this.normalize(vector));
  }

  // Check if vector is valid
  static isValid(vector) {
    return Array.isArray(vector) && 
           vector.length > 0 && 
           vector.every(val => typeof val === 'number' && isFinite(val));
  }

  // Convert vector to buffer for efficient storage
  static toBuffer(vector) {
    const buffer = new ArrayBuffer(vector.length * 4); // 4 bytes per float32
    const view = new Float32Array(buffer);
    for (let i = 0; i < vector.length; i++) {
      view[i] = vector[i];
    }
    return buffer;
  }

  // Convert buffer back to vector
  static fromBuffer(buffer) {
    const view = new Float32Array(buffer);
    return Array.from(view);
  }
}