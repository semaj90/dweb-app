// Integration Workflow Tests - Complete System Validation
// Tests all GPU/WASM/QUIC service integrations

/// <reference types="vitest" />
import { vi } from 'vitest';
import {
  UnifiedServiceOrchestrator,
  getOrchestrator,
  resetOrchestrator,
  type SystemHealth,
  TaskPriority
} from '../services/unified-service-orchestrator';
import type { CanvasState } from '../services/nes-gpu-bridge';

describe('Unified Service Integration Workflow', () => {
  let orchestrator: UnifiedServiceOrchestrator;

  beforeAll(async () => {
    // Reset any existing instance
    resetOrchestrator();

    // Initialize orchestrator with test configuration
    orchestrator = getOrchestrator({
      enabledServices: ['wasmGPU', 'quicGateway', 'llamaOllama', 'nesGPUBridge'],
      performanceThresholds: {
        maxLatency: 5000, // Increased for tests
        minThroughput: 1,
        maxCpuUsage: 90,
        maxMemoryUsage: 80
      },
      retryConfiguration: {
        maxRetries: 2,
        backoffMultiplier: 1.5,
        initialDelay: 50
      },
      monitoring: {
        healthCheckInterval: 10000, // Reduced frequency for tests
        metricsRetentionPeriod: 60000,
        alertThresholds: {
          latency: 3000,
          errorRate: 10,
          throughput: 1
        }
      }
    });

    // Wait for initial health check
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  afterAll(async () => {
    await orchestrator.shutdown();
    resetOrchestrator();
  });

  describe('System Health Monitoring', () => {
    it('should initialize with healthy system status', async () => {
      const health = orchestrator.getSystemHealth();

      expect(health).toBeDefined();
      expect(health.overall).toBe('healthy');
      expect(health.services).toBeDefined();
      expect(health.performance).toBeDefined();
      expect(health.lastChecked).toBeInstanceOf(Date);
    });

    it('should have all required services configured', async () => {
      const health = orchestrator.getSystemHealth();

      expect(health.services).toHaveProperty('wasmGPU');
      expect(health.services).toHaveProperty('quicGateway');
      expect(health.services).toHaveProperty('llamaOllama');
      expect(health.services).toHaveProperty('nesGPUBridge');
      expect(health.services).toHaveProperty('postgres');
      expect(health.services).toHaveProperty('redis');
    });

    it('should update performance metrics over time', async () => {
      const initialMetrics = orchestrator.getPerformanceMetrics();

      // Perform a test operation to generate metrics
      await orchestrator.performNeuralInference(
        new Float32Array([1, 2, 3, 4, 5]),
        { priority: TaskPriority.LOW }
      );

      const updatedMetrics = orchestrator.getPerformanceMetrics();
      expect(updatedMetrics.length).toBeGreaterThanOrEqual(initialMetrics.length);
    });
  });

  describe('Legal Document Processing', () => {
    it('should process legal documents through primary WASM GPU service', async () => {
      const testDocument = `
        LEGAL CONTRACT AGREEMENT

        This agreement is entered into between Party A and Party B.
        The terms and conditions are as follows:
        1. Payment terms: Net 30 days
        2. Delivery schedule: Within 14 business days
        3. Warranty period: 12 months

        Both parties agree to the terms outlined above.
      `;

      const result = await orchestrator.processLegalDocument(testDocument, {
        analysisType: 'contract',
        priority: TaskPriority.HIGH,
        maxTokens: 1024,
        temperature: 0.7
      });

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.taskId).toMatch(/^doc_\d+_[a-z0-9]+$/);
      expect(result.processingTime).toBeGreaterThan(0);
      expect(result.servicesUsed).toContain('wasmGPU');
      expect(result.performance.latency).toBeGreaterThan(0);
      expect(result.performance.throughput).toBeGreaterThan(0);
    });

    it('should handle document processing with fallbacks', async () => {
      const testDocument = 'Short test document for fallback testing.';

      const result = await orchestrator.processLegalDocument(testDocument, {
        analysisType: 'summary',
        priority: TaskPriority.NORMAL,
        timeout: 1000 // Short timeout to potentially trigger fallbacks
      });

      expect(result).toBeDefined();
      expect(result.servicesUsed.length).toBeGreaterThan(0);

      // Should use at least one service (primary or fallback)
      const expectedServices = ['wasmGPU', 'quicGateway', 'llamaOllama'];
      const usedValidService = result.servicesUsed.some(service =>
        expectedServices.includes(service)
      );
      expect(usedValidService).toBe(true);
    });
  });

  describe('Neural Inference Operations', () => {
    it('should perform neural inference with WASM GPU orchestrator', async () => {
      const testData = new Float32Array([
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        0.15, 0.25, 0.35, 0.45, 0.55
      ]);

      const result = await orchestrator.performNeuralInference(testData, {
        priority: TaskPriority.HIGH,
        modelType: 'transformer',
        precision: 'fp32'
      });

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.taskId).toMatch(/^inference_\d+_[a-z0-9]+$/);
      expect(result.servicesUsed).toContain('wasmGPU');
      expect(result.performance.resourceUsage).toBeGreaterThan(0);
    });

    it('should handle large tensor inference operations', async () => {
      // Create a larger tensor for stress testing
      const largeData = new Float32Array(1024);
      for (let i = 0; i < largeData.length; i++) {
        largeData[i] = Math.sin(i * 0.01) * Math.cos(i * 0.005);
      }

      const result = await orchestrator.performNeuralInference(largeData, {
        priority: TaskPriority.NORMAL,
        batchSize: 32,
        timeout: 10000
      });

      expect(result).toBeDefined();
      expect(result.processingTime).toBeGreaterThan(0);
      expect(result.servicesUsed.length).toBeGreaterThan(0);
    });
  });

  describe('Canvas State Processing', () => {
    it('should process canvas states through NES GPU Bridge', async () => {
      const testCanvas: CanvasState = {
        width: 128,
        height: 128,
        data: new Uint8ClampedArray(128 * 128 * 4),
        format: 'RGBA'
      };

      // Fill with test pattern
      for (let i = 0; i < testCanvas.data.length; i += 4) {
        testCanvas.data[i] = (i / 4) % 256;     // Red
        testCanvas.data[i + 1] = (i / 8) % 256; // Green
        testCanvas.data[i + 2] = (i / 16) % 256; // Blue
        testCanvas.data[i + 3] = 255;           // Alpha
      }

      const result = await orchestrator.processCanvasState(testCanvas, {
        priority: TaskPriority.NORMAL,
        targetBitDepth: 24,
        optimization: 'memory'
      });

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.taskId).toMatch(/^canvas_\d+_[a-z0-9]+$/);
      expect(result.servicesUsed).toContain('nesGPUBridge');
    });

    it('should handle canvas optimization and quantization', async () => {
      const testCanvas: CanvasState = {
        width: 64,
        height: 64,
        data: new Uint8ClampedArray(64 * 64 * 4),
        format: 'RGBA'
      };

      // Create gradient pattern
      for (let y = 0; y < 64; y++) {
        for (let x = 0; x < 64; x++) {
          const index = (y * 64 + x) * 4;
          testCanvas.data[index] = (x / 64) * 255;
          testCanvas.data[index + 1] = (y / 64) * 255;
          testCanvas.data[index + 2] = ((x + y) / 128) * 255;
          testCanvas.data[index + 3] = 255;
        }
      }

      const result = await orchestrator.processCanvasState(testCanvas, {
        priority: TaskPriority.LOW,
        targetBitDepth: 16,
        quantization: 'adaptive'
      });

      expect(result).toBeDefined();
      expect(result.performance.resourceUsage).toBeGreaterThan(0);
    });
  });

  describe('GPU Computation Operations', () => {
    it('should execute matrix multiplication operations', async () => {
      const matrixA = [1, 2, 3, 4, 5, 6];
      const matrixB = [7, 8, 9, 10, 11, 12];

      const result = await orchestrator.executeGPUComputation('matmul', {
        a: matrixA,
        b: matrixB,
        m: 2,
        n: 3,
        k: 3
      }, {
        priority: TaskPriority.HIGH,
        precision: 'fp32'
      });

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.taskId).toMatch(/^gpu_\d+_[a-z0-9]+$/);
      expect(result.servicesUsed.length).toBeGreaterThan(0);
    });

    it('should execute convolution operations', async () => {
      const inputData = Array(64).fill(0).map(() => Math.random());
      const kernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1]; // Edge detection kernel

      const result = await orchestrator.executeGPUComputation('conv2d', {
        input: inputData,
        kernel: kernel,
        width: 8,
        height: 8,
        kernel_size: 3
      }, {
        priority: TaskPriority.NORMAL,
        padding: 'same'
      });

      expect(result).toBeDefined();
      expect(result.processingTime).toBeGreaterThan(0);
    });

    it('should execute attention mechanism operations', async () => {
      const seq_len = 8;
      const dim = 16;

      const query = Array(seq_len * dim).fill(0).map(() => Math.random() - 0.5);
      const key = Array(seq_len * dim).fill(0).map(() => Math.random() - 0.5);
      const value = Array(seq_len * dim).fill(0).map(() => Math.random() - 0.5);

      const result = await orchestrator.executeGPUComputation('attention', {
        query,
        key,
        value,
        seq_len,
        dim
      }, {
        priority: TaskPriority.HIGH,
        scale: true
      });

      expect(result).toBeDefined();
      expect(result.performance).toBeDefined();
      expect(result.performance.latency).toBeGreaterThan(0);
    });
  });

  describe('Service Fallback and Error Handling', () => {
    it('should handle service failures gracefully', async () => {
      // Test with invalid data that might cause service failures
      const invalidData = new Float32Array([NaN, Infinity, -Infinity, 0]);

      const result = await orchestrator.performNeuralInference(invalidData, {
        priority: TaskPriority.LOW,
        timeout: 2000
      });

      // Should either succeed with a service or fail gracefully
      expect(result).toBeDefined();
      expect(result.taskId).toBeDefined();
      expect(result.processingTime).toBeGreaterThan(0);

      if (!result.success) {
        expect(result.error).toBeDefined();
        expect(typeof result.error).toBe('string');
      }
    });

    it('should track fallback usage correctly', async () => {
      const testDocument = 'Fallback test document';

      const result = await orchestrator.processLegalDocument(testDocument, {
        priority: TaskPriority.LOW,
        timeout: 500 // Very short timeout to trigger fallbacks
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.fallbacksTriggered)).toBe(true);
      expect(Array.isArray(result.servicesUsed)).toBe(true);
    });
  });

  describe('Performance and Monitoring', () => {
    it('should track task queue and active tasks', async () => {
      const initialActiveTasks = orchestrator.getActiveTasks();
      expect(Array.isArray(initialActiveTasks)).toBe(true);

      const initialQueue = orchestrator.getTaskQueue();
      expect(Array.isArray(initialQueue)).toBe(true);
    });

    it('should maintain performance metrics history', async () => {
      const initialMetrics = orchestrator.getPerformanceMetrics();

      // Perform several operations to generate metrics
      const operations = [
        () => orchestrator.processLegalDocument('Test doc 1'),
        () => orchestrator.performNeuralInference(new Float32Array([1, 2, 3])),
        () => orchestrator.executeGPUComputation('test', { data: [1, 2, 3] })
      ];

      await Promise.all(operations.map(op => op().catch(() => {
        // Ignore failures for metrics test
      })));

      const finalMetrics = orchestrator.getPerformanceMetrics();
      expect(finalMetrics.length).toBeGreaterThanOrEqual(initialMetrics.length);

      if (finalMetrics.length > 0) {
        const latestMetric = finalMetrics[finalMetrics.length - 1];
        expect(latestMetric.timestamp).toBeInstanceOf(Date);
        expect(typeof latestMetric.latency).toBe('number');
        expect(typeof latestMetric.throughput).toBe('number');
        expect(typeof latestMetric.resourceUsage).toBe('number');
      }
    });

    it('should handle concurrent operations', async () => {
      const concurrentOps = 5;
      const operations = Array(concurrentOps).fill(0).map((_, index) =>
        orchestrator.processLegalDocument(`Concurrent test document ${index}`, {
          priority: TaskPriority.NORMAL,
          timeout: 5000
        })
      );

      const results = await Promise.allSettled(operations);

      expect(results).toHaveLength(concurrentOps);

      // At least some operations should succeed
      const successfulOps = results.filter(result =>
        result.status === 'fulfilled' && result.value.success
      );
      expect(successfulOps.length).toBeGreaterThan(0);
    });
  });

  describe('Integration Workflow End-to-End', () => {
    it('should complete a full workflow with document → inference → GPU operations', async () => {
      // Step 1: Process a legal document
      const documentResult = await orchestrator.processLegalDocument(
        'This is a comprehensive legal document for end-to-end testing.',
        { analysisType: 'comprehensive' }
      );

      expect(documentResult.success).toBe(true);

      // Step 2: Perform neural inference based on document features
      const features = new Float32Array(32);
      for (let i = 0; i < features.length; i++) {
        features[i] = Math.random();
      }

      const inferenceResult = await orchestrator.performNeuralInference(features);
      expect(inferenceResult.success).toBe(true);

      // Step 3: Execute GPU computation for final analysis
      const computationResult = await orchestrator.executeGPUComputation('analysis', {
        documentFeatures: Array.from(features),
        analysisType: 'legal_classification'
      });

      expect(computationResult.success).toBe(true);

      // Verify the full workflow
      const totalTime = documentResult.processingTime +
                       inferenceResult.processingTime +
                       computationResult.processingTime;

      expect(totalTime).toBeGreaterThan(0);

      // Verify all major services were utilized
      const allServicesUsed = [
        ...documentResult.servicesUsed,
        ...inferenceResult.servicesUsed,
        ...computationResult.servicesUsed
      ];

      expect(allServicesUsed).toContain('wasmGPU');
      expect(allServicesUsed.length).toBeGreaterThanOrEqual(3);
    });

    it('should maintain system health throughout intensive operations', async () => {
      const initialHealth = orchestrator.getSystemHealth();

      // Perform intensive operations
      const intensiveOps = Array(10).fill(0).map((_, i) => [
        orchestrator.processLegalDocument(`Intensive document ${i}`),
        orchestrator.performNeuralInference(new Float32Array(Array(64).fill(0).map(() => Math.random()))),
        orchestrator.executeGPUComputation('stress_test', { iteration: i })
      ]).flat();

      await Promise.allSettled(intensiveOps);

      // Wait for health check to update
      await new Promise(resolve => setTimeout(resolve, 1000));

      const finalHealth = orchestrator.getSystemHealth();

      // System should still be operational
      expect(['healthy', 'degraded']).toContain(finalHealth.overall);
      expect(finalHealth.performance.averageLatency).toBeGreaterThan(0);
    });
  });
});

// Additional utility functions for integration testing
export const IntegrationTestUtils = {
  createTestDocument: (type: 'contract' | 'legal' | 'technical' = 'legal') => {
    const templates = {
      contract: `
        SERVICE AGREEMENT CONTRACT

        This Service Agreement ("Agreement") is entered into on [DATE] between:

        CLIENT: [Client Name]
        ADDRESS: [Client Address]

        PROVIDER: [Provider Name]
        ADDRESS: [Provider Address]

        TERMS AND CONDITIONS:
        1. Scope of Work: Provider will deliver services as outlined in Schedule A
        2. Payment Terms: Net 30 days from invoice date
        3. Term: This agreement is effective for 12 months
        4. Termination: Either party may terminate with 30 days notice

        Both parties agree to the terms set forth above.
      `,
      legal: `
        LEGAL MEMORANDUM

        TO: Legal Team
        FROM: Senior Attorney
        RE: Case Analysis and Recommendations

        ISSUE: Whether the defendant's actions constitute breach of fiduciary duty.

        ANALYSIS: Based on the evidence presented, the following factors support
        a finding of breach of fiduciary duty:
        1. Defendant had a duty of loyalty to plaintiff
        2. Defendant's actions were self-dealing in nature
        3. Plaintiff suffered damages as a direct result

        CONCLUSION: Strong likelihood of successful claim for breach of fiduciary duty.
      `,
      technical: `
        TECHNICAL SPECIFICATION DOCUMENT

        System: Legal AI Processing Platform
        Version: 2.0

        ARCHITECTURE OVERVIEW:
        - Frontend: SvelteKit 2 with TypeScript
        - Backend: Go microservices with gRPC/QUIC
        - Database: PostgreSQL with pgvector
        - AI/ML: Ollama integration with GPU acceleration

        PERFORMANCE REQUIREMENTS:
        - Response time: < 500ms for simple queries
        - Throughput: > 100 requests/second
        - Availability: 99.9% uptime

        GPU INTEGRATION:
        - CUDA support for RTX 3060 Ti
        - WebAssembly modules for browser processing
        - Fallback to CPU processing when GPU unavailable
      `
    };

    return templates[type];
  },

  createTestTensor: (dimensions: number[], fillValue: 'random' | 'sequential' | 'zeros' = 'random') => {
    const size = dimensions.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);

    switch (fillValue) {
      case 'random':
        for (let i = 0; i < size; i++) {
          data[i] = Math.random() * 2 - 1; // Range -1 to 1
        }
        break;
      case 'sequential':
        for (let i = 0; i < size; i++) {
          data[i] = i / size;
        }
        break;
      case 'zeros':
        data.fill(0);
        break;
    }

    return data;
  },

  createTestCanvas: (width: number, height: number, pattern: 'gradient' | 'checkerboard' | 'noise' = 'gradient'): CanvasState => {
    const data = new Uint8ClampedArray(width * height * 4);

    switch (pattern) {
      case 'gradient':
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            data[index] = (x / width) * 255;     // Red
            data[index + 1] = (y / height) * 255; // Green
            data[index + 2] = ((x + y) / (width + height)) * 255; // Blue
            data[index + 3] = 255; // Alpha
          }
        }
        break;
      case 'checkerboard':
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const isWhite = (Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0;
            const color = isWhite ? 255 : 0;
            data[index] = color;     // Red
            data[index + 1] = color; // Green
            data[index + 2] = color; // Blue
            data[index + 3] = 255;   // Alpha
          }
        }
        break;
      case 'noise':
        for (let i = 0; i < data.length; i += 4) {
          data[i] = Math.random() * 255;     // Red
          data[i + 1] = Math.random() * 255; // Green
          data[i + 2] = Math.random() * 255; // Blue
          data[i + 3] = 255;                 // Alpha
        }
        break;
    }

    return {
      width,
      height,
      data,
      format: 'RGBA'
    };
  }
};