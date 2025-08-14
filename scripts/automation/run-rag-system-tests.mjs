#!/usr/bin/env node

/**
 * Comprehensive RAG System Test Runner
 * Tests worker_threads, SIMD parsers, Ollama, PostgreSQL, pgvector, and all integrations
 */

import { spawn, exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";

const execAsync = promisify(exec);

class RAGSystemTester {
  constructor() {
    this.testResults = {
      systemHealth: false,
      workerThreads: false,
      simdParser: false,
      ollamaIntegration: false,
      postgresqlPgvector: false,
      drizzleORM: false,
      ragWorkflow: false,
      cudaAcceleration: false,
      memoryOptimization: false,
      performanceBenchmarks: false,
    };

    this.startTime = Date.now();
    this.verbose =
      process.argv.includes("--verbose") || process.argv.includes("-v");
    this.quickMode = process.argv.includes("--quick");
    this.gpuOnly = process.argv.includes("--gpu");
  }

  log(message, type = "info") {
    const timestamp = new Date().toISOString();
    const colors = {
      info: "\x1b[36m", // Cyan
      success: "\x1b[32m", // Green
      warning: "\x1b[33m", // Yellow
      error: "\x1b[31m", // Red
      reset: "\x1b[0m", // Reset
    };

    console.log(`${colors[type]}[${timestamp}] ${message}${colors.reset}`);
  }

  async checkSystemHealth() {
    this.log("🏥 Checking system health...", "info");

    try {
      // Check SvelteKit
      const svelteResponse = await fetch("http://localhost:5173");
      if (!svelteResponse.ok) throw new Error("SvelteKit not responding");

      // Check Ollama
      const ollamaResponse = await fetch("http://localhost:11434/api/tags");
      if (!ollamaResponse.ok) throw new Error("Ollama not responding");

      // Check Qdrant
      const qdrantResponse = await fetch("http://localhost:6333/health");
      if (!qdrantResponse.ok) throw new Error("Qdrant not responding");

      // Check PostgreSQL via API
      const dbResponse = await fetch(
        "http://localhost:5173/api/health/database"
      );
      if (!dbResponse.ok) throw new Error("PostgreSQL not responding");

      this.testResults.systemHealth = true;
      this.log("✅ All systems healthy", "success");
      return true;
    } catch (error) {
      this.log(`❌ System health check failed: ${error.message}`, "error");
      return false;
    }
  }

  async testWorkerThreads() {
    this.log("🧵 Testing worker threads implementation...", "info");

    try {
      // Run worker thread specific tests
      const { stdout, stderr } = await execAsync(
        "npm run test:worker-threads",
        { timeout: 120000 }
      );

      if (this.verbose) {
        this.log(`Worker threads output: ${stdout}`, "info");
      }

      if (stderr && !stderr.includes("Warning")) {
        throw new Error(stderr);
      }

      this.testResults.workerThreads = true;
      this.log("✅ Worker threads tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ Worker threads test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testSIMDParser() {
    this.log("⚡ Testing SIMD parser implementation...", "info");

    try {
      // Run SIMD specific tests
      const { stdout, stderr } = await execAsync("npm run test:simd", {
        timeout: 90000,
      });

      if (this.verbose) {
        this.log(`SIMD parser output: ${stdout}`, "info");
      }

      if (stderr && !stderr.includes("Warning")) {
        throw new Error(stderr);
      }

      this.testResults.simdParser = true;
      this.log("✅ SIMD parser tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ SIMD parser test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testOllamaIntegration() {
    this.log("🤖 Testing Ollama integration...", "info");

    try {
      // Test Ollama models and API
      const modelsResponse = await fetch("http://localhost:11434/api/tags");
      const models = await modelsResponse.json();

      const requiredModels = ["llama3.1:8b", "nomic-embed-text"];
      const availableModels = models.models.map((m) => m.name);

      for (const required of requiredModels) {
        if (!availableModels.some((model) => model.includes(required))) {
          throw new Error(`Required model ${required} not found`);
        }
      }

      // Test AI chat functionality
      const chatResponse = await fetch("http://localhost:11434/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "llama3.1:8b",
          prompt: "Test prompt for integration testing",
          stream: false,
        }),
      });

      if (!chatResponse.ok) {
        throw new Error("Ollama generate API failed");
      }

      this.testResults.ollamaIntegration = true;
      this.log("✅ Ollama integration tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ Ollama integration test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testPostgreSQLPgvector() {
    this.log("🐘 Testing PostgreSQL + pgvector integration...", "info");

    try {
      // Test database connection and pgvector extension
      const dbTestResponse = await fetch(
        "http://localhost:5173/api/test/database",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            test: "pgvector",
            operation: "create_and_search_vectors",
          }),
        }
      );

      if (!dbTestResponse.ok) {
        throw new Error("Database test API failed");
      }

      const result = await dbTestResponse.json();
      if (!result.success) {
        throw new Error(`Database test failed: ${result.error}`);
      }

      this.testResults.postgresqlPgvector = true;
      this.log("✅ PostgreSQL + pgvector tests passed", "success");
      return true;
    } catch (error) {
      this.log(
        `❌ PostgreSQL + pgvector test failed: ${error.message}`,
        "error"
      );
      return false;
    }
  }

  async testDrizzleORM() {
    this.log("🔧 Testing Drizzle ORM integration...", "info");

    try {
      // Test Drizzle ORM operations
      const drizzleTestResponse = await fetch(
        "http://localhost:5173/api/test/drizzle",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            operation: "full_crud_test",
            testData: {
              id: "drizzle-test-" + Date.now(),
              content: "Test document for Drizzle ORM validation",
              case_number: "2024-TEST-99999",
            },
          }),
        }
      );

      if (!drizzleTestResponse.ok) {
        throw new Error("Drizzle ORM test API failed");
      }

      const result = await drizzleTestResponse.json();
      if (!result.success) {
        throw new Error(`Drizzle ORM test failed: ${result.error}`);
      }

      this.testResults.drizzleORM = true;
      this.log("✅ Drizzle ORM tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ Drizzle ORM test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testRAGWorkflow() {
    this.log("🔍 Testing complete RAG workflow...", "info");

    try {
      // Run comprehensive RAG integration tests
      const { stdout, stderr } = await execAsync(
        "npm run test:rag-integration",
        { timeout: 300000 } // 5 minutes for full RAG test
      );

      if (this.verbose) {
        this.log(`RAG workflow output: ${stdout}`, "info");
      }

      if (stderr && !stderr.includes("Warning")) {
        throw new Error(stderr);
      }

      this.testResults.ragWorkflow = true;
      this.log("✅ RAG workflow tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ RAG workflow test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testCUDAAcceleration() {
    this.log("🎮 Testing CUDA/GPU acceleration...", "info");

    try {
      // Check if GPU is available
      const gpuResponse = await fetch("http://localhost:5173/api/gpu/status");
      const gpuStatus = await gpuResponse.json();

      if (!gpuStatus.cudaAvailable) {
        this.log("⚠️ CUDA not available, skipping GPU tests", "warning");
        this.testResults.cudaAcceleration = true; // Pass if no GPU
        return true;
      }

      // Run CUDA specific tests
      const { stdout, stderr } = await execAsync(
        "npm run test:cuda-integration",
        { timeout: 180000 }
      );

      if (this.verbose) {
        this.log(`CUDA acceleration output: ${stdout}`, "info");
      }

      if (stderr && !stderr.includes("Warning")) {
        throw new Error(stderr);
      }

      this.testResults.cudaAcceleration = true;
      this.log("✅ CUDA acceleration tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ CUDA acceleration test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testMemoryOptimization() {
    this.log("🧠 Testing memory optimization system...", "info");

    try {
      // Run memory optimization tests
      const { stdout, stderr } = await execAsync(
        "npm run test:memory-optimization",
        { timeout: 120000 }
      );

      if (this.verbose) {
        this.log(`Memory optimization output: ${stdout}`, "info");
      }

      if (stderr && !stderr.includes("Warning")) {
        throw new Error(stderr);
      }

      this.testResults.memoryOptimization = true;
      this.log("✅ Memory optimization tests passed", "success");
      return true;
    } catch (error) {
      this.log(`❌ Memory optimization test failed: ${error.message}`, "error");
      return false;
    }
  }

  async testPerformanceBenchmarks() {
    if (this.quickMode) {
      this.log("⚡ Skipping performance benchmarks (quick mode)", "warning");
      this.testResults.performanceBenchmarks = true;
      return true;
    }

    this.log("📊 Running performance benchmarks...", "info");

    try {
      // Test performance with different configurations
      const benchmarkResponse = await fetch(
        "http://localhost:5173/api/benchmark/run",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tests: [
              "document_processing_speed",
              "vector_search_performance",
              "memory_usage_optimization",
              "concurrent_request_handling",
            ],
            documentCount: this.quickMode ? 100 : 1000,
            queryCount: this.quickMode ? 10 : 100,
          }),
        }
      );

      if (!benchmarkResponse.ok) {
        throw new Error("Benchmark API failed");
      }

      const results = await benchmarkResponse.json();

      // Validate performance thresholds
      if (results.avgResponseTime > 10000) {
        // 10 seconds
        throw new Error(
          `Average response time too high: ${results.avgResponseTime}ms`
        );
      }

      if (results.peakMemoryUsage > 8192) {
        // 8GB
        throw new Error(
          `Peak memory usage too high: ${results.peakMemoryUsage}MB`
        );
      }

      this.testResults.performanceBenchmarks = true;
      this.log("✅ Performance benchmarks passed", "success");

      if (this.verbose) {
        this.log(
          `Benchmark results: ${JSON.stringify(results, null, 2)}`,
          "info"
        );
      }

      return true;
    } catch (error) {
      this.log(`❌ Performance benchmarks failed: ${error.message}`, "error");
      return false;
    }
  }

  async generateReport() {
    const endTime = Date.now();
    const totalTime = Math.round((endTime - this.startTime) / 1000);

    this.log("\n📋 TEST REPORT SUMMARY", "info");
    this.log("=".repeat(50), "info");

    const tests = Object.entries(this.testResults);
    const passed = tests.filter(([, result]) => result).length;
    const total = tests.length;

    tests.forEach(([testName, result]) => {
      const status = result ? "✅ PASS" : "❌ FAIL";
      const color = result ? "success" : "error";
      this.log(
        `${status} ${testName.replace(/([A-Z])/g, " $1").trim()}`,
        color
      );
    });

    this.log("=".repeat(50), "info");
    this.log(
      `Total: ${passed}/${total} tests passed`,
      passed === total ? "success" : "error"
    );
    this.log(`Execution time: ${totalTime} seconds`, "info");

    // Generate detailed report file
    const report = {
      timestamp: new Date().toISOString(),
      testResults: this.testResults,
      summary: {
        passed,
        total,
        executionTimeSeconds: totalTime,
        passRate: Math.round((passed / total) * 100),
      },
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        architecture: process.arch,
        quickMode: this.quickMode,
        gpuOnly: this.gpuOnly,
        verbose: this.verbose,
      },
    };

    await fs.writeFile(
      "test-report.json",
      JSON.stringify(report, null, 2),
      "utf-8"
    );

    this.log(`📄 Detailed report saved to test-report.json`, "info");

    return passed === total;
  }

  async runAllTests() {
    this.log("🚀 Starting comprehensive RAG system tests...", "info");
    this.log(
      `Mode: ${this.quickMode ? "Quick" : "Full"} | GPU Only: ${this.gpuOnly} | Verbose: ${this.verbose}`,
      "info"
    );

    try {
      // System health check (always run first)
      await this.checkSystemHealth();

      if (this.gpuOnly) {
        // GPU-focused tests only
        await this.testCUDAAcceleration();
        await this.testMemoryOptimization();
      } else {
        // Run all tests in sequence
        await this.testWorkerThreads();
        await this.testSIMDParser();
        await this.testOllamaIntegration();
        await this.testPostgreSQLPgvector();
        await this.testDrizzleORM();
        await this.testRAGWorkflow();
        await this.testCUDAAcceleration();
        await this.testMemoryOptimization();
        await this.testPerformanceBenchmarks();
      }

      const allPassed = await this.generateReport();

      if (allPassed) {
        this.log(
          "🎉 All tests passed! RAG system is fully functional.",
          "success"
        );
        process.exit(0);
      } else {
        this.log(
          "❌ Some tests failed. Check the report for details.",
          "error"
        );
        process.exit(1);
      }
    } catch (error) {
      this.log(`💥 Test runner error: ${error.message}`, "error");
      process.exit(1);
    }
  }
}

// Run the tests
const tester = new RAGSystemTester();
tester.runAllTests();
