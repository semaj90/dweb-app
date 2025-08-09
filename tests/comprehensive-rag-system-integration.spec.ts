/**
 * Comprehensive RAG System Integration Tests
 * Tests Ollama, SvelteKit 2/Svelte 5, PostgreSQL, pgvector, Drizzle ORM, and worker threads
 */

import { test, expect, type Page } from "@playwright/test";
import { Worker } from "worker_threads";

test.describe("🏛️ Legal AI RAG System - Comprehensive Integration", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;

    // Wait for all services to be ready
    await page.goto("http://localhost:5173");
    await page.waitForLoadState("networkidle");

    // Verify the page loaded correctly
    await expect(page).toHaveTitle(/Legal AI System/);
  });

  test.describe("🚀 System Health & Startup", () => {
    test("should verify all core services are running", async () => {
      // Check SvelteKit frontend
      await expect(page.locator('[data-testid="legal-ai-app"]')).toBeVisible();

      // Check Ollama service health
      const ollamaResponse = await page.request.get(
        "http://localhost:11434/api/tags"
      );
      expect(ollamaResponse.ok()).toBeTruthy();

      // Check Qdrant service health
      const qdrantResponse = await page.request.get(
        "http://localhost:6333/health"
      );
      expect(qdrantResponse.ok()).toBeTruthy();

      // Check PostgreSQL connection via API
      const dbResponse = await page.request.get(
        "http://localhost:5173/api/health/database"
      );
      expect(dbResponse.ok()).toBeTruthy();
    });

    test("should display system status dashboard", async () => {
      await page.goto("http://localhost:5173/status");

      // Check service status indicators
      await expect(
        page.locator('[data-testid="service-ollama"]')
      ).toContainText("✅");
      await expect(
        page.locator('[data-testid="service-qdrant"]')
      ).toContainText("✅");
      await expect(
        page.locator('[data-testid="service-postgres"]')
      ).toContainText("✅");
      await expect(
        page.locator('[data-testid="service-sveltekit"]')
      ).toContainText("✅");
    });
  });

  test.describe("🧠 Worker Threads & SIMD Processing", () => {
    test("should process documents using worker threads", async () => {
      // Navigate to document processing page
      await page.goto("http://localhost:5173/process");

      // Upload test document
      const fileInput = page.locator('[data-testid="document-upload"]');
      await fileInput.setInputFiles({
        name: "test-legal-doc.json",
        mimeType: "application/json",
        buffer: Buffer.from(
          JSON.stringify({
            id: "test-doc-001",
            case_number: "2024-CV-12345",
            document_type: "evidence",
            content:
              "This is a test legal document for worker thread processing. ".repeat(
                100
              ),
            metadata: { category: "testimony", importance: "high" },
          })
        ),
      });

      // Trigger worker thread processing
      await page.click('[data-testid="process-with-workers"]');

      // Wait for processing completion
      await page.waitForSelector('[data-testid="processing-complete"]', {
        timeout: 30000,
      });

      // Verify worker thread was used
      const processingLog = await page
        .locator('[data-testid="processing-log"]')
        .textContent();
      expect(processingLog).toContain("worker thread");
      expect(processingLog).toContain("k-means clustering");
    });

    test("should use SIMD parser for batch document processing", async () => {
      await page.goto("http://localhost:5173/batch-process");

      // Create multiple test documents
      const testDocs = Array.from({ length: 50 }, (_, i) => ({
        id: `batch-doc-${i}`,
        case_number: `2024-CV-${String(i).padStart(5, "0")}`,
        document_type: "evidence",
        content: `Legal document content ${i}. `.repeat(50),
        metadata: { batch: true, index: i },
      }));

      // Upload batch
      await page.locator('[data-testid="batch-upload"]').setInputFiles({
        name: "batch-docs.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(testDocs)),
      });

      // Process with SIMD optimization
      await page.click('[data-testid="process-simd"]');

      // Wait for completion and verify SIMD was used
      await page.waitForSelector('[data-testid="simd-complete"]', {
        timeout: 60000,
      });

      const simdStats = await page
        .locator('[data-testid="simd-stats"]')
        .textContent();
      expect(simdStats).toContain("SIMD buffers");
      expect(simdStats).toContain("vectorized operations");
    });
  });

  test.describe("🗄️ PostgreSQL & pgvector Integration", () => {
    test("should store and retrieve document embeddings", async () => {
      await page.goto("http://localhost:5173/documents");

      // Create a test document with embedding
      const testDoc = {
        id: "embed-test-001",
        content:
          "Test document for embedding storage and retrieval using pgvector.",
        case_number: "2024-CV-99999",
      };

      // Submit document
      await page.fill('[data-testid="doc-content"]', testDoc.content);
      await page.fill('[data-testid="doc-case-number"]', testDoc.case_number);
      await page.click('[data-testid="submit-document"]');

      // Wait for embedding generation and storage
      await page.waitForSelector('[data-testid="embedding-stored"]', {
        timeout: 30000,
      });

      // Verify embedding was stored in PostgreSQL
      const dbResponse = await page.request.post(
        "http://localhost:5173/api/embeddings/verify",
        {
          data: { documentId: testDoc.id },
        }
      );
      expect(dbResponse.ok()).toBeTruthy();

      const dbResult = await dbResponse.json();
      expect(dbResult.hasEmbedding).toBeTruthy();
      expect(dbResult.vectorDimensions).toBe(384); // nomic-embed-text dimensions
    });

    test("should perform vector similarity search", async () => {
      await page.goto("http://localhost:5173/search");

      // Perform semantic search
      const searchQuery = "legal contract terms and conditions";
      await page.fill('[data-testid="search-input"]', searchQuery);
      await page.click('[data-testid="vector-search"]');

      // Wait for vector search results
      await page.waitForSelector('[data-testid="search-results"]', {
        timeout: 15000,
      });

      // Verify vector search functionality
      const results = await page
        .locator('[data-testid="search-result-item"]')
        .count();
      expect(results).toBeGreaterThan(0);

      // Check similarity scores
      const firstResult = page
        .locator('[data-testid="search-result-item"]')
        .first();
      const similarityScore = await firstResult
        .locator('[data-testid="similarity-score"]')
        .textContent();
      expect(parseFloat(similarityScore || "0")).toBeGreaterThan(0);
    });

    test("should use Drizzle ORM for database operations", async () => {
      // Test Drizzle ORM integration via API
      const drizzleResponse = await page.request.post(
        "http://localhost:5173/api/drizzle/test",
        {
          data: {
            operation: "createDocument",
            document: {
              id: "drizzle-test-001",
              content: "Testing Drizzle ORM integration",
              case_number: "2024-CV-88888",
              document_type: "test",
            },
          },
        }
      );

      expect(drizzleResponse.ok()).toBeTruthy();

      const result = await drizzleResponse.json();
      expect(result.success).toBeTruthy();
      expect(result.documentId).toBe("drizzle-test-001");
    });
  });

  test.describe("🤖 Ollama AI Integration", () => {
    test("should connect to local Ollama instance", async () => {
      await page.goto("http://localhost:5173/ai-chat");

      // Verify Ollama connection status
      await expect(page.locator('[data-testid="ollama-status"]')).toContainText(
        "Connected"
      );

      // Check available models
      const modelsList = page.locator('[data-testid="available-models"] li');
      await expect(modelsList).toHaveCount(1);

      // Verify required models are available
      await expect(
        page.locator('[data-testid="model-llama3.1:8b"]')
      ).toBeVisible();
      await expect(
        page.locator('[data-testid="model-nomic-embed-text"]')
      ).toBeVisible();
    });

    test("should generate AI responses using local models", async () => {
      await page.goto("http://localhost:5173/ai-chat");

      // Select model
      await page.selectOption('[data-testid="model-selector"]', "llama3.1:8b");

      // Send test query
      const testQuery =
        "Explain the legal concept of due process in simple terms.";
      await page.fill('[data-testid="chat-input"]', testQuery);
      await page.click('[data-testid="send-message"]');

      // Wait for AI response
      await page.waitForSelector('[data-testid="ai-response"]', {
        timeout: 60000,
      });

      // Verify response quality
      const response = await page
        .locator('[data-testid="ai-response"]')
        .textContent();
      expect(response).toBeTruthy();
      expect(response!.length).toBeGreaterThan(50);
      expect(response).toContain("due process");
    });

    test("should track token usage and performance", async () => {
      await page.goto("http://localhost:5173/ai-chat");

      // Enable token tracking
      await page.check('[data-testid="enable-token-tracking"]');

      // Send query and track tokens
      await page.fill(
        '[data-testid="chat-input"]',
        "What are the basic principles of contract law?"
      );
      await page.click('[data-testid="send-message"]');

      // Wait for response and token count
      await page.waitForSelector('[data-testid="token-usage"]', {
        timeout: 60000,
      });

      // Verify token tracking
      const tokenUsage = await page
        .locator('[data-testid="token-usage"]')
        .textContent();
      expect(tokenUsage).toMatch(/Prompt tokens: \d+/);
      expect(tokenUsage).toMatch(/Response tokens: \d+/);
      expect(tokenUsage).toMatch(/Total tokens: \d+/);

      // Check performance metrics
      const responseTime = await page
        .locator('[data-testid="response-time"]')
        .textContent();
      expect(responseTime).toMatch(/\d+ms/);
    });
  });

  test.describe("🔍 RAG (Retrieval-Augmented Generation)", () => {
    test("should perform end-to-end RAG workflow", async () => {
      await page.goto("http://localhost:5173/rag-studio");

      // Step 1: Upload legal documents
      const legalDocs = [
        {
          id: "contract-001",
          content:
            "This is a legal contract between parties for software development services.",
          type: "contract",
        },
        {
          id: "case-law-001",
          content:
            "In this landmark case, the court ruled on intellectual property rights.",
          type: "case_law",
        },
      ];

      for (const doc of legalDocs) {
        await page.fill('[data-testid="doc-upload-content"]', doc.content);
        await page.selectOption('[data-testid="doc-type-select"]', doc.type);
        await page.click('[data-testid="upload-document"]');
        await page.waitForSelector('[data-testid="upload-success"]');
      }

      // Step 2: Wait for embeddings generation
      await page.waitForSelector('[data-testid="embeddings-ready"]', {
        timeout: 30000,
      });

      // Step 3: Perform RAG query
      const ragQuery =
        "What are the key terms in software development contracts?";
      await page.fill('[data-testid="rag-query"]', ragQuery);
      await page.click('[data-testid="execute-rag"]');

      // Step 4: Verify RAG response
      await page.waitForSelector('[data-testid="rag-response"]', {
        timeout: 60000,
      });

      const ragResponse = await page
        .locator('[data-testid="rag-response"]')
        .textContent();
      expect(ragResponse).toBeTruthy();
      expect(ragResponse).toContain("contract");

      // Step 5: Check retrieved context
      const retrievedDocs = page.locator('[data-testid="retrieved-document"]');
      await expect(retrievedDocs).toHaveCount(1);

      // Verify relevance scores
      const relevanceScore = await page
        .locator('[data-testid="relevance-score"]')
        .first()
        .textContent();
      expect(parseFloat(relevanceScore || "0")).toBeGreaterThan(0.5);
    });

    test("should optimize retrieval using memory optimization", async () => {
      await page.goto("http://localhost:5173/rag-studio");

      // Enable advanced memory optimization
      await page.check('[data-testid="enable-memory-optimization"]');
      await page.check('[data-testid="enable-clustering"]');

      // Upload large document set
      const largeDocs = Array.from({ length: 100 }, (_, i) => ({
        id: `large-doc-${i}`,
        content:
          `Legal document ${i} containing various legal concepts and precedents. `.repeat(
            20
          ),
        type: "general",
      }));

      // Batch upload
      await page.locator('[data-testid="batch-upload-json"]').setInputFiles({
        name: "large-docs.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(largeDocs)),
      });

      await page.click('[data-testid="process-batch"]');

      // Wait for clustering and optimization
      await page.waitForSelector('[data-testid="clustering-complete"]', {
        timeout: 120000,
      });

      // Verify memory optimization was applied
      const optimizationStats = await page
        .locator('[data-testid="optimization-stats"]')
        .textContent();
      expect(optimizationStats).toContain("clusters created");
      expect(optimizationStats).toContain("memory pressure");

      // Test optimized retrieval
      await page.fill(
        '[data-testid="rag-query"]',
        "legal precedents and concepts"
      );
      await page.click('[data-testid="execute-optimized-rag"]');

      await page.waitForSelector('[data-testid="optimized-results"]', {
        timeout: 30000,
      });

      // Verify faster response time with optimization
      const responseTime = await page
        .locator('[data-testid="optimized-response-time"]')
        .textContent();
      const timeMs = parseInt(responseTime?.match(/(\d+)ms/)?.[1] || "0");
      expect(timeMs).toBeLessThan(5000); // Should be under 5 seconds with optimization
    });
  });

  test.describe("🎮 NVIDIA CUDA & GPU Acceleration", () => {
    test("should detect and utilize GPU acceleration", async () => {
      await page.goto("http://localhost:5173/gpu-status");

      // Check GPU detection
      const gpuStatus = await page
        .locator('[data-testid="gpu-status"]')
        .textContent();

      if (gpuStatus?.includes("NVIDIA")) {
        // GPU available - test acceleration
        await expect(
          page.locator('[data-testid="cuda-available"]')
        ).toBeVisible();

        // Enable GPU acceleration for Ollama
        await page.check('[data-testid="enable-gpu-acceleration"]');

        // Test GPU-accelerated inference
        await page.goto("http://localhost:5173/ai-chat");
        await page.selectOption(
          '[data-testid="model-selector"]',
          "llama3.1:8b"
        );

        // Send complex query that benefits from GPU acceleration
        const complexQuery =
          "Analyze the following legal scenario and provide detailed recommendations: " +
          "A multinational corporation is facing intellectual property disputes across multiple jurisdictions.";

        await page.fill('[data-testid="chat-input"]', complexQuery);
        await page.click('[data-testid="send-message"]');

        // Wait for GPU-accelerated response
        await page.waitForSelector('[data-testid="ai-response"]', {
          timeout: 120000,
        });

        // Verify GPU was used
        const performanceMetrics = await page
          .locator('[data-testid="performance-metrics"]')
          .textContent();
        expect(performanceMetrics).toContain("GPU");

        // Check inference speed improvement
        const inferenceTime = await page
          .locator('[data-testid="inference-time"]')
          .textContent();
        const timeMs = parseInt(inferenceTime?.match(/(\d+)ms/)?.[1] || "0");
        expect(timeMs).toBeLessThan(30000); // Should be faster with GPU
      } else {
        // No GPU - verify CPU fallback works
        await expect(
          page.locator('[data-testid="cpu-fallback"]')
        ).toBeVisible();
        console.log("GPU not available - testing CPU fallback");
      }
    });

    test("should handle memory optimization with GPU constraints", async () => {
      await page.goto("http://localhost:5173/memory-optimization");

      // Configure memory optimization for GPU environment
      await page.fill('[data-testid="gpu-memory-limit"]', "8192"); // 8GB
      await page.fill('[data-testid="cpu-memory-limit"]', "16384"); // 16GB

      // Enable GPU-aware memory optimization
      await page.check('[data-testid="gpu-aware-optimization"]');

      // Start optimization test
      await page.click('[data-testid="start-memory-optimization"]');

      // Wait for optimization completion
      await page.waitForSelector('[data-testid="optimization-complete"]', {
        timeout: 60000,
      });

      // Verify optimization results
      const memoryStats = await page
        .locator('[data-testid="memory-optimization-stats"]')
        .textContent();
      expect(memoryStats).toContain("GPU memory");
      expect(memoryStats).toContain("CPU memory");
      expect(memoryStats).toContain("optimization level");
    });
  });

  test.describe("📊 Performance & Benchmarking", () => {
    test("should run comprehensive performance benchmarks", async () => {
      await page.goto("http://localhost:5173/benchmark");

      // Configure benchmark parameters
      await page.fill('[data-testid="document-count"]', "1000");
      await page.fill('[data-testid="query-count"]', "100");
      await page.check('[data-testid="include-clustering"]');
      await page.check('[data-testid="include-embeddings"]');

      // Start benchmark
      await page.click('[data-testid="start-benchmark"]');

      // Wait for benchmark completion (this may take several minutes)
      await page.waitForSelector('[data-testid="benchmark-complete"]', {
        timeout: 600000,
      });

      // Verify benchmark results
      const results = await page
        .locator('[data-testid="benchmark-results"]')
        .textContent();
      expect(results).toContain("Documents processed");
      expect(results).toContain("Average response time");
      expect(results).toContain("Memory usage");
      expect(results).toContain("GPU utilization");

      // Check performance thresholds
      const avgResponseTime = await page
        .locator('[data-testid="avg-response-time"]')
        .textContent();
      const responseTimeMs = parseInt(
        avgResponseTime?.match(/(\d+)ms/)?.[1] || "0"
      );
      expect(responseTimeMs).toBeLessThan(10000); // Under 10 seconds average

      const memoryUsage = await page
        .locator('[data-testid="peak-memory-usage"]')
        .textContent();
      const memoryMB = parseInt(memoryUsage?.match(/(\d+)MB/)?.[1] || "0");
      expect(memoryMB).toBeLessThan(8192); // Under 8GB peak memory
    });
  });

  test.describe("🛡️ Error Handling & Recovery", () => {
    test("should handle Ollama service interruption", async () => {
      await page.goto("http://localhost:5173/ai-chat");

      // Simulate Ollama service interruption
      await page.route("http://localhost:11434/**", (route: any) => route.abort());

      // Attempt to send message
      await page.fill(
        '[data-testid="chat-input"]',
        "Test message during service interruption"
      );
      await page.click('[data-testid="send-message"]');

      // Should show error state
      await page.waitForSelector('[data-testid="ollama-error"]', {
        timeout: 10000,
      });
      await expect(page.locator('[data-testid="ollama-error"]')).toContainText(
        "service unavailable"
      );

      // Should show retry option
      await expect(
        page.locator('[data-testid="retry-connection"]')
      ).toBeVisible();

      // Restore service and retry
      await page.unroute("http://localhost:11434/**");
      await page.click('[data-testid="retry-connection"]');

      // Should recover
      await page.waitForSelector('[data-testid="ollama-status"]', {
        timeout: 15000,
      });
      await expect(page.locator('[data-testid="ollama-status"]')).toContainText(
        "Connected"
      );
    });

    test("should handle memory pressure gracefully", async () => {
      await page.goto("http://localhost:5173/stress-test");

      // Trigger high memory usage
      await page.click('[data-testid="trigger-memory-pressure"]');

      // Should activate memory optimization
      await page.waitForSelector('[data-testid="memory-optimization-active"]', {
        timeout: 30000,
      });

      // Should reduce LOD (Level of Detail)
      const lodLevel = await page
        .locator('[data-testid="current-lod"]')
        .textContent();
      expect(lodLevel).toMatch(/low|medium/);

      // Should clear low-priority caches
      const cacheStatus = await page
        .locator('[data-testid="cache-status"]')
        .textContent();
      expect(cacheStatus).toContain("optimized");

      // System should remain responsive
      await page.fill(
        '[data-testid="test-input"]',
        "System responsiveness test"
      );
      await page.click('[data-testid="test-submit"]');
      await page.waitForSelector('[data-testid="response-received"]', {
        timeout: 5000,
      });
    });
  });

  test.describe("🔒 Security & Validation", () => {
    test("should validate input sanitization", async () => {
      await page.goto("http://localhost:5173/documents");

      // Test SQL injection prevention
      const maliciousInput = "'; DROP TABLE documents; --";
      await page.fill('[data-testid="doc-content"]', maliciousInput);
      await page.click('[data-testid="submit-document"]');

      // Should sanitize input
      await page.waitForSelector('[data-testid="document-saved"]', {
        timeout: 10000,
      });

      // Verify database integrity
      const dbResponse = await page.request.get(
        "http://localhost:5173/api/health/database"
      );
      expect(dbResponse.ok()).toBeTruthy();
    });

    test("should validate document upload restrictions", async () => {
      await page.goto("http://localhost:5173/upload");

      // Test file size limit
      const largeFakeFile = {
        name: "large-file.json",
        mimeType: "application/json",
        buffer: Buffer.alloc(100 * 1024 * 1024), // 100MB
      };

      await page
        .locator('[data-testid="file-upload"]')
        .setInputFiles(largeFakeFile);
      await expect(
        page.locator('[data-testid="file-too-large"]')
      ).toBeVisible();

      // Test file type validation
      const invalidFile = {
        name: "malicious.exe",
        mimeType: "application/octet-stream",
        buffer: Buffer.from("fake executable"),
      };

      await page
        .locator('[data-testid="file-upload"]')
        .setInputFiles(invalidFile);
      await expect(
        page.locator('[data-testid="invalid-file-type"]')
      ).toBeVisible();
    });
  });

  test.afterEach(async () => {
    // Cleanup after each test
    await page.evaluate(() => {
      // Clear any remaining intervals or timeouts
      window.location.reload();
    });
  });
});

// Helper test for verifying the one-click launcher worked
test.describe("🚀 One-Click Launcher Verification", () => {
  test("should verify all services started by launcher", async ({ page }) => {
    // This test verifies that our one-click launcher successfully started all services

    // Test each service endpoint
    const services = [
      { name: "SvelteKit", url: "http://localhost:5173", expected: 200 },
      { name: "Ollama", url: "http://localhost:11434/api/tags", expected: 200 },
      { name: "Qdrant", url: "http://localhost:6333/health", expected: 200 },
      {
        name: "PostgreSQL",
        url: "http://localhost:5173/api/health/database",
        expected: 200,
      },
    ];

    for (const service of services) {
      const response = await page.request.get(service.url);
      expect(response.status(), `${service.name} should be running`).toBe(
        service.expected
      );
    }

    console.log("✅ All services are running - one-click launcher successful!");
  });
});
