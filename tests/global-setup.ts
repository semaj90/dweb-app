import { chromium, type FullConfig } from "@playwright/test";

/**
 * Global Setup for Legal AI RAG Testing
 * Initializes services and prepares test environment
 */

async function globalSetup(config: FullConfig) {
  console.log("🚀 Starting Legal AI RAG Test Environment Setup...");

  try {
    // Launch browser for setup tasks
    const browser = await chromium.launch();
    const page = await browser.newPage();

    // Step 1: Check if development server is running
    console.log("📡 Checking development server...");
    try {
      await page.goto("http://localhost:5173", { timeout: 10000 });
      console.log("✅ Development server is running");
    } catch (error) {
      console.log(
        "⚠️  Development server not detected, tests will start it automatically"
      );
    }

    // Step 2: Verify Ollama service
    console.log("🤖 Checking Ollama service...");
    try {
      const response = await page.request.get(
        "http://localhost:11434/api/tags"
      );
      if (response.ok()) {
        console.log("✅ Ollama service is running");

        const data = await response.json();
        const modelCount = data.models?.length || 0;
        console.log(`📚 Available models: ${modelCount}`);

        if (modelCount === 0) {
          console.log("⚠️  No models loaded, some tests may be skipped");
        }
      } else {
        console.log("❌ Ollama service not responding");
      }
    } catch (error) {
      console.log("❌ Ollama service not available");
    }

    // Step 3: Check Docker services
    console.log("🐳 Checking Docker services...");
    try {
      const healthResponse = await page.request.get(
        "http://localhost:5173/api/health"
      );
      if (healthResponse.ok()) {
        const healthData = await healthResponse.json();
        console.log("✅ Backend services accessible");

        if (healthData.services) {
          Object.entries(healthData.services).forEach(([service, status]) => {
            console.log(`   ${service}: ${status ? "✅" : "❌"}`);
          });
        }
      }
    } catch (error) {
      console.log("⚠️  Backend health check failed, some tests may fail");
    }

    // Step 4: Check GPU availability
    console.log("🎮 Checking GPU availability...");
    try {
      const gpuResponse = await page.request.get(
        "http://localhost:5173/api/system/gpu"
      );
      const gpuData = await gpuResponse.json();

      if (gpuData.cuda_available) {
        console.log("✅ NVIDIA CUDA GPU detected");
        console.log(`   GPU Count: ${gpuData.gpu_count}`);
        console.log(`   Driver Version: ${gpuData.driver_version}`);
      } else {
        console.log("⚠️  No CUDA GPU detected, GPU tests will be skipped");
      }
    } catch (error) {
      console.log("⚠️  GPU status check failed");
    }

    // Step 5: Initialize test database
    console.log("🗄️  Preparing test database...");
    try {
      const dbResponse = await page.request.post(
        "http://localhost:5173/api/test/db/init"
      );
      if (dbResponse.ok()) {
        console.log("✅ Test database initialized");
      }
    } catch (error) {
      console.log("⚠️  Test database initialization failed");
    }

    // Step 6: Create test data
    console.log("📄 Creating test data...");
    await createTestData(page);

    await browser.close();

    console.log("🎉 Test environment setup complete!");
    console.log("=".repeat(60));
  } catch (error) {
    console.error("❌ Global setup failed:", error);
    throw error;
  }
}

async function createTestData(page: any) {
  try {
    // Create test legal documents for RAG testing
    const testDocuments = [
      {
        id: "test-doc-1",
        title: "Contract Law Basics",
        content:
          "A contract is a legally binding agreement between two or more parties...",
        type: "legal-guide",
        metadata: { category: "contract-law", difficulty: "basic" },
      },
      {
        id: "test-doc-2",
        title: "Criminal Procedure Overview",
        content:
          "Criminal procedure governs the process by which criminal charges are investigated...",
        type: "legal-guide",
        metadata: { category: "criminal-law", difficulty: "intermediate" },
      },
      {
        id: "test-doc-3",
        title: "Evidence Admissibility Rules",
        content:
          "The rules of evidence determine what information can be presented in court...",
        type: "legal-guide",
        metadata: { category: "evidence-law", difficulty: "advanced" },
      },
    ];

    for (const doc of testDocuments) {
      await page.request.post("http://localhost:5173/api/test/documents", {
        data: doc,
      });
    }

    console.log("✅ Test documents created");

    // Create test vector embeddings
    await page.request.post("http://localhost:5173/api/test/vectors/populate");
    console.log("✅ Test vector embeddings created");
  } catch (error) {
    console.log("⚠️  Test data creation failed:", error);
  }
}

export default globalSetup;
