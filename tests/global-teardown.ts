import { chromium } from "@playwright/test";

/**
 * Global Teardown for Legal AI RAG Testing
 * Cleans up test environment and resources
 */

async function globalTeardown(): Promise<any> {
  console.log("🧹 Starting Legal AI RAG Test Environment Cleanup...");

  try {
    // Launch browser for cleanup tasks
    const browser = await chromium.launch();
    const page = await browser.newPage();

    // Step 1: Clean up test data
    console.log("🗑️  Cleaning up test data...");
    try {
      await page.request.delete("http://localhost:5173/api/test/cleanup");
      console.log("✅ Test data cleaned up");
    } catch (error: any) {
      console.log("⚠️  Test data cleanup failed:", error);
    }

    // Step 2: Clear vector embeddings
    console.log("🔄 Clearing test vector embeddings...");
    try {
      await page.request.delete("http://localhost:5173/api/test/vectors/clear");
      console.log("✅ Vector embeddings cleared");
    } catch (error: any) {
      console.log("⚠️  Vector cleanup failed:", error);
    }

    // Step 3: Reset token counters
    console.log("🔢 Resetting token counters...");
    try {
      await page.request.post("http://localhost:5173/api/test/tokens/reset");
      console.log("✅ Token counters reset");
    } catch (error: any) {
      console.log("⚠️  Token reset failed:", error);
    }

    // Step 4: Clear GPU memory (if available)
    console.log("🎮 Clearing GPU memory...");
    try {
      await page.request.post("http://localhost:5173/api/system/gpu/cleanup");
      console.log("✅ GPU memory cleared");
    } catch (error: any) {
      console.log("⚠️  GPU cleanup not available");
    }

    // Step 5: Generate test report summary
    console.log("📊 Generating test summary...");
    await generateTestSummary(page);

    await browser.close();

    console.log("✨ Test environment cleanup complete!");
    console.log("=".repeat(60));
  } catch (error: any) {
    console.error("❌ Global teardown failed:", error);
    // Don't throw error to avoid failing the test suite
  }
}

async function generateTestSummary(page: unknown): Promise<any> {
  try {
    const summaryResponse = await page.request.get(
      "http://localhost:5173/api/test/summary"
    );

    if (summaryResponse.ok()) {
      const summary = await summaryResponse.json();

      console.log("\n📋 Test Execution Summary:");
      console.log(`   Total API Calls: ${summary.total_requests || 0}`);
      console.log(`   Tokens Used: ${summary.total_tokens || 0}`);
      console.log(`   GPU Accelerated Requests: ${summary.gpu_requests || 0}`);
      console.log(`   RAG Queries: ${summary.rag_queries || 0}`);
      console.log(
        `   Average Response Time: ${summary.avg_response_time || 0}ms`
      );

      if (summary.performance_metrics) {
        console.log("\n⚡ Performance Metrics:");
        console.log(
          `   Peak Memory Usage: ${summary.performance_metrics.peak_memory || 0}MB`
        );
        console.log(
          `   GPU Memory Peak: ${summary.performance_metrics.gpu_memory_peak || 0}MB`
        );
        console.log(
          `   Cache Hit Rate: ${summary.performance_metrics.cache_hit_rate || 0}%`
        );
      }
    }
  } catch (error: any) {
    console.log("⚠️  Could not generate test summary");
  }
}

export default globalTeardown;
