#!/usr/bin/env node

/**
 * Quick Ollama + SvelteKit Integration Test
 * Tests the complete pipeline from API to frontend
 */

console.log("🧪 Testing Ollama + SvelteKit Integration...\n");

const config = {
  healthUrl: "http://localhost:5173/api/ai/health",
  chatUrl: "http://localhost:5173/api/ai/chat",
  frontendUrl: "http://localhost:5173",
  demoUrl: "http://localhost:5173/ai-demo",
};

async function testHealth() {
  console.log("🔍 Testing health endpoint...");

  try {
    const response = await fetch(config.healthUrl);
    const data = await response.json();

    console.log(`✅ Health check: ${data.status}`);
    console.log(
      `   Ollama: ${data.services.ollama?.healthy ? "✅ Healthy" : "❌ Down"}`
    );
    console.log(
      `   Models: ${data.services.ollama?.models?.length || 0} available`
    );
    console.log(`   Memory: ${data.services.system?.memory || "unknown"}`);

    return data.services.ollama?.healthy || false;
  } catch (error) {
    console.log("❌ Health check failed:", error.message);
    return false;
  }
}

async function testChat() {
  console.log("\n🔍 Testing chat endpoint...");

  try {
    const response = await fetch(config.chatUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "What is a criminal prosecution?",
        model: "gemma3-legal",
        useRAG: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    console.log("✅ Chat test successful");
    console.log(`   Model: ${data.model}`);
    console.log(`   Response length: ${data.response?.length || 0} characters`);
    console.log(`   Duration: ${data.performance?.duration || 0}ms`);
    console.log(`   Tokens/sec: ${data.performance?.tokensPerSecond || 0}`);

    return true;
  } catch (error) {
    console.log("❌ Chat test failed:", error.message);
    return false;
  }
}

async function testFrontend() {
  console.log("\n🔍 Testing frontend availability...");

  try {
    const response = await fetch(config.frontendUrl);

    if (response.ok) {
      console.log("✅ Frontend accessible");
      return true;
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    console.log("❌ Frontend not accessible:", error.message);
    return false;
  }
}

async function runTests() {
  const results = {
    health: await testHealth(),
    frontend: await testFrontend(),
    chat: false,
  };

  // Only test chat if health check passed
  if (results.health) {
    results.chat = await testChat();
  }

  console.log("\n" + "=".repeat(50));
  console.log("📊 TEST SUMMARY");
  console.log("=".repeat(50));

  console.log(`Health Check: ${results.health ? "✅ PASS" : "❌ FAIL"}`);
  console.log(`Frontend:     ${results.frontend ? "✅ PASS" : "❌ FAIL"}`);
  console.log(`Chat API:     ${results.chat ? "✅ PASS" : "❌ FAIL"}`);

  const allPassed = Object.values(results).every(Boolean);

  if (allPassed) {
    console.log("\n🎉 All tests passed! System is ready.");
    console.log("\n🔗 Access URLs:");
    console.log(`   Frontend: ${config.frontendUrl}`);
    console.log(`   AI Demo:  ${config.demoUrl}`);
  } else {
    console.log("\n⚠️  Some tests failed. Check the services:");
    if (!results.health) console.log("   - Run: npm run ollama:start");
    if (!results.frontend) console.log("   - Run: npm run dev");
    if (!results.chat)
      console.log("   - Check Ollama models: npm run ollama:models");
  }

  console.log("\n" + "=".repeat(50));

  process.exit(allPassed ? 0 : 1);
}

// Add timeout to prevent hanging
const timeout = setTimeout(() => {
  console.log("❌ Tests timed out after 30 seconds");
  process.exit(1);
}, 30000);

runTests().finally(() => clearTimeout(timeout));
