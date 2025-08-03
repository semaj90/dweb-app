#!/usr/bin/env node

/**
 * Package.json Scripts Verification
 * Validates all Ollama and SvelteKit integration scripts are properly configured
 */

import { readFileSync } from "fs";
import { join } from "path";

console.log("🔍 Verifying package.json scripts configuration...\n");

try {
  const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
  const scripts = packageJson.scripts;

  const expectedScripts = {
    // Core development
    start: "Development environment starter",
    "start:quick": "Quick frontend-only start",
    "start:gpu": "GPU-accelerated development",
    "start:prod": "Production deployment",
    dev: "SvelteKit development server",
    "dev:frontend": "Frontend development only",
    "dev:full": "Full stack development",
    "dev:gpu": "GPU-accelerated full stack",
    "dev:quick": "Quick development mode",

    // Ollama management
    "ollama:start": "Start Ollama service",
    "ollama:stop": "Stop Ollama service",
    "ollama:restart": "Restart Ollama service",
    "ollama:status": "Check Ollama status",
    "ollama:health": "Ollama health check",
    "ollama:models": "List available models",
    "ollama:setup": "Setup legal AI models",
    "ollama:gpu": "Start Ollama with GPU",

    // Docker management
    "docker:up": "Start Docker services",
    "docker:down": "Stop Docker services",
    "docker:status": "Check Docker status",
    "docker:cli": "PowerShell Docker CLI",
    "docker:wsl": "WSL Docker manager",

    // Testing
    "test:integration": "Integration tests",
    "test:quick": "Quick tests",
    "ai:test": "AI pipeline tests",
    health: "System health check",

    // Deployment
    "deploy:prod": "Production deployment",
    "deploy:gpu": "GPU production deployment",
    "deploy:optimized": "Optimized deployment",
  };

  console.log("📋 Script Verification Results:\n");

  let allPresent = true;
  let configuredCount = 0;

  for (const [scriptName, description] of Object.entries(expectedScripts)) {
    if (scripts[scriptName]) {
      console.log(`✅ ${scriptName.padEnd(20)} - ${description}`);
      configuredCount++;
    } else {
      console.log(`❌ ${scriptName.padEnd(20)} - Missing: ${description}`);
      allPresent = false;
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log("📊 VERIFICATION SUMMARY");
  console.log("=".repeat(60));

  console.log(
    `Scripts Configured: ${configuredCount}/${Object.keys(expectedScripts).length}`
  );
  console.log(
    `Status: ${allPresent ? "✅ ALL SCRIPTS PRESENT" : "❌ MISSING SCRIPTS"}`
  );

  if (allPresent) {
    console.log("\n🎉 All integration scripts are properly configured!");
    console.log("\n🚀 Quick Commands to Try:");
    console.log("   npm run ollama:health    - Check Ollama status");
    console.log("   npm run start:quick      - Quick frontend start");
    console.log("   npm run test:integration - Test full integration");
    console.log("   npm start                - Full development environment");
  } else {
    console.log("\n⚠️  Some scripts are missing from package.json");
    console.log("   Please ensure all expected scripts are configured");
  }

  // Additional validation
  console.log("\n📦 Dependencies Check:");
  const deps = { ...packageJson.dependencies, ...packageJson.devDependencies };

  const expectedDeps = [
    "concurrently",
    "@langchain/ollama",
    "drizzle-orm",
    "pgvector",
  ];

  expectedDeps.forEach((dep) => {
    if (deps[dep]) {
      console.log(`✅ ${dep} - ${deps[dep]}`);
    } else {
      console.log(`❌ ${dep} - Missing dependency`);
    }
  });

  console.log("\n" + "=".repeat(60));

  process.exit(allPresent ? 0 : 1);
} catch (error) {
  console.error("❌ Error reading package.json:", error.message);
  process.exit(1);
}
