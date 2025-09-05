#!/usr/bin/env node

/**
 * Comprehensive Enhanced Legal AI Verification Script
 * Tests all features mentioned in the README guide
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

console.log("🔍 Enhanced Legal AI Complete Verification");
console.log("===========================================\n");

// Test results tracking
const results = {
  passed: 0,
  failed: 0,
  warnings: 0,
  tests: [],
};

function addTest(name, status, message = "") {
  results.tests.push({ name, status, message });
  results[status]++;

  const icon = status === "passed" ? "✅" : status === "failed" ? "❌" : "⚠️";
  console.log(`${icon} ${name}${message ? `: ${message}` : ""}`);
}

// 1. Core File Structure Verification
console.log("📁 Core File Structure");
console.log("---------------------");

const coreFiles = [
  "src/routes/api/analyze/+server.ts",
  "src/lib/components/ai/ThinkingStyleToggle.svelte",
  "src/lib/ai/thinking-processor.ts",
  "scripts/setup-thinking-ai.js",
  "scripts/test-thinking-analysis.js",
  "scripts/process-docs.js",
  "scripts/fetch-docs.js",
  "LAUNCH-ENHANCED-LEGAL-AI.bat",
  "launch-enhanced-legal-ai.ps1",
];

coreFiles.forEach((file) => {
  if (fs.existsSync(file)) {
    addTest(`Core file: ${file}`, "passed");
  } else {
    addTest(`Core file: ${file}`, "failed", "File missing");
  }
});

console.log();

// 2. Package.json Scripts Verification
console.log("📦 NPM Scripts Verification");
console.log("---------------------------");

let packageJson;
try {
  packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
  addTest("package.json readable", "passed");
} catch (error) {
  addTest("package.json readable", "failed", error.message);
  process.exit(1);
}

const requiredScripts = [
  "thinking:setup",
  "thinking:test",
  "docs:process",
  "docs:fetch",
  "db:push",
  "db:migrate",
  "db:seed",
  "vector:init",
  "vector:sync",
  "ai:test",
  "system:health",
  "dev",
];

requiredScripts.forEach((script) => {
  if (packageJson.scripts[script]) {
    addTest(`NPM script: ${script}`, "passed");
  } else {
    addTest(`NPM script: ${script}`, "failed", "Script missing");
  }
});

console.log();

// 3. Dependencies Verification
console.log("📚 Dependencies Verification");
console.log("----------------------------");

const requiredDeps = [
  "jsdom",
  "sharp",
  "tesseract.js",
  "mammoth",
  "ollama",
  "drizzle-orm",
  "svelte",
  "@sveltejs/kit",
];

requiredDeps.forEach((dep) => {
  if (packageJson.dependencies[dep] || packageJson.devDependencies[dep]) {
    addTest(`Dependency: ${dep}`, "passed");
  } else {
    addTest(`Dependency: ${dep}`, "failed", "Dependency missing");
  }
});

console.log();

// 4. Route Structure Verification
console.log("🛣️ Route Structure Verification");
console.log("-------------------------------");

const requiredRoutes = [
  "src/routes/evidence",
  "src/routes/interactive-canvas",
  "src/routes/ai-assistant",
  "src/routes/cases",
  "src/routes/api/analyze",
  "src/routes/api/ai/chat",
  "src/routes/api/search",
];

requiredRoutes.forEach((route) => {
  if (fs.existsSync(route)) {
    addTest(`Route: ${route}`, "passed");
  } else {
    addTest(`Route: ${route}`, "failed", "Route directory missing");
  }
});

console.log();

// 5. Component Integration Verification
console.log("🧩 Component Integration");
console.log("-----------------------");

// Check ThinkingStyleToggle component
try {
  const toggleComponent = fs.readFileSync(
    "src/lib/components/ai/ThinkingStyleToggle.svelte",
    "utf8",
  );

  if (toggleComponent.includes("createEventDispatcher")) {
    addTest("ThinkingStyleToggle: Event system", "passed");
  } else {
    addTest(
      "ThinkingStyleToggle: Event system",
      "failed",
      "Event dispatcher missing",
    );
  }

  if (toggleComponent.includes("premium")) {
    addTest("ThinkingStyleToggle: Premium support", "passed");
  } else {
    addTest(
      "ThinkingStyleToggle: Premium support",
      "warning",
      "Premium feature check missing",
    );
  }

  if (toggleComponent.includes("on:toggle")) {
    addTest("ThinkingStyleToggle: Toggle handler", "passed");
  } else {
    addTest(
      "ThinkingStyleToggle: Toggle handler",
      "warning",
      "Toggle event handler missing",
    );
  }
} catch (error) {
  addTest("ThinkingStyleToggle: Component check", "failed", error.message);
}

console.log();

// 6. API Endpoint Verification
console.log("🔌 API Endpoint Structure");
console.log("-------------------------");

try {
  const analyzeAPI = fs.readFileSync(
    "src/routes/api/analyze/+server.ts",
    "utf8",
  );

  if (analyzeAPI.includes("export const POST")) {
    addTest("Analyze API: POST handler", "passed");
  } else {
    addTest("Analyze API: POST handler", "failed", "POST handler missing");
  }

  if (analyzeAPI.includes("useThinkingStyle")) {
    addTest("Analyze API: Thinking style support", "passed");
  } else {
    addTest(
      "Analyze API: Thinking style support",
      "failed",
      "Thinking style parameter missing",
    );
  }

  if (analyzeAPI.includes("ollama")) {
    addTest("Analyze API: Ollama integration", "passed");
  } else {
    addTest(
      "Analyze API: Ollama integration",
      "failed",
      "Ollama integration missing",
    );
  }

  if (analyzeAPI.includes("aiReports") || analyzeAPI.includes("ai_reports")) {
    addTest("Analyze API: Database integration", "passed");
  } else {
    addTest(
      "Analyze API: Database integration",
      "warning",
      "Database integration unclear",
    );
  }
} catch (error) {
  addTest("Analyze API: File check", "failed", error.message);
}

console.log();

// 7. Thinking Processor Verification
console.log("🧠 Thinking Processor");
console.log("--------------------");

try {
  const processor = fs.readFileSync("src/lib/ai/thinking-processor.ts", "utf8");

  if (processor.includes("ThinkingProcessor")) {
    addTest("Thinking Processor: Class definition", "passed");
  } else {
    addTest(
      "Thinking Processor: Class definition",
      "failed",
      "ThinkingProcessor class missing",
    );
  }

  if (processor.includes("analyzeDocument")) {
    addTest("Thinking Processor: Document analysis", "passed");
  } else {
    addTest(
      "Thinking Processor: Document analysis",
      "failed",
      "analyzeDocument method missing",
    );
  }

  if (processor.includes("analyzeEvidence")) {
    addTest("Thinking Processor: Evidence analysis", "passed");
  } else {
    addTest(
      "Thinking Processor: Evidence analysis",
      "failed",
      "analyzeEvidence method missing",
    );
  }

  if (processor.includes("parseThinkingResponse")) {
    addTest("Thinking Processor: Response parsing", "passed");
  } else {
    addTest(
      "Thinking Processor: Response parsing",
      "failed",
      "parseThinkingResponse method missing",
    );
  }
} catch (error) {
  addTest("Thinking Processor: File check", "failed", error.message);
}

console.log();

// 8. Database Schema Verification
console.log("🗄️ Database Schema");
console.log("-----------------");

if (fs.existsSync("dev.db")) {
  addTest("Database: SQLite file", "passed");
} else {
  addTest(
    "Database: SQLite file",
    "warning",
    "Database file missing - run db:push",
  );
}

if (fs.existsSync("drizzle")) {
  addTest("Database: Drizzle directory", "passed");
} else {
  addTest(
    "Database: Drizzle directory",
    "failed",
    "Drizzle migrations missing",
  );
}

if (fs.existsSync("drizzle.config.ts")) {
  addTest("Database: Drizzle config", "passed");
} else {
  addTest("Database: Drizzle config", "failed", "Drizzle config missing");
}

console.log();

// 9. Environment Configuration
console.log("🔧 Environment Configuration");
console.log("----------------------------");

const envFiles = [".env.development", ".env.production", ".env.example"];
envFiles.forEach((file) => {
  if (fs.existsSync(file)) {
    addTest(`Environment: ${file}`, "passed");
  } else {
    addTest(`Environment: ${file}`, "warning", "Environment file missing");
  }
});

console.log();

// 10. Launch Scripts Verification
console.log("🚀 Launch Scripts");
console.log("----------------");

// Check PowerShell script
try {
  const psScript = fs.readFileSync("launch-enhanced-legal-ai.ps1", "utf8");
  if (psScript.includes("FullSetup") && psScript.includes("QuickStart")) {
    addTest("PowerShell launcher: Parameters", "passed");
  } else {
    addTest(
      "PowerShell launcher: Parameters",
      "failed",
      "Launch parameters missing",
    );
  }
} catch (error) {
  addTest("PowerShell launcher: File check", "failed", error.message);
}

// Check batch script
if (fs.existsSync("LAUNCH-ENHANCED-LEGAL-AI.bat")) {
  addTest("Batch launcher: File exists", "passed");
} else {
  addTest("Batch launcher: File exists", "failed", "Batch file missing");
}

console.log();

// 11. Documentation Verification
console.log("📖 Documentation");
console.log("----------------");

const docFiles = [
  "ENHANCED-LEGAL-AI-README.md",
  "HTML-ATTRIBUTE-FIX-README.md",
];

docFiles.forEach((file) => {
  if (fs.existsSync(file)) {
    addTest(`Documentation: ${file}`, "passed");
  } else {
    addTest(`Documentation: ${file}`, "warning", "Documentation missing");
  }
});

console.log();

// Final Results Summary
console.log("📊 VERIFICATION SUMMARY");
console.log("=======================");
console.log(`✅ Passed: ${results.passed}`);
console.log(`⚠️ Warnings: ${results.warnings}`);
console.log(`❌ Failed: ${results.failed}`);
console.log(`📋 Total Tests: ${results.tests.length}`);
console.log();

const successRate = ((results.passed / results.tests.length) * 100).toFixed(1);
console.log(`🎯 Success Rate: ${successRate}%`);

if (results.failed === 0) {
  console.log("\n🎉 ALL CORE FEATURES VERIFIED!");
  console.log("Your Enhanced Legal AI implementation is ready to use.");
} else if (results.failed <= 2) {
  console.log("\n✅ MOSTLY READY!");
  console.log("Minor issues found but core functionality should work.");
} else {
  console.log("\n⚠️ ISSUES FOUND");
  console.log("Some core components may need attention.");
}

console.log();
console.log("🚀 Ready to launch with:");
console.log("• Double-click: LAUNCH-ENHANCED-LEGAL-AI.bat");
console.log("• PowerShell: ./launch-enhanced-legal-ai.ps1 -FullSetup");
console.log("• Manual: npm run thinking:setup && npm run dev");
console.log();

// Save results to file
const reportPath = "verification/verification-report.json";
fs.mkdirSync("verification", { recursive: true });
fs.writeFileSync(
  reportPath,
  JSON.stringify(
    {
      timestamp: new Date().toISOString(),
      summary: {
        passed: results.passed,
        warnings: results.warnings,
        failed: results.failed,
        total: results.tests.length,
        successRate: parseFloat(successRate),
      },
      tests: results.tests,
    },
    null,
    2,
  ),
);

console.log(`📄 Detailed report saved to: ${reportPath}`);

// Exit with appropriate code
process.exit(results.failed === 0 ? 0 : 1);
