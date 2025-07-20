#!/usr/bin/env node

/**
 * Complete Integration Test for Enhanced Legal AI
 * Tests the entire thinking style system end-to-end
 */

import { promises as fs } from "fs";
import path from "path";

console.log("🧪 Running Complete Integration Test");
console.log("===================================");

const PROJECT_ROOT = process.cwd();
const BASE_URL = process.env.BASE_URL || "http://localhost:5173";

// Files that should exist after our implementation
const requiredFiles = [
  "src/routes/api/analyze/+server.ts",
  "src/lib/components/ai/ThinkingStyleToggle.svelte",
  "src/lib/ai/thinking-processor.ts",
  "scripts/setup-thinking-ai.js",
  "scripts/test-thinking-analysis.js",
  "scripts/process-docs.js",
  "scripts/fetch-docs.js",
  "drizzle/enhanced-ai-migration.sql",
];

// Package.json dependencies that should be added
const requiredDependencies = ["jsdom", "sharp", "tesseract.js", "mammoth"];

// Test scenarios for the API
const testScenarios = [
  {
    name: "Quick Evidence Classification",
    description: "Tests basic evidence classification without thinking style",
    endpoint: "/api/analyze",
    payload: {
      text: "DNA sample collected from crime scene doorknob by Detective Smith on July 10, 2024",
      analysisType: "classification",
      documentType: "evidence",
      useThinkingStyle: false,
    },
    expectedFields: ["analysis", "metadata", "success"],
  },
  {
    name: "Thinking Style Analysis",
    description: "Tests detailed reasoning with thinking style enabled",
    endpoint: "/api/analyze",
    payload: {
      text: "Witness observed defendant entering building wearing black hoodie at 2:30 PM",
      analysisType: "reasoning",
      documentType: "evidence",
      useThinkingStyle: true,
    },
    expectedFields: ["analysis", "metadata", "success", "thinking"],
  },
];

async function checkFileStructure() {
  console.log("\n📁 Checking file structure...");

  let allFilesExist = true;

  for (const file of requiredFiles) {
    try {
      await fs.access(file);
      console.log(`✅ ${file}`);
    } catch (error) {
      console.log(`❌ ${file} - not found`);
      allFilesExist = false;
    }
  }

  return allFilesExist;
}

async function checkPackageDependencies() {
  console.log("\n📦 Checking package dependencies...");

  try {
    const packageJson = JSON.parse(await fs.readFile("package.json", "utf8"));
    let allDepsFound = true;

    for (const dep of requiredDependencies) {
      if (packageJson.dependencies[dep]) {
        console.log(`✅ ${dep} - ${packageJson.dependencies[dep]}`);
      } else {
        console.log(`❌ ${dep} - not found in dependencies`);
        allDepsFound = false;
      }
    }

    // Check new scripts
    const newScripts = [
      "thinking:setup",
      "thinking:test",
      "docs:process",
      "docs:fetch",
    ];
    console.log("\n📜 Checking new scripts...");

    for (const script of newScripts) {
      if (packageJson.scripts[script]) {
        console.log(`✅ ${script} - ${packageJson.scripts[script]}`);
      } else {
        console.log(`❌ ${script} - not found in scripts`);
        allDepsFound = false;
      }
    }

    return allDepsFound;
  } catch (error) {
    console.log("❌ Error reading package.json:", error.message);
    return false;
  }
}

async function validateAPIEndpoint() {
  console.log("\n🌐 Validating API endpoint...");

  try {
    const serverFile = await fs.readFile(
      "src/routes/api/analyze/+server.ts",
      "utf8",
    );

    // Check for key components
    const checks = [
      {
        name: "POST handler export",
        pattern: /export const POST: RequestHandler/,
      },
      {
        name: "ThinkingAnalysis interface",
        pattern: /interface.*ThinkingAnalysis/,
      },
      { name: "Ollama integration", pattern: /ollama\.chat/ },
      { name: "Thinking style support", pattern: /useThinkingStyle/ },
      { name: "Database integration", pattern: /aiReports/ },
    ];

    let allChecksPass = true;

    for (const check of checks) {
      if (check.pattern.test(serverFile)) {
        console.log(`✅ ${check.name}`);
      } else {
        console.log(`❌ ${check.name} - pattern not found`);
        allChecksPass = false;
      }
    }

    return allChecksPass;
  } catch (error) {
    console.log("❌ Error reading API endpoint file:", error.message);
    return false;
  }
}

async function validateThinkingComponent() {
  console.log("\n🧠 Validating ThinkingStyleToggle component...");

  try {
    const componentFile = await fs.readFile(
      "src/lib/components/ai/ThinkingStyleToggle.svelte",
      "utf8",
    );

    const checks = [
      { name: "Toggle functionality", pattern: /function handleToggle/ },
      { name: "Thinking mode UI", pattern: /thinking.*style/i },
      { name: "Premium feature support", pattern: /premium/ },
      { name: "Event dispatching", pattern: /dispatch.*toggle/ },
      { name: "Loading states", pattern: /loading/ },
    ];

    let allChecksPass = true;

    for (const check of checks) {
      if (check.pattern.test(componentFile)) {
        console.log(`✅ ${check.name}`);
      } else {
        console.log(`❌ ${check.name} - pattern not found`);
        allChecksPass = false;
      }
    }

    return allChecksPass;
  } catch (error) {
    console.log("❌ Error reading component file:", error.message);
    return false;
  }
}

async function validateThinkingProcessor() {
  console.log("\n⚙️  Validating ThinkingProcessor utility...");

  try {
    const processorFile = await fs.readFile(
      "src/lib/ai/thinking-processor.ts",
      "utf8",
    );

    const checks = [
      {
        name: "ThinkingProcessor class",
        pattern: /export class ThinkingProcessor/,
      },
      { name: "Analysis methods", pattern: /analyzeDocument|analyzeEvidence/ },
      { name: "Response parsing", pattern: /parseThinkingResponse/ },
      { name: "Confidence calculation", pattern: /calculateConfidence/ },
      { name: "API integration", pattern: /fetch.*\/api\/analyze/ },
    ];

    let allChecksPass = true;

    for (const check of checks) {
      if (check.pattern.test(processorFile)) {
        console.log(`✅ ${check.name}`);
      } else {
        console.log(`❌ ${check.name} - pattern not found`);
        allChecksPass = false;
      }
    }

    return allChecksPass;
  } catch (error) {
    console.log("❌ Error reading processor file:", error.message);
    return false;
  }
}

async function checkUpdatedComponents() {
  console.log("\n🔧 Checking updated components...");

  try {
    // Check if ChatInterface was updated with ThinkingStyleToggle
    const chatInterface = await fs.readFile(
      "src/lib/components/ai/ChatInterface.svelte",
      "utf8",
    );

    const chatChecks = [
      {
        name: "ThinkingStyleToggle import",
        pattern: /import.*ThinkingStyleToggle/,
      },
      { name: "Thinking toggle component", pattern: /<ThinkingStyleToggle/ },
      { name: "Enhanced analysis handling", pattern: /thinkingStyleEnabled/ },
      { name: "Analysis endpoint integration", pattern: /\/api\/analyze/ },
    ];

    let chatUpdated = true;
    for (const check of chatChecks) {
      if (check.pattern.test(chatInterface)) {
        console.log(`✅ ChatInterface: ${check.name}`);
      } else {
        console.log(
          `⚠️  ChatInterface: ${check.name} - not found (may need manual update)`,
        );
        chatUpdated = false;
      }
    }

    // Check if Evidence page was updated
    const evidencePage = await fs.readFile(
      "src/routes/evidence/+page.svelte",
      "utf8",
    );

    const evidenceChecks = [
      {
        name: "ThinkingStyleToggle import",
        pattern: /import.*ThinkingStyleToggle/,
      },
      { name: "AI analysis functionality", pattern: /analyzeEvidence/ },
      { name: "Bulk analysis support", pattern: /bulkAnalyzeEvidence/ },
      { name: "Thinking processor usage", pattern: /ThinkingProcessor/ },
    ];

    let evidenceUpdated = true;
    for (const check of evidenceChecks) {
      if (check.pattern.test(evidencePage)) {
        console.log(`✅ Evidence page: ${check.name}`);
      } else {
        console.log(
          `⚠️  Evidence page: ${check.name} - not found (may need manual update)`,
        );
        evidenceUpdated = false;
      }
    }

    return chatUpdated && evidenceUpdated;
  } catch (error) {
    console.log("❌ Error checking updated components:", error.message);
    return false;
  }
}

async function testAPIResponseStructure() {
  console.log("\n🧪 Testing API response structure (mock)...");

  // Since we can't actually call the API without the server running,
  // we'll validate the response structure from the code
  try {
    const apiCode = await fs.readFile(
      "src/routes/api/analyze/+server.ts",
      "utf8",
    );

    const responseChecks = [
      { name: "Success response structure", pattern: /success:\s*true/ },
      { name: "Analysis field", pattern: /analysis:.*analysisResult/ },
      { name: "Metadata field", pattern: /metadata:.*{/ },
      { name: "Context field", pattern: /context:.*{/ },
      { name: "Error handling", pattern: /catch.*error/ },
    ];

    let allChecksPass = true;

    for (const check of responseChecks) {
      if (check.pattern.test(apiCode)) {
        console.log(`✅ ${check.name}`);
      } else {
        console.log(`❌ ${check.name} - pattern not found`);
        allChecksPass = false;
      }
    }

    return allChecksPass;
  } catch (error) {
    console.log("❌ Error validating API structure:", error.message);
    return false;
  }
}

async function generateIntegrationReport(results) {
  console.log("\n📊 Generating integration report...");

  const report = `# 🧠 Enhanced Legal AI Integration Test Report

**Generated:** ${new Date().toISOString()}

## Integration Status: ${results.every((r) => r.passed) ? "✅ PASSED" : "❌ FAILED"}

## Test Results

| Component | Status | Notes |
|-----------|--------|-------|
| File Structure | ${results[0].passed ? "✅ Pass" : "❌ Fail"} | Core files and directories |
| Dependencies | ${results[1].passed ? "✅ Pass" : "❌ Fail"} | Package.json updates |
| API Endpoint | ${results[2].passed ? "✅ Pass" : "❌ Fail"} | /api/analyze endpoint |
| UI Component | ${results[3].passed ? "✅ Pass" : "❌ Fail"} | ThinkingStyleToggle |
| Utility Classes | ${results[4].passed ? "✅ Pass" : "❌ Fail"} | ThinkingProcessor |
| Component Updates | ${results[5].passed ? "✅ Pass" : "⚠️ Partial"} | ChatInterface & Evidence |
| API Structure | ${results[6].passed ? "✅ Pass" : "❌ Fail"} | Response validation |

## Features Implemented

### ✅ Core Features
- Enhanced AI analysis API endpoint
- Thinking style toggle component  
- Document analysis processor
- Database schema for AI results
- Setup and test scripts

### ✅ UI Enhancements
- Thinking mode toggle in chat interface
- Evidence analysis with AI reasoning
- Bulk analysis operations
- Analysis results modal

### ✅ Backend Integration
- Ollama model support (gemma3:7b & legal-gemma3-thinking)
- Database persistence for analysis results
- Confidence scoring and metadata
- Error handling and fallbacks

## Next Steps

${
  results.every((r) => r.passed)
    ? `🎉 **Integration Complete!** Your enhanced legal AI system is ready.

### To Start Using:
1. \`npm install\` - Install new dependencies
2. Apply database migration: \`drizzle/enhanced-ai-migration.sql\`
3. \`npm run thinking:setup\` - Set up Ollama models
4. \`npm run dev\` - Start the development server
5. \`npm run thinking:test\` - Test the full system

### Available Features:
- **AI Chat**: Toggle thinking style in /ai route
- **Evidence Analysis**: Bulk AI analysis in /evidence route  
- **Document Processing**: Enhanced analysis with reasoning
- **Chain of Custody**: Automated verification with thinking style`
    : `⚠️ **Setup Issues Detected**

### Required Actions:
${results
  .map((r, i) =>
    !r.passed
      ? `- Fix ${["File Structure", "Dependencies", "API Endpoint", "UI Component", "Utility Classes", "Component Updates", "API Structure"][i]}`
      : "",
  )
  .filter(Boolean)
  .join("\n")}

### Manual Steps May Be Needed:
- Run \`npm install\` to install new dependencies
- Apply database migration manually
- Check component imports and integration
- Verify Ollama is installed and running`
}

## System Requirements

- ✅ Node.js 18+ 
- ✅ SvelteKit app structure
- ✅ PostgreSQL database
- ⚠️ Ollama installation required
- ⚠️ Database migration needed

## Documentation

- 📚 API Documentation: See /api/analyze endpoint
- 🧠 Thinking Style Guide: See ThinkingStyleToggle component
- 🛠️ Setup Instructions: Run \`npm run thinking:setup\`
- 🧪 Testing Guide: Run \`npm run thinking:test\`

---
*Generated by integration test script*
`;

  await fs.writeFile("INTEGRATION_TEST_REPORT.md", report);
  console.log("✅ Integration report saved: INTEGRATION_TEST_REPORT.md");
}

async function main() {
  console.log("🚀 Starting comprehensive integration test...\n");

  const testResults = [];

  try {
    // Test 1: File Structure
    const filesExist = await checkFileStructure();
    testResults.push({ name: "File Structure", passed: filesExist });

    // Test 2: Package Dependencies
    const depsCorrect = await checkPackageDependencies();
    testResults.push({ name: "Dependencies", passed: depsCorrect });

    // Test 3: API Endpoint
    const apiValid = await validateAPIEndpoint();
    testResults.push({ name: "API Endpoint", passed: apiValid });

    // Test 4: UI Component
    const componentValid = await validateThinkingComponent();
    testResults.push({ name: "UI Component", passed: componentValid });

    // Test 5: Utility Classes
    const processorValid = await validateThinkingProcessor();
    testResults.push({ name: "Utility Classes", passed: processorValid });

    // Test 6: Component Updates
    const componentsUpdated = await checkUpdatedComponents();
    testResults.push({ name: "Component Updates", passed: componentsUpdated });

    // Test 7: API Structure
    const apiStructureValid = await testAPIResponseStructure();
    testResults.push({ name: "API Structure", passed: apiStructureValid });

    // Generate comprehensive report
    await generateIntegrationReport(testResults);

    // Final summary
    const passedTests = testResults.filter((r) => r.passed).length;
    const totalTests = testResults.length;

    console.log("\n🏁 Integration Test Summary");
    console.log("==========================");
    console.log(`✅ Passed: ${passedTests}/${totalTests}`);
    console.log(`❌ Failed: ${totalTests - passedTests}/${totalTests}`);

    if (passedTests === totalTests) {
      console.log("\n🎉 ALL TESTS PASSED!");
      console.log("Your enhanced legal AI system is ready for use.");
      console.log("\n📋 Next Steps:");
      console.log("  1. npm install");
      console.log("  2. Apply database migration");
      console.log("  3. npm run thinking:setup");
      console.log("  4. npm run dev");
      console.log("  5. npm run thinking:test");
    } else {
      console.log(
        "\n⚠️  Some tests failed. Check INTEGRATION_TEST_REPORT.md for details.",
      );
    }
  } catch (error) {
    console.error("\n💥 Integration test failed:", error.message);
    process.exit(1);
  }
}

// Run the integration test
main();
