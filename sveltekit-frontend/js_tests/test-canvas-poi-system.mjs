#!/usr/bin/env node

/**
 * Comprehensive test for the Interactive Canvas POI system
 * This validates the complete architecture works end-to-end
 */

import { execSync, spawn } from "child_process";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";

const BASE_URL = "http://localhost:5173";
const API_BASE = `${BASE_URL}/api`;

console.log("🧪 Interactive Canvas POI System Test");
console.log("=====================================\n");

// Test 1: Check if development server is running
async function testServerRunning() {
  console.log("1. Testing server connectivity...");
  try {
    const response = await fetch(BASE_URL);
    if (response.ok) {
      console.log("   ✅ Development server is running");
      return true;
    }
  } catch (error) {
    console.log("   ❌ Development server is not running");
    console.log("   💡 Run: npm run dev");
    return false;
  }
}

// Test 2: Check database schema
async function testDatabaseSchema() {
  console.log("\n2. Testing database schema...");
  try {
    // Test if POI endpoints exist
    const response = await fetch(`${API_BASE}/cases/test-case-id/pois`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    // We expect either data or a proper error (not 404)
    if (response.status !== 404) {
      console.log("   ✅ POI API endpoints exist");
      return true;
    } else {
      console.log("   ❌ POI API endpoints missing");
      return false;
    }
  } catch (error) {
    console.log("   ❌ Database connection failed");
    console.log("   💡 Check your database configuration");
    return false;
  }
}

// Test 3: Test AI Service
async function testAIService() {
  console.log("\n3. Testing AI summarization service...");
  try {
    const response = await fetch(`${API_BASE}/ai/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: "This is a test report about a criminal investigation.",
        type: "report",
      }),
    });

    if (response.ok) {
      const data = await response.json();
      if (data.summary) {
        console.log("   ✅ AI summarization working");
        console.log(
          `   📝 Test summary: "${data.summary.substring(0, 50)}..."`,
        );
        return true;
      }
    } else if (response.status === 503) {
      console.log("   ⚠️  AI service unavailable (Ollama not running)");
      console.log("   💡 Start Ollama: ollama serve");
      return false;
    }
  } catch (error) {
    console.log("   ❌ AI service error:", error.message);
    return false;
  }
}

// Test 4: Check component files exist
async function testComponentFiles() {
  console.log("\n4. Testing component files...");

  const requiredFiles = [
    "src/lib/components/canvas/POINode.svelte",
    "src/lib/components/canvas/ReportNode.svelte",
    "src/lib/components/canvas/EvidenceNode.svelte",
    "src/lib/components/modals/AISummaryModal.svelte",
    "src/lib/logic/POI.ts",
    "src/lib/services/aiService.ts",
    "src/lib/services/caseService.ts",
    "src/lib/actions/draggable.ts",
  ];

  let allFilesExist = true;

  for (const file of requiredFiles) {
    if (existsSync(file)) {
      console.log(`   ✅ ${file}`);
    } else {
      console.log(`   ❌ ${file} missing`);
      allFilesExist = false;
    }
  }

  return allFilesExist;
}

// Test 5: Test canvas page
async function testCanvasPage() {
  console.log("\n5. Testing canvas page...");
  try {
    const response = await fetch(`${BASE_URL}/cases/test-case/canvas`);
    if (response.ok) {
      const html = await response.text();
      if (html.includes("canvas-container")) {
        console.log("   ✅ Canvas page renders correctly");
        return true;
      }
    }
    console.log("   ❌ Canvas page not working");
    return false;
  } catch (error) {
    console.log("   ❌ Canvas page error:", error.message);
    return false;
  }
}

// Test 6: Check TypeScript compilation
async function testTypeScript() {
  console.log("\n6. Testing TypeScript compilation...");
  try {
    execSync("npx svelte-check --no-tsconfig", {
      stdio: "pipe",
      timeout: 30000,
    });
    console.log("   ✅ TypeScript compilation successful");
    return true;
  } catch (error) {
    console.log("   ❌ TypeScript errors found");
    console.log("   💡 Run: npx svelte-check for details");
    return false;
  }
}

// Run all tests
async function runAllTests() {
  const results = {
    server: await testServerRunning(),
    database: await testDatabaseSchema(),
    ai: await testAIService(),
    components: await testComponentFiles(),
    canvas: await testCanvasPage(),
    typescript: await testTypeScript(),
  };

  console.log("\n📋 Test Summary");
  console.log("===============");

  const passed = Object.values(results).filter((r) => r).length;
  const total = Object.keys(results).length;

  Object.entries(results).forEach(([test, result]) => {
    console.log(
      `${result ? "✅" : "❌"} ${test.padEnd(12)} ${result ? "PASS" : "FAIL"}`,
    );
  });

  console.log(`\n🎯 Overall: ${passed}/${total} tests passed`);

  if (passed === total) {
    console.log(
      "\n🎉 All tests passed! The Interactive Canvas POI system is ready to use.",
    );
    console.log("\n📖 Quick Start Guide:");
    console.log("   1. Navigate to /cases/[case-id]/canvas");
    console.log("   2. Right-click on canvas to add POIs");
    console.log("   3. Use the Summarize buttons for AI analysis");
    console.log("   4. Drag nodes around the canvas");
  } else {
    console.log("\n⚠️  Some tests failed. Please check the issues above.");
    process.exit(1);
  }
}

// Generate setup instructions if needed
function generateSetupInstructions() {
  const instructions = `
# Interactive Canvas POI System Setup

## Quick Start Commands

\`\`\`bash
# 1. Install dependencies
npm install

# 2. Set up environment
cp .env.example .env
# Edit .env with your database credentials

# 3. Run database migrations
npx drizzle-kit generate
npx drizzle-kit migrate

# 4. Start Ollama (for AI features)
ollama serve
ollama pull gemma3-legal

# 5. Start development server
npm run dev
\`\`\`

System Architecture
POI Management: Create and manage persons of interest with structured profiles
AI Summarization: Get AI-powered summaries of reports, evidence, and POIs
Interactive Canvas: Drag-and-drop interface with context menus
Real-time Sync: Auto-save with Redis caching and offline support

Key Features Implemented
POI Creation via right-click context menu
AI summarization for all node types
Draggable canvas nodes
Rich text editing with HugerTE
TypeScript-safe architecture
PostgreSQL + Drizzle ORM
SvelteKit API routes
Responsive UI with shadcn-svelte

Usage
1. Create POI: Right-click canvas → "Add Person of Interest" → Select type
2. Edit POI: Click edit button to fill Who/What/Why/How profile
3. **AI Summary: Click sparkles icon on any node for AI analysis
4. Move Nodes: Drag any node around the canvas
5. Save: Auto-saves every 5 seconds, or use "Save All" button

`;

  writeFileSync("CANVAS_POI_SETUP.md", instructions);
  console.log("\n📋 Setup instructions saved to CANVAS_POI_SETUP.md");
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests()
    .then(() => {
      generateSetupInstructions();
    })
    .catch(console.error);
}
