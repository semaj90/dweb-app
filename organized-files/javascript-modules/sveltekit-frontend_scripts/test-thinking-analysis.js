#!/usr/bin/env node

/**
 * Test Script for Enhanced Legal AI Thinking Style Analysis
 * Tests the analyze API endpoint and thinking functionality
 */

import { promises as fs } from "fs";

console.log("🧪 Testing Enhanced Legal AI Analysis...");
console.log("=====================================");

const BASE_URL = process.env.BASE_URL || "http://localhost:5173";
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Test cases for different analysis scenarios
const testCases = [
  {
    name: "Quick Evidence Classification",
    payload: {
      text: "DNA sample collected from crime scene doorknob on July 10, 2024 by Detective Smith. Chain of custody maintained.",
      analysisType: "classification",
      documentType: "evidence",
      useThinkingStyle: false,
    },
  },
  {
    name: "Thinking Style Evidence Analysis",
    payload: {
      text: "Witness statement: John Doe observed defendant entering building at 2:30 PM. Defendant was wearing black hoodie and appeared nervous.",
      analysisType: "reasoning",
      documentType: "evidence",
      useThinkingStyle: true,
    },
  },
  {
    name: "Chain of Custody Verification",
    payload: {
      text: "Evidence ID: EV-001. Collected by Officer Johnson at 14:30. Transported to lab at 15:00. Analyzed by Tech Williams at 16:30.",
      analysisType: "chain_of_custody",
      documentType: "evidence",
      useThinkingStyle: true,
    },
  },
  {
    name: "Legal Compliance Check",
    payload: {
      text: "Search warrant executed at 456 Oak Avenue. Warrant signed by Judge Williams. Items seized: laptop, documents, cell phone.",
      analysisType: "compliance",
      documentType: "legal_document",
      useThinkingStyle: false,
    },
  },
  {
    name: "Complex Case Reasoning",
    payload: {
      text: "Case involves burglary charges. Evidence includes fingerprints (95% match), witness testimony (reliable), and recovered stolen property. Defense argues fingerprints could be from previous legitimate visit.",
      analysisType: "reasoning",
      documentType: "case_file",
      useThinkingStyle: true,
    },
  },
];

async function checkSystemHealth() {
  console.log("🔍 Checking system health...");

  try {
    // Check if SvelteKit app is running
    const appResponse = await fetch(`${BASE_URL}/api/ai/health/local`);
    if (appResponse.ok) {
      console.log("✅ SvelteKit app is running");
    } else {
      throw new Error("SvelteKit app not responding");
    }
  } catch (error) {
    console.log("❌ SvelteKit app not accessible at", BASE_URL);
    console.log("💡 Start the app with: npm run dev");
    return false;
  }

  try {
    // Check Ollama availability
    const ollamaResponse = await fetch(`${OLLAMA_URL}/api/tags`);
    if (ollamaResponse.ok) {
      console.log("✅ Ollama is running");
    } else {
      throw new Error("Ollama not responding");
    }
  } catch (error) {
    console.log("❌ Ollama not accessible at", OLLAMA_URL);
    console.log("💡 Start Ollama with: ollama serve");
    return false;
  }

  try {
    // Check if legal thinking model exists
    const modelsResponse = await fetch(`${OLLAMA_URL}/api/tags`);
    const models = await modelsResponse.json();
    const hasThinkingModel = models.models?.some((model) =>
      model.name.includes("legal-gemma3-thinking"),
    );

    if (hasThinkingModel) {
      console.log("✅ Legal thinking model available");
    } else {
      console.log("⚠️  Legal thinking model not found");
      console.log("💡 Run setup with: npm run thinking:setup");
      return false;
    }
  } catch (error) {
    console.log("❌ Error checking models:", error.message);
    return false;
  }

  return true;
}

async function runAnalysisTest(testCase) {
  console.log(`\n🧪 Testing: ${testCase.name}`);
  console.log("─".repeat(50));

  const startTime = Date.now();

  try {
    const response = await fetch(`${BASE_URL}/api/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(testCase.payload),
    });

    const processingTime = Date.now() - startTime;

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const result = await response.json();

    // Validate response structure
    if (!result.success) {
      throw new Error(`Analysis failed: ${result.error}`);
    }

    console.log("✅ Test passed");
    console.log(`⏱️  Processing time: ${processingTime}ms`);
    console.log(`🤖 Model used: ${result.metadata.model_used}`);
    console.log(
      `🎯 Confidence: ${Math.round(result.metadata.confidence * 100)}%`,
    );
    console.log(`🧠 Thinking enabled: ${result.metadata.thinking_enabled}`);

    // Show thinking process if enabled
    if (result.analysis.thinking) {
      console.log("\n🧠 Thinking Process:");
      console.log(result.analysis.thinking.substring(0, 200) + "...");
    }

    // Show analysis result
    if (result.analysis.analysis) {
      console.log("\n📋 Analysis Result:");
      if (result.analysis.analysis.key_findings) {
        console.log(
          "Key Findings:",
          result.analysis.analysis.key_findings.slice(0, 2),
        );
      }
      if (result.analysis.analysis.recommendations) {
        console.log(
          "Recommendations:",
          result.analysis.analysis.recommendations.slice(0, 2),
        );
      }
    }

    return {
      success: true,
      processingTime,
      confidence: result.metadata.confidence,
      thinkingEnabled: result.metadata.thinking_enabled,
    };
  } catch (error) {
    console.log("❌ Test failed:", error.message);
    return {
      success: false,
      error: error.message,
      processingTime: Date.now() - startTime,
    };
  }
}

async function benchmarkPerformance() {
  console.log("\n📊 Performance Benchmark");
  console.log("========================");

  const quickTest = {
    text: "Simple evidence classification test",
    analysisType: "classification",
    useThinkingStyle: false,
  };

  const thinkingTest = {
    text: "Complex legal reasoning analysis requiring detailed thought process",
    analysisType: "reasoning",
    useThinkingStyle: true,
  };

  console.log("\n⚡ Quick Mode Performance:");
  const quickResults = [];
  for (let i = 0; i < 3; i++) {
    const result = await runAnalysisTest({
      name: `Quick Test ${i + 1}`,
      payload: quickTest,
    });
    if (result.success) {
      quickResults.push(result.processingTime);
    }
  }

  console.log("\n🧠 Thinking Mode Performance:");
  const thinkingResults = [];
  for (let i = 0; i < 3; i++) {
    const result = await runAnalysisTest({
      name: `Thinking Test ${i + 1}`,
      payload: thinkingTest,
    });
    if (result.success) {
      thinkingResults.push(result.processingTime);
    }
  }

  if (quickResults.length > 0 && thinkingResults.length > 0) {
    const quickAvg =
      quickResults.reduce((a, b) => a + b, 0) / quickResults.length;
    const thinkingAvg =
      thinkingResults.reduce((a, b) => a + b, 0) / thinkingResults.length;

    console.log("\n📈 Performance Summary:");
    console.log(`⚡ Quick Mode Average: ${Math.round(quickAvg)}ms`);
    console.log(`🧠 Thinking Mode Average: ${Math.round(thinkingAvg)}ms`);
    console.log(
      `📊 Thinking Mode Overhead: ${Math.round((thinkingAvg / quickAvg - 1) * 100)}%`,
    );
  }
}

async function generateTestReport(results) {
  console.log("\n📝 Generating test report...");

  const report = `# 🧪 Legal AI Analysis Test Report

**Generated:** ${new Date().toISOString()}

## Test Results Summary

| Test Case | Status | Processing Time | Confidence | Thinking |
|-----------|--------|----------------|------------|----------|
${results
  .map(
    (result) =>
      `| ${result.name} | ${result.success ? "✅ Pass" : "❌ Fail"} | ${result.processingTime}ms | ${result.confidence ? Math.round(result.confidence * 100) + "%" : "N/A"} | ${result.thinkingEnabled ? "🧠" : "⚡"} |`,
  )
  .join("\n")}

## System Status

- **SvelteKit App**: Running at ${BASE_URL}
- **Ollama Service**: Running at ${OLLAMA_URL}
- **Legal Thinking Model**: Available
- **Analysis API**: /api/analyze

## Test Coverage

- ✅ Quick evidence classification
- ✅ Thinking style reasoning
- ✅ Chain of custody verification
- ✅ Legal compliance checking
- ✅ Complex case analysis

## Performance Metrics

- **Quick Mode**: Optimized for speed
- **Thinking Mode**: Enhanced reasoning with step-by-step analysis
- **API Response Time**: ${results
    .filter((r) => r.success)
    .reduce((avg, r, _, arr) => avg + r.processingTime / arr.length, 0)
    .toFixed(0)}ms average

## Recommendations

${
  results.every((r) => r.success)
    ? "🎉 All tests passed! Your legal AI system is working correctly."
    : "⚠️ Some tests failed. Check the logs above for details."
}

---
*Report generated by thinking:test script*
`;

  await fs.writeFile("test-report.md", report);
  console.log("✅ Test report saved: test-report.md");
}

async function main() {
  try {
    // Check system health
    const systemHealthy = await checkSystemHealth();
    if (!systemHealthy) {
      console.log("\n🛑 System health check failed");
      console.log("💡 Ensure all services are running before testing");
      process.exit(1);
    }

    console.log("\n🚀 Running analysis tests...");

    // Run all test cases
    const results = [];
    for (const testCase of testCases) {
      const result = await runAnalysisTest(testCase);
      results.push({ ...result, name: testCase.name });
    }

    // Run performance benchmark
    await benchmarkPerformance();

    // Generate test report
    await generateTestReport(results);

    // Final summary
    const passedTests = results.filter((r) => r.success).length;
    const totalTests = results.length;

    console.log("\n🏁 Test Summary");
    console.log("===============");
    console.log(`✅ Passed: ${passedTests}/${totalTests}`);
    console.log(`❌ Failed: ${totalTests - passedTests}/${totalTests}`);

    if (passedTests === totalTests) {
      console.log(
        "\n🎉 All tests passed! Your enhanced legal AI is working perfectly.",
      );
    } else {
      console.log("\n⚠️  Some tests failed. Check the details above.");
      process.exit(1);
    }
  } catch (error) {
    console.error("\n💥 Test execution failed:", error.message);
    process.exit(1);
  }
}

// Run tests
main();
