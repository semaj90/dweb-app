#!/usr/bin/env node

/**
 * Comprehensive RAG System Test Runner
 * Tests Ollama, PostgreSQL, pgvector, Drizzle ORM, CUDA, and SvelteKit integration
 */

import { spawn } from "child_process";
import fs from "fs";
import path from "path";

const colors = {
  reset: "\x1b[0m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function header(message) {
  const line = "=".repeat(60);
  log(`\n${line}`, colors.cyan);
  log(`🚀 ${message}`, colors.cyan);
  log(line, colors.cyan);
}

function success(message) {
  log(`✅ ${message}`, colors.green);
}

function error(message) {
  log(`❌ ${message}`, colors.red);
}

function warning(message) {
  log(`⚠️  ${message}`, colors.yellow);
}

function info(message) {
  log(`ℹ️  ${message}`, colors.blue);
}

// Test configurations
const testSuites = [
  {
    name: "Token Usage Management",
    command: "npm",
    args: ["run", "test:token-usage"],
    description: "Tests token tracking, limits, and usage optimization",
  },
  {
    name: "Comprehensive RAG System",
    command: "npm",
    args: ["run", "test:rag:comprehensive"],
    description: "Tests complete RAG pipeline with all integrations",
  },
  {
    name: "Ollama GPU Acceleration",
    command: "npm",
    args: ["run", "test:ollama"],
    description: "Tests NVIDIA CUDA and GPU performance",
  },
  {
    name: "PostgreSQL and pgvector",
    command: "npm",
    args: ["run", "test:postgresql"],
    description: "Tests database integration and vector search",
  },
  {
    name: "SvelteKit 2/Svelte 5",
    command: "npm",
    args: ["run", "test:sveltekit"],
    description: "Tests frontend framework integration",
  },
  {
    name: "Performance and CUDA",
    command: "npm",
    args: ["run", "test:performance"],
    description: "Tests system performance and GPU acceleration",
  },
];

// Pre-flight checks
async function preflightChecks() {
  header("Pre-flight System Checks");

  const checks = [
    {
      name: "Node.js",
      command: "node",
      args: ["--version"],
      required: true,
    },
    {
      name: "npm",
      command: "npm",
      args: ["--version"],
      required: true,
    },
    {
      name: "Docker",
      command: "docker",
      args: ["--version"],
      required: false,
    },
    {
      name: "Playwright",
      command: "npx",
      args: ["playwright", "--version"],
      required: true,
    },
  ];

  let allPassed = true;

  for (const check of checks) {
    try {
      await runCommand(check.command, check.args, { silent: true });
      success(`${check.name} is available`);
    } catch (error) {
      if (check.required) {
        error(`${check.name} is required but not available`);
        allPassed = false;
      } else {
        warning(`${check.name} is not available (optional)`);
      }
    }
  }

  // Check for required files
  const requiredFiles = [
    "package.json",
    "playwright.config.ts",
    "tests/comprehensive-rag-system.spec.ts",
    "tests/token-usage.spec.ts",
    "sveltekit-frontend/src/routes/ai-demo/+page.svelte",
  ];

  for (const file of requiredFiles) {
    if (fs.existsSync(file)) {
      success(`${file} exists`);
    } else {
      error(`${file} is missing`);
      allPassed = false;
    }
  }

  return allPassed;
}

// Service health checks
async function serviceHealthChecks() {
  header("Service Health Checks");

  const services = [
    {
      name: "Ollama",
      url: "http://localhost:11434/api/tags",
      required: true,
    },
    {
      name: "PostgreSQL",
      url: "http://localhost:5432",
      required: false,
    },
    {
      name: "Redis",
      url: "http://localhost:6379",
      required: false,
    },
    {
      name: "SvelteKit Dev Server",
      url: "http://localhost:5173",
      required: false,
    },
  ];

  for (const service of services) {
    try {
      const response = await fetch(service.url, {
        method: "GET",
        timeout: 5000,
      });

      if (response.ok) {
        success(`${service.name} is healthy`);
      } else {
        warning(`${service.name} responded with status ${response.status}`);
      }
    } catch (error) {
      if (service.required) {
        error(`${service.name} is not available (required)`);
      } else {
        warning(`${service.name} is not available (will start if needed)`);
      }
    }
  }
}

// Run a command and return a promise
function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      stdio: options.silent ? "pipe" : "inherit",
      shell: true,
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve(code);
      } else {
        reject(new Error(`Command failed with code ${code}`));
      }
    });

    proc.on("error", (error) => {
      reject(error);
    });
  });
}

// Run individual test suite
async function runTestSuite(suite) {
  header(`${suite.name} Tests`);
  info(suite.description);

  try {
    const startTime = Date.now();
    await runCommand(suite.command, suite.args);
    const duration = Date.now() - startTime;

    success(`${suite.name} tests completed in ${duration}ms`);
    return { name: suite.name, success: true, duration };
  } catch (error) {
    error(`${suite.name} tests failed: ${error.message}`);
    return { name: suite.name, success: false, error: error.message };
  }
}

// Generate test report
function generateReport(results, totalDuration) {
  header("Test Results Summary");

  const passed = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  log(`\n📊 Test Statistics:`, colors.cyan);
  log(`   Total Suites: ${results.length}`);
  log(`   Passed: ${passed.length}`, colors.green);
  log(
    `   Failed: ${failed.length}`,
    failed.length > 0 ? colors.red : colors.green
  );
  log(`   Total Duration: ${totalDuration}ms`);

  if (passed.length > 0) {
    log(`\n✅ Passed Tests:`, colors.green);
    passed.forEach((result) => {
      log(`   • ${result.name} (${result.duration}ms)`);
    });
  }

  if (failed.length > 0) {
    log(`\n❌ Failed Tests:`, colors.red);
    failed.forEach((result) => {
      log(`   • ${result.name}: ${result.error}`);
    });
  }

  // Write detailed report to file
  const reportPath = `test-report-${Date.now()}.json`;
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      total: results.length,
      passed: passed.length,
      failed: failed.length,
      duration: totalDuration,
    },
    results: results,
  };

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  info(`Detailed report saved to: ${reportPath}`);

  return failed.length === 0;
}

// Main test runner
async function main() {
  header("Legal AI System - Comprehensive Test Suite");
  info(
    "Testing Ollama, PostgreSQL, pgvector, Drizzle ORM, CUDA, and SvelteKit integration"
  );

  const startTime = Date.now();

  try {
    // Pre-flight checks
    const preflightPassed = await preflightChecks();
    if (!preflightPassed) {
      error(
        "Pre-flight checks failed. Please resolve issues before running tests."
      );
      process.exit(1);
    }

    // Service health checks
    await serviceHealthChecks();

    // Run test suites
    const results = [];

    for (const suite of testSuites) {
      const result = await runTestSuite(suite);
      results.push(result);

      // Brief pause between test suites
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }

    // Generate report
    const totalDuration = Date.now() - startTime;
    const allPassed = generateReport(results, totalDuration);

    if (allPassed) {
      header("🎉 All Tests Passed!");
      success("Legal AI system is fully operational and ready for production.");
      process.exit(0);
    } else {
      header("⚠️  Some Tests Failed");
      error("Please review the failed tests and resolve issues.");
      process.exit(1);
    }
  } catch (error) {
    error(`Test runner failed: ${error.message}`);
    process.exit(1);
  }
}

// Handle command line arguments
const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
  log("Legal AI System Test Runner", colors.cyan);
  log("\nUsage: node run-comprehensive-tests.mjs [options]");
  log("\nOptions:");
  log("  --help, -h     Show this help message");
  log("  --quick, -q    Run quick tests only");
  log("  --gpu, -g      Run GPU tests only");
  log("  --verbose, -v  Verbose output");
  log("\nTest Suites:");
  testSuites.forEach((suite) => {
    log(`  • ${suite.name}: ${suite.description}`);
  });
  process.exit(0);
}

if (args.includes("--quick") || args.includes("-q")) {
  // Run only essential tests
  testSuites.splice(
    0,
    testSuites.length,
    testSuites.find((s) => s.name === "Token Usage Management"),
    testSuites.find((s) => s.name === "Comprehensive RAG System")
  );
  info("Running quick tests only");
}

if (args.includes("--gpu") || args.includes("-g")) {
  // Run only GPU-related tests
  testSuites.splice(
    0,
    testSuites.length,
    testSuites.find((s) => s.name === "Ollama GPU Acceleration"),
    testSuites.find((s) => s.name === "Performance and CUDA")
  );
  info("Running GPU tests only");
}

// Start the test runner
main().catch((error) => {
  console.error(`Unhandled error: ${error.message}`);
  process.exit(1);
});
