#!/usr/bin/env node
/**
 * Dev Automation Script for SvelteKit + MCP Legal AI Project
 * - Runs `npm run check` for TypeScript/Svelte errors
 * - Starts the dev server automatically
 * - Optionally runs MCP best practices analysis on open workspace
 *
 * Usage: node dev-auto.js [--best-practices]
 */

const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");

const runCheck = () => {
  return new Promise((resolve, reject) => {
    const check = exec("npm run check", { cwd: process.cwd() });
    check.stdout.on("data", (data) => process.stdout.write(data));
    check.stderr.on("data", (data) => process.stderr.write(data));
    check.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error("npm run check failed"));
    });
  });
};

const startDevServer = () => {
  const dev = exec("npm run dev", { cwd: process.cwd() });
  dev.stdout.on("data", (data) => process.stdout.write(data));
  dev.stderr.on("data", (data) => process.stderr.write(data));
  // Do not await, keep process running
};

const runBestPractices = async () => {
  try {
    // Dynamically import MCP helpers if available
    const helpersPath = path.resolve(
      process.cwd(),
      "src/lib/utils/mcp-helpers.js"
    );
    if (fs.existsSync(helpersPath)) {
      const mcp = require(helpersPath);
      if (mcp && mcp.commonMCPQueries && mcp.mcpSuggestBestPractices) {
        const bestPracticesReq =
          mcp.commonMCPQueries.performanceBestPractices();
        const prompt = mcp.generateMCPPrompt(bestPracticesReq);
        console.log("\n[Best Practices Prompt]:", prompt);
        const bestPractices = await mcp.mcpSuggestBestPractices({});
        console.log("\n[Best Practices]:", bestPractices);
      } else {
        console.log("MCP helpers do not export required functions.");
      }
    } else {
      console.log("MCP helpers not found. Skipping best practices analysis.");
    }
  } catch (err) {
    console.error("Error running best practices:", err);
  }
};

(async () => {
  const args = process.argv.slice(2);
  const doBestPractices = args.includes("--best-practices");
  try {
    console.log("Running TypeScript/Svelte check...");
    await runCheck();
    if (doBestPractices) {
      await runBestPractices();
    }
    console.log("Starting dev server...");
    startDevServer();
  } catch (err) {
    console.error("Dev automation failed:", err);
    process.exit(1);
  }
})();
