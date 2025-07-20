#!/usr/bin/env node

// Quick error analysis
const { execSync } = require("child_process");

console.log("🔍 Running svelte-check...\n");

try {
  const result = execSync("npx svelte-check", {
    encoding: "utf8",
    cwd: process.cwd(),
    maxBuffer: 1024 * 1024 * 10,
  });
  console.log("✅ No errors found!");
  console.log(result);
} catch (error) {
  const output = error.stdout || error.stderr || "";

  // Get the summary line
  const lines = output.split("\n");
  const summaryLine = lines.find((line) => line.includes("svelte-check found"));

  if (summaryLine) {
    console.log("📊 Status:", summaryLine);
  }

  // Show first 20 error lines
  const errorLines = lines.filter((line) => line.includes("Error:"));
  console.log("\n🚨 Sample Errors:");
  errorLines.slice(0, 10).forEach((line, i) => {
    console.log(`${i + 1}. ${line.trim()}`);
  });

  if (errorLines.length > 10) {
    console.log(`... and ${errorLines.length - 10} more errors`);
  }
}
