#!/usr/bin/env node

/**
 * GPU-Accelerated JSON Parser Validation Script
 * Tests the complete system integration
 */

import fs from "fs";
import path from "path";
import { execSync } from "child_process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log("🔍 Validating GPU-Accelerated JSON Parser setup...\n");

const requiredFiles = [
  "src/lib/wasm/rapid-json-parser.cpp",
  "src/lib/wasm/gpu-json-parser.ts",
  "src/lib/wasm/benchmark-json-parser.ts",
  "src/lib/wasm/build-wasm.ps1",
  "src/lib/wasm/build-wasm.sh",
  "src/lib/wasm/Makefile",
  "src/lib/vscode/json-processor-extension.ts",
  "setup-gpu-json-parser.ts",
  "setup-gpu-json-parser.ps1",
  "GPU_JSON_PARSER_README.md",
];

let allFilesPresent = true;

console.log("📁 Checking required files...");
for (const file of requiredFiles) {
  const filePath = path.join(process.cwd(), file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} (MISSING)`);
    allFilesPresent = false;
  }
}

console.log(
  "\n📋 File validation:",
  allFilesPresent ? "✅ PASSED" : "❌ FAILED"
);

// Check package.json scripts
console.log("\n🔧 Checking package.json scripts...");
try {
  const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
  const requiredScripts = [
    "setup:gpu-json",
    "build:wasm",
    "test:wasm",
    "benchmark:json",
  ];

  let scriptsValid = true;
  for (const script of requiredScripts) {
    if (packageJson.scripts && packageJson.scripts[script]) {
      console.log(`✅ ${script}: ${packageJson.scripts[script]}`);
    } else {
      console.log(`❌ ${script} (MISSING)`);
      scriptsValid = false;
    }
  }

  console.log(
    "\n📋 Scripts validation:",
    scriptsValid ? "✅ PASSED" : "❌ FAILED"
  );
} catch (error) {
  console.log("❌ Failed to read package.json:", error.message);
}

// Check system requirements
console.log("\n🛠️  Checking system requirements...");

const requirements = [
  { name: "Node.js", command: "node --version", required: true },
  { name: "npm", command: "npm --version", required: true },
  { name: "Git", command: "git --version", required: true },
  {
    name: "PowerShell",
    command: 'powershell -Command "Get-Host"',
    required: false,
  },
  { name: "Docker", command: "docker --version", required: false },
];

let requirementsMet = true;

for (const req of requirements) {
  try {
    const result = execSync(req.command, { encoding: "utf8", stdio: "pipe" });
    console.log(`✅ ${req.name}: Available`);
  } catch (error) {
    if (req.required) {
      console.log(`❌ ${req.name}: Not found (REQUIRED)`);
      requirementsMet = false;
    } else {
      console.log(`⚠️  ${req.name}: Not found (OPTIONAL)`);
    }
  }
}

console.log(
  "\n📋 Requirements validation:",
  requirementsMet ? "✅ PASSED" : "❌ FAILED"
);

// Summary
console.log("\n" + "=".repeat(60));
console.log("📊 VALIDATION SUMMARY");
console.log("=".repeat(60));

if (allFilesPresent && requirementsMet) {
  console.log("🎉 All validations PASSED!");
  console.log("\n✨ Ready to build GPU-accelerated JSON parser!");
  console.log("\n🚀 Next steps:");
  console.log("1. Run: npm run setup:gpu-json");
  console.log("2. Build: npm run build:wasm");
  console.log("3. Test: npm run test:wasm");
  console.log("4. Benchmark: npm run benchmark:json");
  console.log("\n📚 Documentation: GPU_JSON_PARSER_README.md");
} else {
  console.log("❌ Some validations FAILED!");
  console.log("\n🔧 Please fix the issues above and run validation again.");
}

console.log("\n" + "=".repeat(60));

// Exit with appropriate code
process.exit(allFilesPresent && requirementsMet ? 0 : 1);
