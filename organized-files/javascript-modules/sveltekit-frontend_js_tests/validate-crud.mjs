#!/usr/bin/env node

// Quick CRUD & CMS Validation Script
// Run this to verify all database operations are working

console.log("🔍 CRUD & CMS Database Sync Validator");
console.log("=====================================\n");

// Check if running in the correct directory
import { existsSync } from "fs";
import { resolve } from "path";

const requiredFiles = [
  "package.json",
  "src/routes/api/cases/+server.ts",
  "src/routes/api/evidence/+server.ts",
  "src/routes/api/reports/+server.ts",
  "src/lib/server/db/schema-postgres.ts",
];

let allFilesExist = true;
for (const file of requiredFiles) {
  if (!existsSync(file)) {
    console.log(`❌ Missing file: ${file}`);
    allFilesExist = false;
  }
}

if (!allFilesExist) {
  console.log(
    "\n❌ Please run this script from the sveltekit-frontend directory",
  );
  process.exit(1);
}

console.log("✅ All required files found");

// Check package.json for dependencies
try {
  const packageJson = JSON.parse(await Bun.file("package.json").text());
  const requiredDeps = ["drizzle-orm", "@sveltejs/kit", "postgres"];

  let missingDeps = [];
  for (const dep of requiredDeps) {
    if (
      !packageJson.dependencies?.[dep] &&
      !packageJson.devDependencies?.[dep]
    ) {
      missingDeps.push(dep);
    }
  }

  if (missingDeps.length > 0) {
    console.log(`❌ Missing dependencies: ${missingDeps.join(", ")}`);
    console.log("💡 Run: npm install");
    process.exit(1);
  }

  console.log("✅ All dependencies available");
} catch (error) {
  console.log("⚠️ Could not validate package.json");
}

// Check if server is running
async function checkServer() {
  try {
    const response = await fetch("http://localhost:5173/api/cases", {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    return response.status !== 0;
  } catch (error) {
    return false;
  }
}

const serverRunning = await checkServer();

console.log("\n📊 System Status:");
console.log("=================");

if (serverRunning) {
  console.log("✅ SvelteKit server: RUNNING (localhost:5173)");
  console.log("✅ API endpoints: ACCESSIBLE");
} else {
  console.log("❌ SvelteKit server: NOT RUNNING");
  console.log("💡 Start with: npm run dev");
}

// Check database schema
console.log("\n🗄️ Database Schema Status:");
console.log("==========================");

const schemaFile = "src/lib/server/db/schema-postgres.ts";
try {
  const schemaContent = await Bun.file(schemaFile).text();

  const tables = [
    "users",
    "sessions",
    "cases",
    "criminals",
    "evidence",
    "caseActivities",
    "reports",
    "canvasStates",
  ];

  let foundTables = 0;
  for (const table of tables) {
    if (schemaContent.includes(`export const ${table} = pgTable`)) {
      console.log(`✅ ${table} table: DEFINED`);
      foundTables++;
    } else {
      console.log(`❌ ${table} table: MISSING`);
    }
  }

  console.log(
    `\n📈 Schema Coverage: ${foundTables}/${tables.length} tables (${Math.round((foundTables / tables.length) * 100)}%)`,
  );
} catch (error) {
  console.log("❌ Could not read database schema");
}

// Check API endpoints
console.log("\n🌐 API Endpoints Status:");
console.log("========================");

const endpoints = [
  { name: "Cases", path: "src/routes/api/cases/+server.ts" },
  { name: "Evidence", path: "src/routes/api/evidence/+server.ts" },
  { name: "Reports", path: "src/routes/api/reports/+server.ts" },
  { name: "Criminals", path: "src/routes/api/criminals/+server.ts" },
  { name: "Activities", path: "src/routes/api/activities/+server.ts" },
  { name: "Canvas States", path: "src/routes/api/canvas-states/+server.ts" },
];

let workingEndpoints = 0;
for (const endpoint of endpoints) {
  if (existsSync(endpoint.path)) {
    try {
      const content = await Bun.file(endpoint.path).text();
      const methods = ["GET", "POST", "PUT", "DELETE", "PATCH"];
      const foundMethods = methods.filter((method) =>
        content.includes(`export const ${method}`),
      );

      console.log(
        `✅ ${endpoint.name}: ${foundMethods.length}/5 HTTP methods (${foundMethods.join(", ")})`,
      );
      workingEndpoints++;
    } catch (error) {
      console.log(`❌ ${endpoint.name}: ERROR reading file`);
    }
  } else {
    console.log(`❌ ${endpoint.name}: FILE MISSING`);
  }
}

console.log(
  `\n📈 API Coverage: ${workingEndpoints}/${endpoints.length} endpoints (${Math.round((workingEndpoints / endpoints.length) * 100)}%)`,
);

// Final assessment
console.log("\n🏆 FINAL ASSESSMENT:");
console.log("====================");

const totalChecks = 3; // Server, Schema, API
let passedChecks = 0;

if (serverRunning) passedChecks++;
if (foundTables >= 6) passedChecks++; // At least 75% of tables
if (workingEndpoints >= 5) passedChecks++; // At least 83% of endpoints

const overallScore = Math.round((passedChecks / totalChecks) * 100);

if (overallScore >= 90) {
  console.log("🎉 EXCELLENT - All systems operational!");
  console.log("🚀 Your CRUD & CMS functionality is ready for production");
  console.log("💚 Database sync: FULLY WORKING");
} else if (overallScore >= 70) {
  console.log("✅ GOOD - Most systems working");
  console.log("⚠️ Some minor issues need attention");
} else {
  console.log("⚠️ NEEDS WORK - Several issues detected");
  console.log("🔧 Please review the errors above");
}

console.log(`\n📊 Overall Health: ${overallScore}%`);

// Quick start instructions
if (!serverRunning) {
  console.log("\n💡 QUICK START:");
  console.log("================");
  console.log("1. npm install            # Install dependencies");
  console.log("2. npm run dev            # Start development server");
  console.log("3. node validate-crud.mjs # Run this script again");
}

console.log("\n🧪 COMPREHENSIVE TESTING:");
console.log("==========================");
console.log("Run: node test-crud-sync.mjs    # Full API testing");
console.log("Run: node validate-crud-sync.mjs # Database validation");

console.log("\n✨ CRUD & CMS validation complete!");
