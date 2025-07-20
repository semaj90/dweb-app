#!/usr/bin/env node

import { execSync } from "child_process";

console.log("🚀 Setting up PostgreSQL for Legal AI Assistant...");

try {
  // 1. Start Docker services
  console.log("\n1️⃣ Starting Docker services...");
  execSync("docker-compose up -d postgres redis qdrant", {
    stdio: "inherit",
    cwd: process.cwd(),
  });

  // Wait a moment for services to start
  console.log("\n⏳ Waiting for services to start...");
  await new Promise((resolve) => setTimeout(resolve, 5000));

  // 2. Run database migrations
  console.log("\n2️⃣ Creating database schema...");
  execSync("npx drizzle-kit push", {
    stdio: "inherit",
    cwd: process.cwd(),
  });

  // 3. Initialize database with demo data
  console.log("\n3️⃣ Initializing database...");
  execSync("node init-postgres.js", {
    stdio: "inherit",
    cwd: process.cwd(),
  });

  console.log("\n✅ PostgreSQL setup complete!");
  console.log(
    "\n🎉 You can now start the development server with: npm run dev",
  );
  console.log("\n👤 Demo login credentials:");
  console.log("   Email: admin@prosecutor.com");
  console.log("   Password: password");
} catch (error) {
  console.error("\n❌ Setup failed:", error.message);
  console.log("\n💡 Try running the steps manually:");
  console.log("   1. docker-compose up -d postgres");
  console.log("   2. npx drizzle-kit push");
  console.log("   3. node init-postgres.js");
}

// Async wrapper to use await
(async () => {})();
