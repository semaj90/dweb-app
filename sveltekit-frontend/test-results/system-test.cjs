// Quick test script to verify the legal document system
const fs = require("fs");
const path = require("path");

console.log("🧪 Testing Legal Document Management System");
console.log("============================================");

// Test 1: Component Loading
console.log("\n📦 Testing Component Structure...");

const components = [
  "src/lib/components/editor/LegalDocumentEditor.svelte",
  "src/lib/components/ui/CommandMenu.svelte",
  "src/lib/components/ui/GoldenLayout.svelte",
  "src/lib/components/ui/ExpandGrid.svelte",
  "src/lib/components/evidence/EvidenceCard.svelte",
];

components.forEach((component) => {
  if (fs.existsSync(component)) {
    console.log(`✅ ${path.basename(component)} - Found`);
  } else {
    console.log(`❌ ${path.basename(component)} - Missing`);
  }
});

// Test 2: API Routes
console.log("\n🛣️  Testing API Routes...");

const apiRoutes = [
  "src/routes/api/documents/+server.ts",
  "src/routes/api/documents/[id]/+server.ts",
  "src/routes/api/documents/[id]/auto-save/+server.ts",
  "src/routes/api/citations/+server.ts",
  "src/routes/api/ai/ask/+server.ts",
];

apiRoutes.forEach((route) => {
  if (fs.existsSync(route)) {
    console.log(`✅ ${path.basename(path.dirname(route))} - Implemented`);
  } else {
    console.log(`❌ ${path.basename(path.dirname(route))} - Missing`);
  }
});

// Test 3: Database Schema
console.log("\n🗃️  Testing Database Schema...");

const schemaFile = "src/lib/server/db/unified-schema.ts";
if (fs.existsSync(schemaFile)) {
  const schemaContent = fs.readFileSync(schemaFile, "utf8");
  if (schemaContent.includes("legalDocuments")) {
    console.log("✅ Legal Documents schema - Defined");
  } else {
    console.log("❌ Legal Documents schema - Missing");
  }
} else {
  console.log("❌ Database schema file - Missing");
}

// Test 4: Demo Pages
console.log("\n🎭 Testing Demo Pages...");

const demoPages = [
  "src/routes/document-editor-demo/+page.svelte",
  "src/routes/modern-demo/+page.svelte",
];

demoPages.forEach((page) => {
  if (fs.existsSync(page)) {
    console.log(`✅ ${path.basename(path.dirname(page))} - Available`);
  } else {
    console.log(`❌ ${path.basename(path.dirname(page))} - Missing`);
  }
});

console.log("\n🎉 System Test Complete!");
console.log("========================");
console.log(
  "✨ Visit http://localhost:5173/document-editor-demo to test the editor",
);
console.log("✨ Visit http://localhost:5173/modern-demo to see all components");
console.log("");
console.log("🚀 To start the development server:");
console.log("   npm run dev");
console.log("");
console.log("📊 To generate project status:");
console.log("   node generate-todo-demo.js");
