#!/usr/bin/env node
/**
 * Enhanced RAG Document Testing Script
 * Tests document upload and query functionality
 */

import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const BASE_URL = "http://localhost:5173";
const UPLOADS_DIR = join(__dirname, "uploads", "documents");

console.log("🧪 Enhanced RAG Document Testing\n");

async function testDocumentUpload(filePath, fileName) {
  try {
    console.log(`📤 Testing upload: ${fileName}`);

    // For now, we'll simulate the upload since the actual endpoint needs multipart form data
    const content = readFileSync(filePath, "utf8");
    const preview = content.substring(0, 200) + "...";

    console.log(`   ✅ File read successfully (${content.length} chars)`);
    console.log(`   📄 Preview: ${preview}`);

    // Test via API endpoint (simplified for demo)
    const response = await fetch(`${BASE_URL}/api/rag?action=status`);
    if (response.ok) {
      console.log(`   ✅ RAG API is accessible`);
    } else {
      console.log(`   ⚠️ RAG API returned: ${response.status}`);
    }

    return true;
  } catch (error) {
    console.log(`   ❌ Error: ${error.message}`);
    return false;
  }
}

async function testQuery(query) {
  try {
    console.log(`🔍 Testing query: "${query}"`);

    // Test the query endpoint
    const response = await fetch(`${BASE_URL}/api/rag`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        action: "search",
        query: query,
        type: "semantic",
      }),
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`   ✅ Query successful`);
      console.log(`   📊 Result: ${JSON.stringify(result, null, 2)}`);
    } else {
      console.log(`   ⚠️ Query failed: ${response.status}`);
    }

    return true;
  } catch (error) {
    console.log(`   ❌ Query error: ${error.message}`);
    return false;
  }
}

async function runTests() {
  console.log("📁 Available test documents:");

  try {
    const files = readdirSync(UPLOADS_DIR);
    const testFiles = files.filter(
      (f) => f.endsWith(".md") || f.endsWith(".txt")
    );

    console.log(`   Found ${testFiles.length} test documents\n`);

    // Test document processing
    for (const file of testFiles) {
      const filePath = join(UPLOADS_DIR, file);
      await testDocumentUpload(filePath, file);
      console.log("");
    }

    // Test queries
    const testQueries = [
      "What are the main legal requirements?",
      "Explain the technical architecture",
      "What are the AI ethics principles?",
      "How does the vector search work?",
      "What are the compliance requirements?",
    ];

    console.log("🔍 Testing Enhanced RAG Queries:\n");

    for (const query of testQueries) {
      await testQuery(query);
      console.log("");
    }
  } catch (error) {
    console.log(`❌ Test setup error: ${error.message}`);
  }
}

// Web interface testing instructions
function showWebInterfaceInstructions() {
  console.log("🌐 Web Interface Testing:\n");
  console.log("1. Open: http://localhost:5173/rag-studio");
  console.log("2. Upload documents from: uploads/documents/");
  console.log("3. Test queries like:");
  console.log('   - "What are the legal frameworks?"');
  console.log('   - "Explain the system architecture"');
  console.log('   - "What are the AI ethics principles?"');
  console.log("");
  console.log("📱 VS Code Testing:");
  console.log("1. Press: Ctrl+Shift+P");
  console.log('2. Type: "Context7 MCP: Enhanced RAG Query"');
  console.log('3. Ask: "Summarize the uploaded legal documents"');
  console.log("");
}

// Main execution
async function main() {
  showWebInterfaceInstructions();
  await runTests();

  console.log("✨ Enhanced RAG Testing Complete!");
  console.log("");
  console.log("🎯 Next Steps:");
  console.log("1. Visit http://localhost:5173/rag-studio to upload documents");
  console.log("2. Use VS Code commands for enhanced queries");
  console.log("3. Test multi-agent workflows");
  console.log("4. Monitor performance metrics");
}

main().catch(console.error);
