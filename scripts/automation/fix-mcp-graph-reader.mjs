#!/usr/bin/env node
/**
 * Fix MCP Graph Reader Schema Mismatches
 * Updates mcp-graph-reader.ts to match actual database schema
 */

import fs from "fs/promises";
import path from "path";

const MCP_GRAPH_READER_FILE =
  "sveltekit-frontend/src/lib/utils/mcp-graph-reader.ts";

async function fixMCPGraphReader() {
  console.log("🔧 Fixing MCP Graph Reader schema mismatches...");

  try {
    const filePath = path.resolve(MCP_GRAPH_READER_FILE);
    let content = await fs.readFile(filePath, "utf8");

    // Fix the core schema mismatches
    const fixes = [
      // Fix missing userId references - use createdBy instead
      {
        from: /cases\.userId/g,
        to: "cases.createdBy",
      },
      {
        from: /evidence\.userId/g,
        to: "evidence.uploadedBy",
      },
      {
        from: /reports\.userId/g,
        to: "reports.createdBy",
      },

      // Fix aiSessions references to sessions
      {
        from: /aiSessions/g,
        to: "sessions",
      },

      // Fix missing caseType field - use category instead
      {
        from: /item\.case\.caseType/g,
        to: "item.case.category",
      },

      // Fix missing filePath field - use fileUrl instead
      {
        from: /item\.evidence\.filePath/g,
        to: "item.evidence.fileUrl",
      },

      // Fix missing createdAt in evidence - use uploadedAt instead
      {
        from: /item\.evidence\.createdAt/g,
        to: "item.evidence.uploadedAt",
      },

      // Fix missing aiAnalysis in reports - use metadata instead
      {
        from: /item\.report\.aiAnalysis/g,
        to: "item.report.metadata",
      },

      // Fix query builder issues - add proper where method chaining
      {
        from: /caseQuery = caseQuery\.where/g,
        to: "caseQuery = db.select({\n      case: cases,\n      user: users\n    })\n    .from(cases)\n    .leftJoin(users, eq(cases.createdBy, users.id))\n    .where",
      },

      {
        from: /evidenceQuery = evidenceQuery\.where/g,
        to: "evidenceQuery = db.select({\n      evidence: evidence,\n      user: users\n    })\n    .from(evidence)\n    .leftJoin(users, eq(evidence.uploadedBy, users.id))\n    .where",
      },

      {
        from: /reportQuery = reportQuery\.where/g,
        to: "reportQuery = db.select({\n      report: reports,\n      user: users\n    })\n    .from(reports)\n    .leftJoin(users, eq(reports.createdBy, users.id))\n    .where",
      },
    ];

    // Apply fixes
    for (const fix of fixes) {
      const before = content;
      content = content.replace(fix.from, fix.to);
      if (content !== before) {
        console.log(`✅ Applied fix: ${fix.from.toString()}`);
      }
    }

    // Write the fixed content back
    await fs.writeFile(filePath, content);
    console.log("✅ Fixed MCP Graph Reader schema mismatches");

    return true;
  } catch (error) {
    console.error("❌ Error fixing MCP Graph Reader:", error);
    return false;
  }
}

async function main() {
  console.log("🚀 Starting MCP Graph Reader fixes...\n");

  const success = await fixMCPGraphReader();

  if (success) {
    console.log("\n✅ MCP Graph Reader fixes completed!");
    console.log("\n📋 Next steps:");
    console.log("1. Review the fixed file for any remaining issues");
    console.log("2. Run TypeScript check: npm run check");
    console.log("3. Test MCP functionality");
  } else {
    console.log("\n❌ MCP Graph Reader fixes failed");
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}
