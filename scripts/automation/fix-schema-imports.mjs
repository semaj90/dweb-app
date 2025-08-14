#!/usr/bin/env node
/**
 * Fix Schema Import Errors
 * Adds missing type exports to schema-postgres.ts and fixes import mismatches
 */

import fs from "fs/promises";
import path from "path";

const SCHEMA_FILE = "sveltekit-frontend/src/lib/server/db/schema-postgres.ts";
const CASE_STORE_FILE = "sveltekit-frontend/src/lib/stores/caseStore.ts";

async function fixSchemaImports() {
  console.log("🔧 Fixing schema import errors...");

  try {
    // Read schema file
    const schemaPath = path.resolve(SCHEMA_FILE);
    let schemaContent = await fs.readFile(schemaPath, "utf8");

    // Check if type exports are already present
    if (!schemaContent.includes("export type Case =")) {
      console.log("📝 Adding missing type exports to schema...");

      // Add type exports at the end of the file
      const typeExports = `
// === MISSING TYPE EXPORTS ===

export type Case = typeof cases.$inferSelect;
export type NewCase = typeof cases.$inferInsert;

export type Evidence = typeof evidence.$inferSelect;
export type NewEvidence = typeof evidence.$inferInsert;

export type Report = typeof reports.$inferSelect;
export type NewReport = typeof reports.$inferInsert;

export type Criminal = typeof criminals.$inferSelect;
export type NewCriminal = typeof criminals.$inferInsert;

export type CaseActivity = typeof caseActivities.$inferSelect;
export type NewCaseActivity = typeof caseActivities.$inferInsert;

export type PersonOfInterest = typeof personsOfInterest.$inferSelect;
export type NewPersonOfInterest = typeof personsOfInterest.$inferInsert;

export type CanvasState = typeof canvasStates.$inferSelect;
export type NewCanvasState = typeof canvasStates.$inferInsert;

export type LegalDocument = typeof legalDocuments.$inferSelect;
export type NewLegalDocument = typeof legalDocuments.$inferInsert;

export type RagSession = typeof ragSessions.$inferSelect;
export type NewRagSession = typeof ragSessions.$inferInsert;

export type RagMessage = typeof ragMessages.$inferSelect;
export type NewRagMessage = typeof ragMessages.$inferInsert;

export type Theme = typeof themes.$inferSelect;
export type NewTheme = typeof themes.$inferInsert;

export type ContentEmbedding = typeof contentEmbeddings.$inferSelect;
export type NewContentEmbedding = typeof contentEmbeddings.$inferInsert;

export type AttachmentVerification = typeof attachmentVerifications.$inferSelect;
export type NewAttachmentVerification = typeof attachmentVerifications.$inferInsert;

export type EmbeddingCache = typeof embeddingCache.$inferSelect;
export type NewEmbeddingCache = typeof embeddingCache.$inferInsert;

export type VectorMetadata = typeof vectorMetadata.$inferSelect;
export type NewVectorMetadata = typeof vectorMetadata.$inferInsert;
`;

      // Insert before the last existing type exports (User types)
      const insertPosition = schemaContent.lastIndexOf(
        "// === TYPE EXPORTS ==="
      );
      if (insertPosition !== -1) {
        schemaContent =
          schemaContent.slice(0, insertPosition) +
          typeExports +
          "\n" +
          schemaContent.slice(insertPosition);
      } else {
        // If no existing type exports section, add at the end
        schemaContent += typeExports;
      }

      await fs.writeFile(schemaPath, schemaContent);
      console.log("✅ Added missing type exports to schema-postgres.ts");
    } else {
      console.log("✅ Type exports already present in schema");
    }

    // Check and fix caseStore imports
    console.log("🔍 Checking caseStore imports...");
    const caseStorePath = path.resolve(CASE_STORE_FILE);
    let caseStoreContent = await fs.readFile(caseStorePath, "utf8");

    // Check if import is correct
    if (
      caseStoreContent.includes(
        "import type { Case, Evidence, Report } from '$lib/server/db/schema-postgres';"
      )
    ) {
      console.log("✅ CaseStore imports are correct");
    } else {
      console.log("⚠️ CaseStore imports may need manual review");
    }

    console.log("🎉 Schema import fixes completed!");
    return true;
  } catch (error) {
    console.error("❌ Error fixing schema imports:", error);
    return false;
  }
}

async function main() {
  console.log("🚀 Starting schema import fix...\n");

  const success = await fixSchemaImports();

  if (success) {
    console.log("\n✅ Schema import fix completed successfully!");
    console.log("\n📋 Next steps:");
    console.log("1. Run TypeScript check: npm run check");
    console.log("2. Test build: npm run build");
    console.log("3. Check for remaining errors");
  } else {
    console.log("\n❌ Schema import fix failed");
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}
