#!/usr/bin/env node

/**
 * Fix Database Import Paths
 * This script fixes all incorrect database import paths in API routes
 */

import { readFileSync, writeFileSync } from "fs";
import { glob } from "glob";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log("🔧 Fixing Database Import Paths");
console.log("=================================\n");

// Find all TypeScript files in routes/api
const pattern = join(__dirname, "src/routes/api/**/*.ts");
const files = glob.sync(pattern);

console.log(`Found ${files.length} API files to check...\n`);

let fixedCount = 0;

for (const file of files) {
  try {
    const content = readFileSync(file, "utf-8");
    let modified = false;
    let newContent = content;

    // Fix db.js imports to db/index.js
    const dbImportRegex = /from ['"](\.\.\/)+lib\/server\/db\.js['"]/g;
    if (dbImportRegex.test(content)) {
      newContent = newContent.replace(dbImportRegex, (match) => {
        const pathPart = match.match(/from ['"](.+)\.js['"]/)[1];
        return `from '${pathPart}/index.js'`;
      });
      modified = true;
    }

    // Fix other problematic database imports
    const problemImports = [
      // Fix connection.js to index.js
      {
        from: /lib\/server\/database\/connection\.js/g,
        to: "lib/server/db/index.js",
      },
      // Fix pg.js to index.js
      {
        from: /lib\/server\/db\/pg\.js/g,
        to: "lib/server/db/index.js",
      },
      // Fix schema-postgres.js to schema.js
      {
        from: /lib\/server\/db\/schema-postgres\.js/g,
        to: "lib/server/db/schema.js",
      },
      // Fix schema-canvas.js to schema.js (if it doesn't exist)
      {
        from: /lib\/server\/db\/schema-canvas\.js/g,
        to: "lib/server/db/schema.js",
      },
    ];

    for (const fix of problemImports) {
      if (fix.from.test(newContent)) {
        newContent = newContent.replace(fix.from, fix.to);
        modified = true;
      }
    }

    if (modified) {
      writeFileSync(file, newContent);
      console.log(`✅ Fixed: ${file.replace(__dirname, ".")}`);
      fixedCount++;
    }
  } catch (error) {
    console.log(`⚠️  Error processing ${file}: ${error.message}`);
  }
}

console.log(`\n🎉 Fixed ${fixedCount} files with incorrect database imports`);

if (fixedCount > 0) {
  console.log("\n📋 Changes made:");
  console.log("  • lib/server/db.js → lib/server/db/index.js");
  console.log("  • lib/server/database/connection.js → lib/server/db/index.js");
  console.log("  • lib/server/db/pg.js → lib/server/db/index.js");
  console.log("  • lib/server/db/schema-postgres.js → lib/server/db/schema.js");
  console.log("  • lib/server/db/schema-canvas.js → lib/server/db/schema.js");

  console.log("\n✅ All database import paths are now fixed!");
  console.log("   You can now run: npm run dev");
} else {
  console.log("\n✅ All database import paths were already correct!");
}
