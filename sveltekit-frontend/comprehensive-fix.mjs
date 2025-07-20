#!/usr/bin/env node

import { existsSync } from "fs";
import { mkdir, readdir, readFile, writeFile } from "fs/promises";
import { dirname, extname, join } from "path";

// Comprehensive list of fixes
const FIXES = {
  // Type imports
  typeImports: {
    User: {
      check: /(?::\s*User(?:\s|\[|\||;|$)|user:\s*User)/,
      import: "import type { User } from '$lib/types/user';",
    },
    Case: {
      check: /(?::\s*Case(?:\s|\[|\||;|$)|case:\s*Case)/,
      import: "import type { Case } from '$lib/types';",
    },
    Evidence: {
      check: /(?::\s*Evidence(?:\s|\[|\||;|$)|evidence:\s*Evidence)/,
      import: "import type { Evidence } from '$lib/types';",
    },
  },

  // Property mappings
  propertyMappings: [
    { from: /user\.avatar(?!Url)/g, to: "user.avatarUrl" },
    { from: /user\.username/g, to: "user.name" },
    { from: /user\.image/g, to: "user.avatarUrl" },
  ],

  // Import conflicts
  importConflicts: [
    {
      pattern:
        /import\s*{\s*([^}]*)\bUser\b(?!\s+as)([^}]*)\s*}\s*from\s*['"]lucide-svelte['"]/g,
      fix: (match, before, after) => {
        const imports = [
          ...before
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          "User as UserIcon",
          ...after
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
        ].filter(Boolean);
        return `import { ${imports.join(", ")} } from "lucide-svelte"`;
      },
    },
  ],

  // Component usage
  componentUsage: [{ from: /<User\s+(?=size|class)/g, to: "<UserIcon " }],

  // CSS classes
  cssClasses: [
    {
      from: /class="mx-auto px-4 max-w-7xl"/g,
      to: 'class="container mx-auto px-4"',
    },
    { from: /class="btn btn-primary"/g, to: 'class="nier-button-primary"' },
    { from: /class="card"/g, to: 'class="nier-card"' },
  ],

  // Import paths
  importPaths: [
    {
      from: /from\s+['"]\.\.\/(?:\.\.\/)*lib\/([^'"]+)['"]/g,
      to: "from '$lib/$1'",
    },
    { from: /from\s+(['"])([^'"]+)\.js\1/g, to: "from $1$2$1" },
  ],
};

let totalFixed = 0;
let totalErrors = 0;

async function ensureDirectoryExists(filePath) {
  const dir = dirname(filePath);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }
}

async function fixFile(filePath) {
  try {
    let content = await readFile(filePath, "utf-8");
    let modified = false;
    const fixes = [];

    // Check and add missing type imports
    for (const [typeName, config] of Object.entries(FIXES.typeImports)) {
      if (
        config.check.test(content) &&
        !content.includes(`type { ${typeName} }`)
      ) {
        const scriptMatch = content.match(/<script[^>]*>/);
        if (scriptMatch) {
          const insertPos = scriptMatch.index + scriptMatch[0].length;
          content =
            content.slice(0, insertPos) +
            "\n  " +
            config.import +
            content.slice(insertPos);
          modified = true;
          fixes.push(`Added ${typeName} import`);
        }
      }
    }

    // Apply property mappings
    for (const mapping of FIXES.propertyMappings) {
      const before = content;
      content = content.replace(mapping.from, mapping.to);
      if (before !== content) {
        modified = true;
        fixes.push(`Fixed property: ${mapping.from.source} → ${mapping.to}`);
      }
    }

    // Fix import conflicts
    for (const conflict of FIXES.importConflicts) {
      const before = content;
      content = content.replace(conflict.pattern, conflict.fix);
      if (before !== content) {
        modified = true;
        fixes.push("Fixed lucide-svelte User import conflict");
      }
    }

    // Fix component usage
    for (const usage of FIXES.componentUsage) {
      const before = content;
      content = content.replace(usage.from, usage.to);
      if (before !== content) {
        modified = true;
        fixes.push(`Fixed component usage: ${usage.from.source}`);
      }
    }

    // Fix CSS classes
    for (const cssClass of FIXES.cssClasses) {
      const before = content;
      content = content.replace(cssClass.from, cssClass.to);
      if (before !== content) {
        modified = true;
        fixes.push(`Fixed CSS class: ${cssClass.from.source}`);
      }
    }

    // Fix import paths
    for (const path of FIXES.importPaths) {
      const before = content;
      content = content.replace(path.from, path.to);
      if (before !== content) {
        modified = true;
        fixes.push("Fixed import path");
      }
    }

    // Add lang="ts" to script tags in Svelte files
    if (filePath.endsWith(".svelte")) {
      const before = content;
      content = content.replace(
        /<script(?!.*lang=["']ts["'])>/g,
        '<script lang="ts">',
      );
      if (before !== content) {
        modified = true;
        fixes.push('Added lang="ts" to script tag');
      }
    }

    // Remove duplicate imports
    const importRegex = /^import\s+.*?from\s+['"][^'"]+['"];?\s*$/gm;
    const imports = content.match(importRegex) || [];
    const uniqueImports = [...new Set(imports)];
    if (imports.length !== uniqueImports.length) {
      const nonImportContent = content.replace(importRegex, "").trim();
      content = uniqueImports.join("\n") + "\n\n" + nonImportContent;
      modified = true;
      fixes.push("Removed duplicate imports");
    }

    if (modified) {
      await writeFile(filePath, content);
      console.log(`✓ Fixed ${filePath.replace(process.cwd(), ".")}`);
      fixes.forEach((fix) => console.log(`  - ${fix}`));
      totalFixed++;
    }

    return modified;
  } catch (error) {
    console.error(`✗ Error in ${filePath}: ${error.message}`);
    totalErrors++;
    return false;
  }
}

async function processDirectory(dir) {
  const entries = await readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = join(dir, entry.name);

    if (entry.isDirectory()) {
      if (
        ["node_modules", ".svelte-kit", "dist", "build", ".git"].includes(
          entry.name,
        )
      ) {
        continue;
      }
      await processDirectory(fullPath);
    } else if (entry.isFile()) {
      const ext = extname(entry.name);
      if (
        [".svelte", ".ts", ".js"].includes(ext) &&
        !entry.name.endsWith(".backup")
      ) {
        await fixFile(fullPath);
      }
    }
  }
}

// Create missing type export files
async function createTypeExports() {
  const typeExports = {
    "src/lib/types/index.ts": `// Re-export all types
export * from './user';
export * from './database';
export * from './api';
export * from './canvas';
export * from './global.d';
`,
    "src/lib/components/ui/index.ts": `// UI component exports
export { default as Button } from './button/Button.svelte';
export { default as Card } from './Card.svelte';
export { default as Input } from './Input.svelte';
export { default as Label } from './Label.svelte';
export { default as Modal } from './Modal.svelte';
export { default as Badge } from './Badge.svelte';
export { default as Tooltip } from './Tooltip.svelte';
`,
  };

  for (const [path, content] of Object.entries(typeExports)) {
    const fullPath = join(process.cwd(), path);
    try {
      await ensureDirectoryExists(fullPath);
      if (!existsSync(fullPath)) {
        await writeFile(fullPath, content);
        console.log(`✓ Created ${path}`);
      }
    } catch (error) {
      console.error(`✗ Could not create ${path}: ${error.message}`);
    }
  }
}

// Main execution
async function main() {
  console.log("🔧 Running comprehensive TypeScript fixes...\n");

  console.log("Creating type exports...");
  await createTypeExports();

  console.log("\nProcessing files...\n");
  await processDirectory(join(process.cwd(), "src"));

  console.log("\n" + "=".repeat(50));
  console.log("✨ SUMMARY");
  console.log("=".repeat(50));
  console.log(`✅ Fixed: ${totalFixed} files`);
  if (totalErrors > 0) {
    console.log(`❌ Errors: ${totalErrors} files`);
  }
  console.log("\n✨ All fixes completed!");
  console.log('Run "npm run check" to verify remaining issues.\n');
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
