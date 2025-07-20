#!/usr/bin/env node

import { readdir, readFile, writeFile } from "fs/promises";
import { join, extname } from "path";

const FIX_PATTERNS = [
  // Fix User type imports
  {
    pattern:
      /import\s+type\s*{\s*User\s*}\s*from\s*['"](?!.*\/types\/user)[^'"]+['"]/g,
    replacement: "import type { User } from '$lib/types/user'",
  },
  // Fix missing User type imports
  {
    pattern: /^(\s*export\s+let\s+user\s*:\s*User)/m,
    replacement: "import type { User } from '$lib/types/user';\n\n$1",
  },
  // Fix avatar to avatarUrl
  {
    pattern: /user\.avatar(?!Url)/g,
    replacement: "user.avatarUrl",
  },
  // Fix username to name
  {
    pattern: /user\.username/g,
    replacement: "user.name",
  },
  // Fix lucide-svelte User icon conflicts
  {
    pattern:
      /import\s*{\s*([^}]*)\bUser\b([^}]*)\s*}\s*from\s*['"]lucide-svelte['"]/g,
    replacement: (match, before, after) => {
      const imports = (before + "User as UserIcon" + after)
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      return `import { ${imports.join(", ")} } from "lucide-svelte"`;
    },
  },
  // Fix User icon usage
  {
    pattern: /<User\s+(?=size|class)/g,
    replacement: "<UserIcon ",
  },
  // Fix class attribute with multiple mx-auto
  {
    pattern: /class\s*=\s*"mx-auto px-4 max-w-7xl"/g,
    replacement: 'class="container"',
  },
  // Fix NieR component imports in showcase
  {
    pattern:
      /import\s+Header\s+from\s+['"]\$lib\/components\/Header\.svelte['"]/g,
    replacement: "import NierHeader from '$lib/components/NierHeader.svelte'",
  },
  // Fix createEventDispatcher import
  {
    pattern:
      /createEventDispatcher<\{\s*search:\s*{\s*query:\s*string\s*}\s*}>/g,
    replacement: "createEventDispatcher",
  },
];

async function fixSvelteFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = join(dir, entry.name);

    if (
      entry.isDirectory() &&
      !entry.name.includes("node_modules") &&
      !entry.name.includes(".svelte-kit")
    ) {
      await fixSvelteFiles(fullPath);
    } else if (
      entry.isFile() &&
      (extname(entry.name) === ".svelte" || extname(entry.name) === ".ts")
    ) {
      if (entry.name.endsWith(".backup")) continue;

      try {
        let content = await readFile(fullPath, "utf-8");
        let modified = false;

        for (const fix of FIX_PATTERNS) {
          const newContent = content.replace(fix.pattern, fix.replacement);
          if (newContent !== content) {
            content = newContent;
            modified = true;
          }
        }

        // Special fix for missing imports
        if (
          content.includes("export let user: User") &&
          !content.includes("import type { User }")
        ) {
          content =
            "import type { User } from '$lib/types/user';\n\n" + content;
          modified = true;
        }

        if (modified) {
          await writeFile(fullPath, content);
          console.log(`Fixed: ${fullPath}`);
        }
      } catch (error) {
        console.error(`Error processing ${fullPath}:`, error.message);
      }
    }
  }
}

// Run the fixes
console.log("Fixing TypeScript errors in Svelte files...");
fixSvelteFiles(join(process.cwd(), "src"))
  .then(() => {
    console.log("Done!");
  })
  .catch(console.error);
