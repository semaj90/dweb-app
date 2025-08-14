#!/usr/bin/env node

import { readFileSync, writeFileSync, readdirSync, statSync } from "fs";
import { join, extname } from "path";

console.log("🔧 Fixing critical Svelte 5 runes and component errors...");

const projectRoot = process.cwd();
const svelteFiles = [];

// Find all Svelte files recursively
function findSvelteFiles(dir) {
  try {
    const items = readdirSync(dir);
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      if (
        stat.isDirectory() &&
        !item.startsWith(".") &&
        item !== "node_modules"
      ) {
        findSvelteFiles(fullPath);
      } else if (
        stat.isFile() &&
        (item.endsWith(".svelte") || item.endsWith(".ts"))
      ) {
        svelteFiles.push(fullPath);
      }
    }
  } catch (err) {
    console.log(`Skipping directory: ${dir} (${err.message})`);
  }
}

findSvelteFiles(join(projectRoot, "sveltekit-frontend", "src"));

console.log(`Found ${svelteFiles.length} files to process`);

// Critical fixes for Svelte 5 and components
const fixes = [
  {
    description: "Remove $state imports (Svelte 5 runes are global)",
    pattern: /import\s+\{\s*\$state[^}]*\}\s+from\s+['"']svelte['"'];?\s*/g,
    replacement: "",
  },
  {
    description: "Fix $state usage with proper Svelte 5 runes",
    pattern: /let\s+(\w+)\s*=\s*\$state\s*\(([^)]*)\);?/g,
    replacement: "let $1 = $state($2);",
  },
  {
    description: "Fix createDialog import (doesn't exist in bits-ui)",
    pattern:
      /import\s+\{\s*createDialog[^}]*\}\s+from\s+['"']bits-ui['"'];?\s*/g,
    replacement: "",
  },
  {
    description: "Fix case-sensitive Card import",
    pattern: /from\s+['"]\$lib\/components\/ui\/Card['"']/g,
    replacement: 'from "$lib/components/ui/card"',
  },
  {
    description: "Fix case-sensitive Badge import",
    pattern: /from\s+['"]\$lib\/components\/ui\/Badge['"']/g,
    replacement: 'from "$lib/components/ui/badge"',
  },
  {
    description: "Fix KeyboardEvent type mismatches",
    pattern: /\(e:\s*KeyboardEvent\)\s*=>/g,
    replacement: "(e: CustomEvent<any>) =>",
  },
  {
    description: "Fix Select component exports",
    pattern: /Select\.Root/g,
    replacement: "SelectRoot",
  },
  {
    description: "Fix Select component exports",
    pattern: /Select\.Trigger/g,
    replacement: "SelectTrigger",
  },
  {
    description: "Fix Select component exports",
    pattern: /Select\.Content/g,
    replacement: "SelectContent",
  },
  {
    description: "Fix Select component exports",
    pattern: /Select\.Item/g,
    replacement: "SelectItem",
  },
  {
    description: "Fix Select component exports",
    pattern: /Select\.Value/g,
    replacement: "SelectValue",
  },
  {
    description: "Fix missing export references",
    pattern: /import\s+\{\s*Select\s*\}\s+from\s+['"']bits-ui['"']/g,
    replacement:
      'import { Select as SelectRoot, SelectTrigger, SelectContent, SelectItem, SelectValue } from "bits-ui"',
  },
];

let totalFixes = 0;

for (const filePath of svelteFiles) {
  try {
    let content = readFileSync(filePath, "utf8");
    let fileFixed = false;

    for (const fix of fixes) {
      const matches = content.match(fix.pattern);
      if (matches) {
        content = content.replace(fix.pattern, fix.replacement);
        fileFixed = true;
        totalFixes += matches.length;
        console.log(`  ✓ ${fix.description} (${matches.length} fixes)`);
      }
    }

    if (fileFixed) {
      writeFileSync(filePath, content, "utf8");
      console.log(`📝 Fixed: ${filePath}`);
    }
  } catch (err) {
    console.error(`❌ Error processing ${filePath}:`, err.message);
  }
}

console.log(
  `\n✅ Applied ${totalFixes} fixes across ${svelteFiles.length} files`
);

// Now let's create missing component files
const missingComponents = [
  {
    path: "sveltekit-frontend/src/lib/components/ui/separator/index.ts",
    content: `export { default as Separator } from './Separator.svelte';`,
  },
  {
    path: "sveltekit-frontend/src/lib/components/ui/separator/Separator.svelte",
    content: `<script lang="ts">
  interface Props {
    orientation?: 'horizontal' | 'vertical';
    decorative?: boolean;
    class?: string;
  }

  let { orientation = 'horizontal', decorative = true, class: className, ...restProps }: Props = $props();
</script>

<div
  role={decorative ? 'none' : 'separator'}
  aria-orientation={orientation}
  class="shrink-0 bg-border {orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]'} {className || ''}"
  {...restProps}
></div>`,
  },
  {
    path: "sveltekit-frontend/src/lib/components/ui/select/SelectRoot.svelte",
    content: `<script lang="ts">
  import { Select } from 'bits-ui';

  interface Props {
    value?: string;
    onValueChange?: (value: string) => void;
    children?: any;
  }

  let { value = $bindable(), onValueChange, children, ...restProps }: Props = $props();
</script>

<Select.Root bind:value {onValueChange} {...restProps}>
  {@render children?.()}
</Select.Root>`,
  },
];

for (const component of missingComponents) {
  try {
    const fullPath = join(projectRoot, component.path);
    const dir = fullPath.substring(0, fullPath.lastIndexOf("\\"));

    // Create directory if it doesn't exist
    try {
      statSync(dir);
    } catch {
      import("fs").then((fs) => {
        fs.mkdirSync(dir, { recursive: true });
      });
    }

    writeFileSync(fullPath, component.content, "utf8");
    console.log(`📁 Created: ${component.path}`);
  } catch (err) {
    console.error(`❌ Error creating ${component.path}:`, err.message);
  }
}

console.log("\n🎉 Svelte 5 runes and component fixes completed!");
