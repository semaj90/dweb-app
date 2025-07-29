#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

// Quick fixes for immediate SvelteKit errors
const fixes = [
  // ScrollArea.svelte - Fix incomplete render tag
  {
    file: "src/lib/components/ui/scrollarea/ScrollArea.svelte",
    find: "{@render}",
    replace: "<slot />",
  },

  // Context menu components - Fix unterminated functions
  {
    file: "src/lib/components/ui/context-menu/context-menu-item.svelte",
    find: /interface ContextMenuContext \{\s*close: \(\) => void;\s*\}/,
    replace: "interface ContextMenuContext {\n    close: () => void;\n  }",
  },

  // Fix Button component to use slots instead of children prop
  {
    file: "src/lib/components/ui/Button.svelte",
    find: "{#if children}\n    {@render children()}\n  {/if}",
    replace: "<slot />",
  },

  // Fix Card component
  {
    file: "src/lib/components/ui/Card.svelte",
    find: /{@render children\(\)}/g,
    replace: "<slot />",
  },
];

function applyFix(fix) {
  const filePath = path.join(__dirname, fix.file);

  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${fix.file}`);
    return false;
  }

  try {
    let content = fs.readFileSync(filePath, "utf8");
    const before = content;

    if (fix.find instanceof RegExp) {
      content = content.replace(fix.find, fix.replace);
    } else {
      content = content.replace(fix.find, fix.replace);
    }

    if (content !== before) {
      fs.writeFileSync(filePath, content, "utf8");
      console.log(`✅ Fixed: ${fix.file}`);
      return true;
    } else {
      console.log(`⚪ No changes needed: ${fix.file}`);
      return false;
    }
  } catch (error) {
    console.error(`❌ Error fixing ${fix.file}:`, error.message);
    return false;
  }
}

// Apply all fixes
console.log("🔧 Applying SvelteKit error fixes...\n");

let fixedCount = 0;
for (const fix of fixes) {
  if (applyFix(fix)) {
    fixedCount++;
  }
}

console.log(`\n✨ Applied ${fixedCount} fixes`);
console.log("\nNext steps:");
console.log("1. Run: npm run check");
console.log("2. Run: npm run dev");
