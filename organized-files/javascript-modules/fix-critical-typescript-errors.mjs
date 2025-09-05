#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync } from "fs";
import { glob } from "glob";
import path from "path";

console.log("🔧 Starting critical TypeScript error fixes...");

const frontendPath = "./sveltekit-frontend";

// Fix 1: Component Export Issues
const componentExportFixes = {
  "src/lib/components/ai/AskAI.svelte": {
    exportDefault: true,
    scriptLang: "ts",
  },
  "src/lib/components/ai/NierAIAssistant.svelte": {
    exportDefault: true,
    scriptLang: "ts",
  },
  "src/lib/components/ai/EnhancedLegalAIDemo.svelte": {
    exportDefault: true,
    scriptLang: "ts",
  },
  "src/lib/components/ui/dialog/Dialog.svelte": {
    exportDefault: true,
    scriptLang: "ts",
  },
  "src/lib/components/keyboard/KeyboardShortcuts.svelte": {
    exportDefault: true,
    scriptLang: "ts",
  },
};

// Fix 2: UI Component Exports
const uiComponentFixes = {
  "src/lib/components/ui/button/index.ts": `export { default as Button } from './Button.svelte';
export type { ButtonProps } from './Button.svelte';`,

  "src/lib/components/ui/Card/index.ts": `export { default as Card } from './Card.svelte';
export { default as CardContent } from './CardContent.svelte';
export { default as CardHeader } from './CardHeader.svelte';
export { default as CardTitle } from './CardTitle.svelte';`,

  "src/lib/components/ui/Badge/index.ts": `export { default as Badge } from './Badge.svelte';
export type { BadgeProps } from './Badge.svelte';`,
};

// Fix 3: Missing Dependencies
const missingDependencies = ["fuse.js"];

// Fix 4: Type Definition Fixes
const typeDefinitionFixes = {
  "src/lib/types/index.ts": `
// Core Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'prosecutor' | 'investigator' | 'analyst';
  preferences?: {
    theme: 'light' | 'dark' | 'auto';
    aiProvider: 'ollama' | 'openai' | 'anthropic';
  };
}

export interface Case {
  id: string;
  title: string;
  description: string;
  status: 'open' | 'closed' | 'pending' | 'investigating' | 'archived';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignedTo?: string;
  createdAt: Date;
  updatedAt: Date;
  metadata?: Record<string, any>;
}

export interface Evidence {
  id: string;
  title: string;
  description?: string;
  content: string;
  type: string;
  evidenceType?: string;
  caseId?: string;
  metadata?: any;
  analysis?: {
    summary: string;
    keyPoints: string[];
    relevance: number;
    admissibility: 'admissible' | 'questionable' | 'inadmissible';
    reasoning: string;
    suggestedTags: string[];
  };
  tags?: string[];
  similarEvidence?: Array<{
    id: string;
    title: string;
    similarity: number;
  }>;
  createdAt?: Date;
  updatedAt?: Date;
  thumbnailUrl?: string;
  fileSize?: number;
}

export interface Report {
  id: string;
  title: string;
  content: string;
  caseId?: string;
  reportType?: string;
  status?: 'draft' | 'review' | 'final';
  createdAt: Date;
  updatedAt: Date;
}

export interface HelpArticle {
  id: string;
  category: string;
  title: string;
  description: string;
  type: string;
  duration: string;
  popularity: number;
  tags: string[];
  lastUpdated?: string;
  content: string;
}

export interface AnalysisResults {
  summary?: string;
  keyEntities: Array<{
    text: string;
    type: string;
    confidence: number;
  }>;
  similarity?: number;
  riskAssessment?: string;
  classification?: any;
  error?: string;
}
`,
};

function ensureComponentExport(filePath) {
  if (!existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  let content = readFileSync(filePath, "utf-8");

  // Check if component already has proper export
  if (content.includes("export default")) {
    return;
  }

  // Add export default at the end of script if missing
  const scriptMatch = content.match(/<script[^>]*>([\s\S]*?)<\/script>/);
  if (scriptMatch) {
    const scriptContent = scriptMatch[1];
    const componentNameMatch = path.basename(filePath, ".svelte");

    // Add export default if missing
    if (!scriptContent.includes("export default")) {
      const newScriptContent =
        scriptContent.trim() +
        "\n\n// Auto-added default export\nexport default {};\n";
      content = content.replace(
        scriptMatch[0],
        `<script lang="ts">\n${newScriptContent}</script>`
      );
      writeFileSync(filePath, content);
      console.log(`✅ Added default export to ${filePath}`);
    }
  }
}

function fixUIComponentExports() {
  console.log("🔧 Fixing UI component exports...");

  for (const [filePath, content] of Object.entries(uiComponentFixes)) {
    const fullPath = path.join(frontendPath, filePath);
    const dir = path.dirname(fullPath);

    // Create directory if it doesn't exist
    if (!existsSync(dir)) {
      import("fs").then((fs) => fs.mkdirSync(dir, { recursive: true }));
    }

    writeFileSync(fullPath, content);
    console.log(`✅ Created/Updated ${filePath}`);
  }
}

function fixTypeDefinitions() {
  console.log("🔧 Fixing type definitions...");

  for (const [filePath, content] of Object.entries(typeDefinitionFixes)) {
    const fullPath = path.join(frontendPath, filePath);
    const dir = path.dirname(fullPath);

    // Create directory if it doesn't exist
    if (!existsSync(dir)) {
      import("fs").then((fs) => fs.mkdirSync(dir, { recursive: true }));
    }

    writeFileSync(fullPath, content);
    console.log(`✅ Updated ${filePath}`);
  }
}

function fixSvelteExports() {
  console.log("🔧 Fixing Svelte component exports...");

  for (const [componentPath, config] of Object.entries(componentExportFixes)) {
    const fullPath = path.join(frontendPath, componentPath);
    ensureComponentExport(fullPath);
  }
}

function fixUnterminatedStrings() {
  console.log("🔧 Fixing unterminated string constants...");

  const svelteFiles = glob.sync(`${frontendPath}/src/**/*.svelte`);

  for (const filePath of svelteFiles) {
    if (!existsSync(filePath)) continue;

    let content = readFileSync(filePath, "utf-8");
    let changed = false;

    // Common unterminated string patterns
    const fixes = [
      // Fix class attributes with missing quotes
      {
        pattern: /class="container mx-auto px-4"(?!\s*[}>])/g,
        replacement: 'class="container mx-auto px-4"',
      },
      // Fix transition attributes
      {
        pattern: /transition:(\w+)=\{\{\s*[^}]*(?!")/g,
        replacement: (match) => match + '"',
      },
      // Fix incomplete template strings
      { pattern: /\$\{1(?!\d)/g, replacement: "${1}" },
    ];

    for (const fix of fixes) {
      const newContent = content.replace(fix.pattern, fix.replacement);
      if (newContent !== content) {
        content = newContent;
        changed = true;
      }
    }

    if (changed) {
      writeFileSync(filePath, content);
      console.log(
        `✅ Fixed string issues in ${path.relative(frontendPath, filePath)}`
      );
    }
  }
}

async function installMissingDependencies() {
  console.log("📦 Installing missing dependencies...");

  const { exec } = await import("child_process");
  const { promisify } = await import("util");
  const execAsync = promisify(exec);

  for (const dep of missingDependencies) {
    try {
      console.log(`Installing ${dep}...`);
      await execAsync(`cd ${frontendPath} && npm install ${dep}`, {
        cwd: process.cwd(),
      });
      console.log(`✅ Installed ${dep}`);
    } catch (error) {
      console.log(`⚠️  Failed to install ${dep}: ${error.message}`);
    }
  }
}

// Main execution
async function main() {
  try {
    // 1. Fix type definitions first
    fixTypeDefinitions();

    // 2. Fix UI component exports
    fixUIComponentExports();

    // 3. Fix Svelte component exports
    fixSvelteExports();

    // 4. Fix unterminated strings
    fixUnterminatedStrings();

    // 5. Install missing dependencies
    await installMissingDependencies();

    console.log("\n🎉 Critical fixes completed!");
    console.log("\n📋 Next steps:");
    console.log("1. Run `npm run check` again to see remaining issues");
    console.log("2. Fix remaining component-specific issues manually");
    console.log("3. Consider running `npm run format` to fix formatting");
  } catch (error) {
    console.error("❌ Error during fixes:", error);
    process.exit(1);
  }
}

main();
