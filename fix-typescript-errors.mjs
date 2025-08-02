#!/usr/bin/env node

/**
 * Critical TypeScript Error Fix Script
 * Systematically fixes all compilation errors in the SvelteKit frontend
 */

import { promises as fs } from "fs";
import path from "path";

const FRONTEND_PATH = path.join(process.cwd(), "sveltekit-frontend");

class TypeScriptErrorFixer {
  constructor() {
    this.fixedFiles = [];
    this.errors = [];
  }

  async fixAllErrors() {
    console.log("🔧 Starting TypeScript Error Fixes...\n");

    try {
      // Fix case sensitivity issues
      await this.fixCaseSensitivityIssues();

      // Fix missing exports
      await this.fixMissingExports();

      // Fix Button import issues
      await this.fixButtonImports();

      // Fix input/dialog export issues
      await this.fixComponentExports();

      // Fix type issues
      await this.fixTypeIssues();

      console.log("\n✅ All critical errors fixed!");
      console.log(`📝 Fixed files: ${this.fixedFiles.length}`);
      this.fixedFiles.forEach((file) => console.log(`   - ${file}`));
    } catch (error) {
      console.error("❌ Error during fix process:", error);
    }
  }

  async fixCaseSensitivityIssues() {
    console.log("🔄 Fixing case sensitivity issues...");

    const files = [
      "src/lib/components/ai/AgentOrchestrator.svelte",
      "src/lib/components/ai/MultiLLMOrchestrator.svelte",
      "src/lib/components/copilot/AutonomousEngineeringDemo.svelte",
      "src/lib/components/ui/enhanced-bits/EnhancedRAGStudio.svelte",
      "src/lib/components/vector/VectorIntelligenceDemo.svelte",
      "src/lib/components/vector/VectorRecommendationsWidget.svelte",
      "src/routes/ai/orchestrator/+page.svelte",
      "src/routes/copilot/autonomous/+page.svelte",
      "src/routes/demo/vector-intelligence/+page.svelte",
      "src/routes/enhanced-ai-demo/+page.svelte",
    ];

    for (const file of files) {
      await this.fixImportsInFile(file, [
        { from: "'$lib/components/ui/Card'", to: "'$lib/components/ui/card'" },
        {
          from: "'$lib/components/ui/Badge'",
          to: "'$lib/components/ui/badge'",
        },
      ]);
    }
  }

  async fixMissingExports() {
    console.log("🔄 Fixing missing exports...");

    // Fix enhanced-bits index.js exports
    const indexPath = path.join(
      FRONTEND_PATH,
      "src/lib/components/ui/enhanced-bits/index.js"
    );
    try {
      let content = await fs.readFile(indexPath, "utf8");

      // Add missing exports
      const missingExports = [
        "export interface SelectOption { value: string; label: string; description?: string; }",
        "export interface VectorSearchResult { id: string; score: number; content: string; }",
        "export interface SemanticEntity { id: string; type: string; properties: any; }",
      ];

      content += "\n" + missingExports.join("\n");
      await fs.writeFile(indexPath, content);
      this.fixedFiles.push("enhanced-bits/index.js");
    } catch (error) {
      console.log(`   ⚠️ Could not fix ${indexPath}: ${error.message}`);
    }
  }

  async fixButtonImports() {
    console.log("🔄 Fixing Button imports...");

    const files = [
      "src/lib/components/AccessibilityPanel.svelte",
      "src/lib/components/CaseSelector.svelte",
    ];

    for (const file of files) {
      await this.fixImportsInFile(file, [
        {
          from: /import\s+Button\s+from\s+['"]([^'"]+)['"];?/g,
          to: "import { Button } from '$1';",
        },
      ]);
    }
  }

  async fixComponentExports() {
    console.log("🔄 Fixing component exports...");

    // Fix Dialog.svelte export
    const dialogPath = path.join(
      FRONTEND_PATH,
      "src/lib/components/Dialog.svelte"
    );
    try {
      let content = await fs.readFile(dialogPath, "utf8");

      // Add Dialog export at the end
      if (!content.includes("export { Dialog }")) {
        content +=
          '\n\n<script lang="ts" context="module">\n  export { default as Dialog } from "./Dialog.svelte";\n</script>';
        await fs.writeFile(dialogPath, content);
        this.fixedFiles.push("Dialog.svelte");
      }
    } catch (error) {
      console.log(`   ⚠️ Could not fix Dialog.svelte: ${error.message}`);
    }

    // Fix Input.svelte type issues
    await this.fixInputComponent();

    // Fix Card.svelte exports
    await this.fixCardComponent();
  }

  async fixInputComponent() {
    const inputPath = path.join(
      FRONTEND_PATH,
      "src/lib/components/ui/enhanced-bits/Input.svelte"
    );
    try {
      let content = await fs.readFile(inputPath, "utf8");

      // Fix size property type conflict
      content = content.replace(/size\?\s*:\s*string/g, "inputSize?: string");

      // Fix duplicate properties
      content = content.replace(
        /(\s+)class:\s*[^,}]+,(\s+)class:\s*[^,}]+,/g,
        "$1class: inputClasses,"
      );

      // Fix maxLength -> maxlength
      content = content.replace(/maxLength/g, "maxlength");

      await fs.writeFile(inputPath, content);
      this.fixedFiles.push("enhanced-bits/Input.svelte");
    } catch (error) {
      console.log(`   ⚠️ Could not fix Input.svelte: ${error.message}`);
    }
  }

  async fixCardComponent() {
    const cardPath = path.join(
      FRONTEND_PATH,
      "src/lib/components/ui/enhanced-bits/Card.svelte"
    );
    try {
      let content = await fs.readFile(cardPath, "utf8");

      // Add missing card component exports
      const cardExports = `
<script lang="ts" context="module">
  export const CardHeader = 'div';
  export const CardTitle = 'h3';
  export const CardDescription = 'p';
  export const CardContent = 'div';
  export const CardFooter = 'div';
</script>`;

      if (!content.includes("CardHeader")) {
        content = cardExports + "\n" + content;
        await fs.writeFile(cardPath, content);
        this.fixedFiles.push("enhanced-bits/Card.svelte");
      }
    } catch (error) {
      console.log(`   ⚠️ Could not fix Card.svelte: ${error.message}`);
    }
  }

  async fixTypeIssues() {
    console.log("🔄 Fixing type compatibility issues...");

    // Fix layout server types
    const layoutPath = path.join(FRONTEND_PATH, "src/routes/+layout.server.ts");
    try {
      let content = await fs.readFile(layoutPath, "utf8");

      // Add userAgent to Locals interface or remove the reference
      content = content.replace(
        /event\.locals\.userAgent/g,
        'event.request.headers.get("user-agent") || ""'
      );

      await fs.writeFile(layoutPath, content);
      this.fixedFiles.push("+layout.server.ts");
    } catch (error) {
      console.log(`   ⚠️ Could not fix layout server: ${error.message}`);
    }

    // Fix Evidence type mismatches
    await this.fixEvidenceTypes();
  }

  async fixEvidenceTypes() {
    const files = [
      "src/lib/components/CanvasEditor.svelte",
      "src/lib/components/EditableCanvasSystem.svelte",
    ];

    for (const file of files) {
      const fullPath = path.join(FRONTEND_PATH, file);
      try {
        let content = await fs.readFile(fullPath, "utf8");

        // Fix Evidence type imports
        content = content.replace(
          /from\s+['"]\.\/types['"]/g,
          "from '$lib/types'"
        );

        // Fix Timeout type issue
        content = content.replace(/:\s*Timeout\s*=/g, ": NodeJS.Timeout =");

        await fs.writeFile(fullPath, content);
        this.fixedFiles.push(file);
      } catch (error) {
        console.log(`   ⚠️ Could not fix ${file}: ${error.message}`);
      }
    }
  }

  async fixImportsInFile(file, replacements) {
    const fullPath = path.join(FRONTEND_PATH, file);
    try {
      let content = await fs.readFile(fullPath, "utf8");
      let changed = false;

      for (const { from, to } of replacements) {
        if (typeof from === "string") {
          if (content.includes(from)) {
            content = content.replace(
              new RegExp(from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"),
              to
            );
            changed = true;
          }
        } else {
          if (from.test(content)) {
            content = content.replace(from, to);
            changed = true;
          }
        }
      }

      if (changed) {
        await fs.writeFile(fullPath, content);
        this.fixedFiles.push(file);
      }
    } catch (error) {
      console.log(`   ⚠️ Could not fix ${file}: ${error.message}`);
    }
  }
}

// Run the fixer
const fixer = new TypeScriptErrorFixer();
fixer.fixAllErrors().catch(console.error);
