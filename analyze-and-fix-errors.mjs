#!/usr/bin/env node

/**
 * TypeScript Error Analysis & Priority Fix Script
 * Analyzes svelte-check output and applies targeted fixes
 */

import { spawn } from "child_process";
import { promises as fs } from "fs";
import path from "path";

class ErrorAnalyzer {
  constructor() {
    this.errors = [];
    this.errorCategories = new Map();
    this.priorityFixes = [];
  }

  async analyzeErrors() {
    console.log("🔍 Analyzing TypeScript errors...\n");

    return new Promise((resolve) => {
      const svelteCheck = spawn(
        "npx",
        ["svelte-check", "--threshold", "error", "--output", "human"],
        {
          cwd: "sveltekit-frontend",
          stdio: "pipe",
        }
      );

      let output = "";

      svelteCheck.stdout.on("data", (data) => {
        output += data.toString();
      });

      svelteCheck.stderr.on("data", (data) => {
        output += data.toString();
      });

      svelteCheck.on("close", () => {
        this.parseErrors(output);
        this.categorizeErrors();
        this.generatePriorityFixes();
        resolve();
      });
    });
  }

  parseErrors(output) {
    const errorLines = output
      .split("\n")
      .filter((line) => line.includes("Error:"));

    for (const line of errorLines) {
      const match = line.match(/^(.+):(\d+):(\d+)\s+Error:\s+(.+)\s+\(ts\)$/);
      if (match) {
        const [, file, lineNum, colNum, message] = match;
        this.errors.push({
          file: file.replace(/\\/g, "/"),
          line: parseInt(lineNum),
          column: parseInt(colNum),
          message,
          category: this.categorizeError(message),
        });
      }
    }

    console.log(`📊 Found ${this.errors.length} TypeScript errors`);
  }

  categorizeError(message) {
    if (
      message.includes("Cannot find module") ||
      message.includes("has no exported member")
    ) {
      return "IMPORT_EXPORT";
    }
    if (
      message.includes("does not exist on type") ||
      message.includes("Property")
    ) {
      return "MISSING_PROPERTY";
    }
    if (
      message.includes("Type") &&
      message.includes("is not assignable to type")
    ) {
      return "TYPE_MISMATCH";
    }
    if (message.includes("Cannot find name")) {
      return "MISSING_DECLARATION";
    }
    if (message.includes("Object literal may only specify known properties")) {
      return "INVALID_PROPERTY";
    }
    if (
      message.includes("differs from already included file name") &&
      message.includes("only in casing")
    ) {
      return "CASE_SENSITIVITY";
    }
    return "OTHER";
  }

  categorizeErrors() {
    for (const error of this.errors) {
      const category = error.category;
      if (!this.errorCategories.has(category)) {
        this.errorCategories.set(category, []);
      }
      this.errorCategories.get(category).push(error);
    }

    console.log("\n📋 Error Categories:");
    for (const [category, errors] of this.errorCategories) {
      console.log(`   ${category}: ${errors.length} errors`);
    }
  }

  generatePriorityFixes() {
    console.log("\n🎯 Priority Fixes Needed:\n");

    // Fix 1: Import/Export Issues (Highest Priority)
    if (this.errorCategories.has("IMPORT_EXPORT")) {
      console.log("1. 🔴 CRITICAL: Import/Export Issues");
      const errors = this.errorCategories.get("IMPORT_EXPORT").slice(0, 5);
      errors.forEach((error) => {
        console.log(`   • ${path.basename(error.file)}: ${error.message}`);
      });
      console.log("");
    }

    // Fix 2: Missing Properties (High Priority)
    if (this.errorCategories.has("MISSING_PROPERTY")) {
      console.log("2. 🟡 HIGH: Missing Properties");
      const errors = this.errorCategories.get("MISSING_PROPERTY").slice(0, 5);
      errors.forEach((error) => {
        console.log(`   • ${path.basename(error.file)}: ${error.message}`);
      });
      console.log("");
    }

    // Fix 3: Type Mismatches (Medium Priority)
    if (this.errorCategories.has("TYPE_MISMATCH")) {
      console.log("3. 🟠 MEDIUM: Type Mismatches");
      const errors = this.errorCategories.get("TYPE_MISMATCH").slice(0, 3);
      errors.forEach((error) => {
        console.log(`   • ${path.basename(error.file)}: ${error.message}`);
      });
      console.log("");
    }
  }

  async applyQuickFixes() {
    console.log("🔧 Applying quick fixes...\n");

    // Quick Fix 1: Fix common button imports
    await this.fixButtonImports();

    // Quick Fix 2: Fix missing Dialog exports
    await this.fixDialogExports();

    // Quick Fix 3: Fix common type issues
    await this.fixCommonTypes();

    console.log("✅ Quick fixes applied!\n");
  }

  async fixButtonImports() {
    const buttonErrors = this.errors.filter(
      (e) =>
        e.message.includes("has no default export") &&
        e.message.includes("button")
    );

    console.log(`🔧 Fixing ${buttonErrors.length} button import issues...`);

    for (const error of buttonErrors) {
      try {
        const filePath = path.join("sveltekit-frontend", error.file);
        let content = await fs.readFile(filePath, "utf8");

        // Fix button imports
        content = content.replace(
          /import\s+Button\s+from\s+(['"])([^'"]+button[^'"]*)\1/g,
          "import { Button } from $1$2$1"
        );

        await fs.writeFile(filePath, content);
      } catch (err) {
        // Skip files we can't fix
      }
    }
  }

  async fixDialogExports() {
    const dialogErrors = this.errors.filter(
      (e) =>
        e.message.includes("has no exported member") &&
        e.message.includes("Dialog")
    );

    console.log(`🔧 Fixing ${dialogErrors.length} dialog export issues...`);

    // Ensure Dialog component has proper exports
    try {
      const dialogPath = path.join(
        "sveltekit-frontend",
        "src/lib/components/Dialog.svelte"
      );
      let content = await fs.readFile(dialogPath, "utf8");

      if (!content.includes("export { default as Dialog }")) {
        content +=
          '\n\n<script lang="ts" context="module">\n  export { default as Dialog } from "./Dialog.svelte";\n</script>';
        await fs.writeFile(dialogPath, content);
      }
    } catch (err) {
      console.log("   ⚠️ Could not fix Dialog exports");
    }
  }

  async fixCommonTypes() {
    console.log("🔧 Fixing common type issues...");

    // Add User and Collection types that are missing
    const typesPath = path.join("sveltekit-frontend", "src/lib/types/index.ts");
    try {
      let content = await fs.readFile(typesPath, "utf8");

      const missingTypes = `
// Additional types for compatibility
export interface User {
    id: string;
    name: string;
    email: string;
    role: string;
    permissions: string[];
}

export interface Collection<T> {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
}
`;

      if (!content.includes("interface User")) {
        content += missingTypes;
        await fs.writeFile(typesPath, content);
      }
    } catch (err) {
      console.log("   ⚠️ Could not fix types file");
    }
  }
}

// Run the analyzer
const analyzer = new ErrorAnalyzer();
analyzer
  .analyzeErrors()
  .then(() => analyzer.applyQuickFixes())
  .catch(console.error);
