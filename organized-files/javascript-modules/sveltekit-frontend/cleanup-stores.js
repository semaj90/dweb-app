#!/usr/bin/env node

/**
 * Store Merge Cleanup Script
 * Removes duplicate files and creates backup directory
 */

const fs = require("fs");
const path = require("path");

const storesDir =
  "C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/src/lib/stores";
const backupDir = path.join(storesDir, "backup-duplicates");

// Files to backup and remove (duplicates that are now unified)
const duplicateFiles = [
  "ai-commands.js",
  "ai-command-parser.js",
  "evidence.ts",
  "evidenceStore.ts",
  "evidence-store.ts",
];

// Files with .backup extensions to clean up
const backupFiles = [
  "ai-store.ts.backup",
  "auth.ts.backup",
  "autoTaggingMachine.ts.backup",
  "avatarStore.ts.backup",
  "cases.ts.backup",
  "casesStore.ts.backup",
  "chatStore.ts.backup",
  "citations.ts.backup",
  "error-handler.ts.backup",
  "evidence-store.ts.backup",
  "evidence.ts.backup",
  "evidenceStore.ts.backup",
  "form.ts.backup",
  "lokiStore.ts.backup",
  "modal.ts.backup",
  "notification.ts.backup",
  "report.ts.backup",
  "saved-notes.ts.backup",
  "ui.ts.backup",
  "user.ts.backup",
];

console.log("🧹 Phase 2 Store Cleanup Starting...");

// Create backup directory
if (!fs.existsSync(backupDir)) {
  fs.mkdirSync(backupDir, { recursive: true });
  console.log("📁 Created backup directory");
}

// Move duplicates to backup
duplicateFiles.forEach((file) => {
  const sourcePath = path.join(storesDir, file);
  const backupPath = path.join(backupDir, file);

  if (fs.existsSync(sourcePath)) {
    fs.renameSync(sourcePath, backupPath);
    console.log(`📦 Moved ${file} to backup`);
  }
});

// Remove .backup files
backupFiles.forEach((file) => {
  const filePath = path.join(storesDir, file);

  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
    console.log(`🗑️ Removed ${file}`);
  }
});

// Create summary
const summary = {
  unifiedStores: [
    "ai-unified.ts (merged ai-commands + ai-command-parser)",
    "evidence-unified.ts (merged evidence + evidenceStore)",
  ],
  activeStores: fs
    .readdirSync(storesDir)
    .filter((f) => f.endsWith(".ts") || f.endsWith(".js"))
    .filter((f) => !f.includes("backup") && !f.includes("unified")).length,
  backupCount: duplicateFiles.length + backupFiles.length,
};

console.log("\n✅ Cleanup Complete!");
console.log(`📊 Active stores: ${summary.activeStores}`);
console.log(`📦 Files backed up: ${summary.backupCount}`);
console.log(`🎯 Unified stores: ${summary.unifiedStores.length}`);

console.log("\n🚀 Phase 2 stores ready for launch!");
