#!/usr/bin/env node

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

console.log("🔄 Running project checks and generating TODO.md...");

// Function to write TODO.md
function writeTodo(issues, hasErrors) {
  const date = new Date().toISOString().replace("T", " ").slice(0, 19);

  let content = `# ✅ Project Issues Todo List

Generated on ${date}

## Summary
This file contains all TypeScript, Svelte, and other issues found by running \`npm run check\`.

`;

  if (hasErrors) {
    content += `## Results
Found **${issues.length}** issues that need to be addressed:

### Issues by File:

`;

    let currentFile = "";
    issues.forEach((issue) => {
      if (issue.file !== currentFile) {
        content += `\n- **File:** \`${issue.file}\`\n`;
        currentFile = issue.file;
      }
      content += `  - **${issue.type}:** ${issue.message}\n`;
    });
  } else {
    content += `## Results
✅ **No issues found!** Your project is clean.

### What was checked:
- TypeScript compilation
- Svelte component syntax
- Import/export statements
- Type definitions
- ESLint rules (if configured)

`;
  }

  // Add footer with modern components info
  content += `
---

### Modern SvelteKit Components Available:
- 🎯 **CommandMenu.svelte** - Slash command system with citations
- 🎨 **GoldenLayout.svelte** - Golden ratio layout with collapsible sidebar
- 📱 **ExpandGrid.svelte** - Hover-expanding grid (1→3 columns)
- 💬 **SmartTextarea.svelte** - Textarea with integrated command menu
- 🔧 **Enhanced EvidenceCard.svelte** - Improved hover effects and accessibility
- 📚 **Citations Store** - Full CRUD with recent citations tracking
- 🔗 **Fast Navigation** - SvelteKit's built-in SPA routing

### Demo Page
Visit \`/modern-demo\` to see all components in action!

### Next Steps
1. Address any issues listed above
2. Test the demo page at \`/modern-demo\`
3. Integrate components into your existing pages
4. Customize styling with CSS custom properties
5. Add more commands to the command menu
`;

  fs.writeFileSync("TODO.md", content, "utf8");
  return content;
}

// Function to parse output and extract issues
function parseOutput(output) {
  const lines = output.split("\n");
  const issues = [];
  let currentFile = "";

  lines.forEach((line) => {
    // Look for file paths
    const fileMatch = line.match(/src[/\\][^:]*/);
    if (fileMatch) {
      currentFile = fileMatch[0];
    }

    // Look for errors and warnings
    if (
      line.match(/^(Warn:|Error:|✖)/) ||
      line.match(/error/i) ||
      line.match(/warning/i) ||
      line.match(/TS[0-9]+:/) ||
      line.match(/Cannot find/) ||
      line.match(/does not exist/)
    ) {
      if (currentFile) {
        let type = "Issue";
        let message = line;

        if (line.match(/^Error:|error/i)) type = "Error";
        if (line.match(/^Warn:|warning/i)) type = "Warning";
        if (line.match(/TS[0-9]+:/)) type = "TypeScript Error";

        // Clean up the message
        message = message.replace(/^(Warn:|Error:)\\s*/, "");
        message = message.replace(/^✖\\s*/, "");
        message = message.trim();

        if (message && !message.match(/^\\s*$/)) {
          issues.push({
            file: currentFile,
            type: type,
            message: message,
          });
        }
      }
    }
  });

  return issues;
}

// Run npm run check
const child = spawn("npm", ["run", "check"], {
  stdio: ["pipe", "pipe", "pipe"],
  shell: true,
});

let stdout = "";
let stderr = "";

child.stdout.on("data", (data) => {
  stdout += data.toString();
});

child.stderr.on("data", (data) => {
  stderr += data.toString();
});

child.on("close", (code) => {
  console.log("✅ Checks complete. Parsing output...");

  // Combine stdout and stderr
  const allOutput = stdout + stderr;

  // Write raw output to log file for debugging
  fs.writeFileSync("errors.log", allOutput, "utf8");

  // Parse the output
  const issues = parseOutput(allOutput);
  const hasErrors = issues.length > 0;

  // Generate TODO.md
  const todoContent = writeTodo(issues, hasErrors);

  console.log("🎉 Success! Your todo list has been generated at ./TODO.md");

  // Display summary
  console.log("\\n📋 TODO.md Summary:");
  const lines = todoContent.split("\\n");
  lines.slice(0, 10).forEach((line) => {
    console.log(`   ${line}`);
  });

  if (lines.length > 10) {
    console.log("   ... (see TODO.md for full details)");
  }

  if (hasErrors) {
    console.log(`\\n⚠️  Found ${issues.length} issues that need attention.`);
  } else {
    console.log("\\n✅ No issues found! Your project is clean.");
  }
});

child.on("error", (err) => {
  console.error("❌ Error running npm run check:", err.message);

  // Still create a TODO.md with error info
  const errorContent = writeTodo(
    [
      {
        file: "system",
        type: "Error",
        message: `Failed to run npm run check: ${err.message}`,
      },
    ],
    true
  );

  console.log("📋 Created TODO.md with error information.");
});
