import { execSync } from "child_process";
import fs from "fs";

console.log("🔄 Running project checks and generating TODO.md...");

try {
  // Run npm run check and capture output
  const output = execSync("npm run check", {
    encoding: "utf8",
    cwd: process.cwd(),
    stdio: "pipe",
  });

  console.log("✅ Checks complete. No errors found!");

  // Generate TODO.md
  const todoContent = `# ✅ Project Issues Todo List

Generated on ${new Date().toISOString()}

## Summary
This file contains all TypeScript, Svelte, and other issues found by running \`npm run check\`.

## Results
✅ **No issues found!** Your project is clean.

### What was checked:
- TypeScript compilation
- Svelte component syntax
- Import/export statements
- Type definitions
- ESLint rules (if configured)

### Modern SvelteKit components implemented:
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
1. Test the demo page at \`/modern-demo\`
2. Integrate components into your existing pages
3. Customize styling with CSS custom properties
4. Add more commands to the command menu
5. Implement pgvector for semantic search
`;

  fs.writeFileSync("TODO.md", todoContent);
  console.log("🎉 Success! Generated TODO.md - No issues found!");
} catch (error) {
  console.log("⚠️ Found issues. Parsing output...");

  // Parse the error output
  const errorOutput = error.stdout || error.stderr || error.message;
  const lines = errorOutput.split("\n");

  let todoContent = `# ✅ Project Issues Todo List

Generated on ${new Date().toISOString()}

## Summary
This file contains all TypeScript, Svelte, and other issues found by running \`npm run check\`.

## Issues Found

`;

  let currentFile = "";
  let issueCount = 0;

  for (const line of lines) {
    // Look for file paths
    const fileMatch = line.match(/src[/\\].*\.(svelte|ts|js)/);
    if (fileMatch) {
      const newFile = fileMatch[0];
      if (newFile !== currentFile) {
        currentFile = newFile;
        todoContent += `\n### File: \`${currentFile}\`\n`;
      }
    }

    // Look for errors/warnings
    if (
      line.includes("Error:") ||
      line.includes("Warning:") ||
      line.includes("×") ||
      line.includes("✖")
    ) {
      issueCount++;
      const cleanLine = line
        .replace(/^[✖×]\s*/, "")
        .replace(/^\s*Error:\s*/, "")
        .replace(/^\s*Warning:\s*/, "")
        .trim();
      if (cleanLine) {
        todoContent += `- **Issue #${issueCount}:** ${cleanLine}\n`;
      }
    }
  }

  if (issueCount === 0) {
    todoContent += "\n✅ **No specific issues found in output parsing.**\n";
  } else {
    todoContent += `\n📊 **Total Issues Found:** ${issueCount}\n`;
  }

  todoContent += `\n## Raw Output\n\`\`\`\n${errorOutput}\n\`\`\`\n`;

  fs.writeFileSync("TODO.md", todoContent);
  console.log(`🎉 Generated TODO.md with ${issueCount} issues found.`);
}

// Display the generated file
console.log("\n📄 Generated TODO.md:");
console.log(fs.readFileSync("TODO.md", "utf8"));
