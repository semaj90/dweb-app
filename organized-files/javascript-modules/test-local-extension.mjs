#!/usr/bin/env node

/**
 * Local VS Code Extension Testing Script
 * Comprehensive testing of the Context7 MCP Assistant extension
 */

import { promises as fs } from "fs";
import { spawn } from "child_process";
import path from "path";

class ExtensionTester {
  constructor() {
    this.testResults = [];
    this.extensionPath = ".vscode/extensions/mcp-context7-assistant";
  }

  log(message) {
    console.log(message);
  }

  addResult(test, status, details = "") {
    this.testResults.push({ test, status, details });
    const icon = status === "pass" ? "✅" : status === "fail" ? "❌" : "⚠️";
    this.log(`${icon} ${test}${details ? ": " + details : ""}`);
  }

  async runAllTests() {
    this.log("🧪 LOCAL VS CODE EXTENSION TESTING");
    this.log("=".repeat(50));
    this.log("");

    await this.testExtensionStructure();
    await this.testExtensionManifest();
    await this.testExtensionCode();
    await this.testMCPIntegration();
    await this.generateVSCodeTestInstructions();

    this.log("");
    this.generateSummary();
  }

  async testExtensionStructure() {
    this.log("📁 Testing Extension Structure...");

    const requiredFiles = ["package.json", "out/extension.js", "README.md"];

    for (const file of requiredFiles) {
      const filePath = path.join(this.extensionPath, file);
      try {
        await fs.access(filePath);
        this.addResult(`Structure: ${file}`, "pass");
      } catch {
        this.addResult(`Structure: ${file}`, "fail", "File missing");
      }
    }
    this.log("");
  }

  async testExtensionManifest() {
    this.log("📋 Testing Extension Manifest...");

    try {
      const packagePath = path.join(this.extensionPath, "package.json");
      const packageJson = JSON.parse(await fs.readFile(packagePath, "utf8"));

      // Test basic properties
      if (packageJson.name === "mcp-context7-assistant") {
        this.addResult("Manifest: Name", "pass");
      } else {
        this.addResult("Manifest: Name", "fail");
      }

      if (packageJson.main === "./out/extension.js") {
        this.addResult("Manifest: Entry point", "pass");
      } else {
        this.addResult("Manifest: Entry point", "fail");
      }

      // Test commands
      const commands = packageJson.contributes?.commands || [];
      if (commands.length >= 15) {
        this.addResult(
          "Manifest: Commands",
          "pass",
          `${commands.length} commands`
        );
      } else {
        this.addResult(
          "Manifest: Commands",
          "fail",
          `Only ${commands.length} commands`
        );
      }

      // Test activation events
      const activationEvents = packageJson.activationEvents || [];
      if (activationEvents.includes("onStartupFinished")) {
        this.addResult("Manifest: Activation", "pass");
      } else {
        this.addResult("Manifest: Activation", "warn", "No startup activation");
      }
    } catch (error) {
      this.addResult("Manifest: Parse", "fail", error.message);
    }
    this.log("");
  }

  async testExtensionCode() {
    this.log("🔧 Testing Extension Code...");

    try {
      const extensionPath = path.join(this.extensionPath, "out/extension.js");
      const extensionCode = await fs.readFile(extensionPath, "utf8");

      // Test for required exports
      if (
        extensionCode.includes("activate") &&
        extensionCode.includes("deactivate")
      ) {
        this.addResult("Code: Exports", "pass");
      } else {
        this.addResult("Code: Exports", "fail", "Missing activate/deactivate");
      }

      // Test for command registration
      if (extensionCode.includes("registerCommand")) {
        this.addResult("Code: Command registration", "pass");
      } else {
        this.addResult("Code: Command registration", "fail");
      }

      // Test for MCP integration
      if (extensionCode.includes("mcp") || extensionCode.includes("MCP")) {
        this.addResult("Code: MCP integration", "pass");
      } else {
        this.addResult(
          "Code: MCP integration",
          "warn",
          "Limited MCP references"
        );
      }
    } catch (error) {
      this.addResult("Code: Parse", "fail", error.message);
    }
    this.log("");
  }

  async testMCPIntegration() {
    this.log("🔗 Testing MCP Integration...");

    try {
      // Test MCP server files
      const mcpServerPath = "mcp/custom-context7-server.js";
      await fs.access(mcpServerPath);
      this.addResult("MCP: Server file", "pass");

      const mcpPackagePath = "mcp/package.json";
      const mcpPackage = JSON.parse(await fs.readFile(mcpPackagePath, "utf8"));

      if (mcpPackage.main === "custom-context7-server.js") {
        this.addResult("MCP: Server config", "pass");
      } else {
        this.addResult("MCP: Server config", "fail");
      }

      // Check for required MCP dependencies
      const requiredDeps = ["@modelcontextprotocol/sdk"];
      const hasDeps = requiredDeps.every(
        (dep) =>
          mcpPackage.dependencies?.[dep] || mcpPackage.devDependencies?.[dep]
      );

      if (hasDeps) {
        this.addResult("MCP: Dependencies", "pass");
      } else {
        this.addResult("MCP: Dependencies", "warn", "Some MCP deps missing");
      }
    } catch (error) {
      this.addResult("MCP: Integration", "fail", error.message);
    }
    this.log("");
  }

  async generateVSCodeTestInstructions() {
    this.log("🚀 VS CODE MANUAL TESTING INSTRUCTIONS");
    this.log("-".repeat(50));
    this.log("");

    this.log("1. 📂 OPEN WORKSPACE:");
    this.log("   • Open VS Code in this directory:");
    this.log("     c:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app");
    this.log("");

    this.log("2. 🔍 TEST COMMAND PALETTE:");
    this.log("   • Press Ctrl+Shift+P");
    this.log('   • Type "Context7" - you should see:');
    this.log("     ✨ Context7 MCP: Analyze Current Context");
    this.log("     📚 Context7 MCP: Get Context-Aware Documentation");
    this.log("     🎛️ Context7 MCP: Open EnhancedRAG Studio");
    this.log("     🤖 Context7 MCP: Run Agent Orchestrator");
    this.log("     🐛 Context7 MCP: Analyze TypeScript Errors");
    this.log("");

    this.log("3. 🧪 TEST COMMANDS:");
    this.log('   A. Try "🔍 Analyze Current Context":');
    this.log("      → Should analyze current file/workspace");
    this.log('   B. Try "📚 Get Context-Aware Documentation":');
    this.log("      → Should provide relevant docs");
    this.log('   C. Try "🎛️ Open EnhancedRAG Studio":');
    this.log("      → Should open/navigate to RAG interface");
    this.log("");

    this.log("4. 🔧 TEST CONTEXT MENU:");
    this.log("   • Right-click in editor");
    this.log("   • Look for Context7 MCP options");
    this.log("");

    this.log("5. 📊 EXPECTED BEHAVIOR:");
    this.log("   ✅ Commands execute without errors");
    this.log("   ✅ Status bar shows MCP activity");
    this.log("   ✅ Output panel shows MCP logs");
    this.log("   ✅ Extension remains responsive");
    this.log("");
  }

  generateSummary() {
    this.log("📊 TEST SUMMARY");
    this.log("=".repeat(50));

    const passed = this.testResults.filter((r) => r.status === "pass").length;
    const failed = this.testResults.filter((r) => r.status === "fail").length;
    const warnings = this.testResults.filter((r) => r.status === "warn").length;
    const total = this.testResults.length;

    this.log(`✅ Passed: ${passed}`);
    this.log(`❌ Failed: ${failed}`);
    this.log(`⚠️ Warnings: ${warnings}`);
    this.log(`📊 Total: ${total}`);
    this.log("");

    const successRate = Math.round((passed / total) * 100);
    this.log(`🎯 Success Rate: ${successRate}%`);

    if (successRate >= 80) {
      this.log("🎉 EXTENSION IS READY FOR TESTING!");
      this.log("");
      this.log("🚀 QUICK START:");
      this.log("   1. Open VS Code in this workspace");
      this.log("   2. Press Ctrl+Shift+P");
      this.log('   3. Type "Context7" and test commands');
    } else if (successRate >= 60) {
      this.log("⚠️ Extension mostly functional but needs attention");
    } else {
      this.log("❌ Extension needs significant fixes");
    }

    this.log("");
    this.log("=".repeat(50));
  }
}

// Run all tests
const tester = new ExtensionTester();
tester.runAllTests().catch(console.error);
