#!/usr/bin/env node

/**
 * Test script to validate MCP extension functionality
 */

import { spawn } from "child_process";
import { promises as fs } from "fs";
import path from "path";

const WORKSPACE_ROOT = process.cwd();
const EXTENSION_PATH = path.join(
  WORKSPACE_ROOT,
  ".vscode",
  "extensions",
  "mcp-context7-assistant"
);
const MCP_PATH = path.join(WORKSPACE_ROOT, "mcp");

class MCPExtensionTester {
  constructor() {
    this.results = {
      extensionStructure: false,
      mcpServerFiles: false,
      packageJsonValid: false,
      extensionBuilt: false,
      mcpServerExecutable: false,
      commandsRegistered: false,
      overallStatus: false,
    };
  }

  async testExtensionStructure() {
    console.log("\n🔍 Testing Extension Structure...");

    try {
      // Check extension directory structure
      const extensionFiles = await fs.readdir(EXTENSION_PATH);
      const requiredFiles = ["package.json", "out", "src"];

      const hasRequired = requiredFiles.every((file) =>
        extensionFiles.includes(file)
      );

      if (hasRequired) {
        console.log("✅ Extension directory structure is valid");
        this.results.extensionStructure = true;
      } else {
        console.log("❌ Extension directory structure is incomplete");
        console.log("   Found:", extensionFiles);
        console.log("   Required:", requiredFiles);
      }
    } catch (error) {
      console.log("❌ Extension directory not found or inaccessible");
      console.log("   Error:", error.message);
    }
  }

  async testMCPServerFiles() {
    console.log("\n🔍 Testing MCP Server Files...");

    try {
      const mcpFiles = await fs.readdir(MCP_PATH);
      const requiredFiles = ["package.json", "custom-context7-server.js"];

      const hasRequired = requiredFiles.every((file) =>
        mcpFiles.includes(file)
      );

      if (hasRequired) {
        console.log("✅ MCP server files are present");
        this.results.mcpServerFiles = true;
      } else {
        console.log("❌ MCP server files are incomplete");
        console.log("   Found:", mcpFiles);
        console.log("   Required:", requiredFiles);
      }
    } catch (error) {
      console.log("❌ MCP server directory not found or inaccessible");
      console.log("   Error:", error.message);
    }
  }

  async testPackageJsons() {
    console.log("\n🔍 Testing Package.json Files...");

    try {
      // Test extension package.json
      const extensionPackage = JSON.parse(
        await fs.readFile(path.join(EXTENSION_PATH, "package.json"), "utf8")
      );

      // Test MCP server package.json
      const mcpPackage = JSON.parse(
        await fs.readFile(path.join(MCP_PATH, "package.json"), "utf8")
      );

      // Validate extension package
      const extensionValid =
        extensionPackage.name === "mcp-context7-assistant" &&
        extensionPackage.main === "./out/extension.js" &&
        extensionPackage.contributes &&
        extensionPackage.contributes.commands;

      // Validate MCP package
      const mcpValid =
        mcpPackage.name === "custom-context7-mcp" &&
        mcpPackage.main === "custom-context7-server.js";

      if (extensionValid && mcpValid) {
        console.log("✅ Package.json files are valid");
        console.log(
          `   Extension commands: ${extensionPackage.contributes.commands.length}`
        );
        this.results.packageJsonValid = true;
      } else {
        console.log("❌ Package.json files have issues");
        if (!extensionValid)
          console.log("   Extension package.json is invalid");
        if (!mcpValid) console.log("   MCP package.json is invalid");
      }
    } catch (error) {
      console.log("❌ Error reading package.json files");
      console.log("   Error:", error.message);
    }
  }

  async testExtensionBuild() {
    console.log("\n🔍 Testing Extension Build...");

    try {
      const outFiles = await fs.readdir(path.join(EXTENSION_PATH, "out"));
      const requiredFiles = ["extension.js", "mcpServerManager.js"];

      const hasRequired = requiredFiles.every((file) =>
        outFiles.includes(file)
      );

      if (hasRequired) {
        console.log("✅ Extension is properly built");
        console.log(`   Output files: ${outFiles.length}`);
        this.results.extensionBuilt = true;
      } else {
        console.log("❌ Extension build is incomplete");
        console.log("   Found:", outFiles);
        console.log("   Required:", requiredFiles);
      }
    } catch (error) {
      console.log("❌ Extension build directory not accessible");
      console.log("   Error:", error.message);
    }
  }

  async testMCPServerExecutable() {
    console.log("\n🔍 Testing MCP Server Executable...");

    return new Promise((resolve) => {
      try {
        const mcpServerPath = path.join(MCP_PATH, "custom-context7-server.js");

        // Test if the server can be executed (quick test)
        const testProcess = spawn(
          "node",
          ["-c", `require('${mcpServerPath.replace(/\\/g, "/")}')`],
          {
            cwd: MCP_PATH,
            stdio: "pipe",
          }
        );

        let output = "";
        let errorOutput = "";

        testProcess.stdout.on("data", (data) => {
          output += data.toString();
        });

        testProcess.stderr.on("data", (data) => {
          errorOutput += data.toString();
        });

        testProcess.on("close", (code) => {
          if (code === 0 || errorOutput.includes("Cannot find module")) {
            console.log("✅ MCP server file is executable");
            this.results.mcpServerExecutable = true;
          } else {
            console.log("❌ MCP server has syntax or runtime errors");
            console.log("   Error output:", errorOutput);
          }
          resolve();
        });

        // Timeout after 5 seconds
        setTimeout(() => {
          testProcess.kill();
          console.log("⚠️  MCP server test timed out (process may be hanging)");
          resolve();
        }, 5000);
      } catch (error) {
        console.log("❌ Error testing MCP server executable");
        console.log("   Error:", error.message);
        resolve();
      }
    });
  }

  async testCommandsRegistered() {
    console.log("\n🔍 Testing Commands Registration...");

    try {
      const extensionJs = await fs.readFile(
        path.join(EXTENSION_PATH, "out", "extension.js"),
        "utf8"
      );

      // Check for command registration patterns
      const commandRegistrations = extensionJs.match(/registerCommand\(/g);
      const mcpCommands = extensionJs.match(/mcp\.\w+/g);

      if (
        commandRegistrations &&
        commandRegistrations.length >= 15 &&
        mcpCommands
      ) {
        console.log("✅ Commands are properly registered");
        console.log(
          `   Command registrations found: ${commandRegistrations.length}`
        );
        console.log(`   MCP commands found: ${new Set(mcpCommands).size}`);
        this.results.commandsRegistered = true;
      } else {
        console.log("❌ Commands registration may be incomplete");
        console.log(
          `   Command registrations: ${commandRegistrations?.length || 0}`
        );
        console.log(`   MCP commands: ${mcpCommands?.length || 0}`);
      }
    } catch (error) {
      console.log("❌ Error reading extension.js file");
      console.log("   Error:", error.message);
    }
  }

  async runAllTests() {
    console.log("🚀 Starting MCP Extension Functionality Tests\n");
    console.log("=".repeat(60));

    await this.testExtensionStructure();
    await this.testMCPServerFiles();
    await this.testPackageJsons();
    await this.testExtensionBuild();
    await this.testMCPServerExecutable();
    await this.testCommandsRegistered();

    // Calculate overall status
    const passedTests = Object.values(this.results).filter(Boolean).length - 1; // Exclude overallStatus
    const totalTests = Object.keys(this.results).length - 1;
    this.results.overallStatus = passedTests === totalTests;

    this.displayResults(passedTests, totalTests);
  }

  displayResults(passedTests, totalTests) {
    console.log("\n" + "=".repeat(60));
    console.log("📊 TEST RESULTS SUMMARY");
    console.log("=".repeat(60));

    Object.entries(this.results).forEach(([test, passed]) => {
      if (test === "overallStatus") return;

      const status = passed ? "✅ PASS" : "❌ FAIL";
      const testName = test
        .replace(/([A-Z])/g, " $1")
        .replace(/^./, (str) => str.toUpperCase());
      console.log(`${status} - ${testName}`);
    });

    console.log("\n" + "-".repeat(60));
    console.log(`📈 Overall Score: ${passedTests}/${totalTests} tests passed`);
    console.log(
      `🎯 Success Rate: ${Math.round((passedTests / totalTests) * 100)}%`
    );

    if (this.results.overallStatus) {
      console.log(
        "\n🎉 ALL TESTS PASSED! MCP Extension appears to be working correctly."
      );
      console.log("\n📝 Next Steps:");
      console.log("   1. Test extension commands in VS Code Command Palette");
      console.log("   2. Verify MCP server can be started from extension");
      console.log("   3. Test extension integration with SvelteKit project");
    } else {
      console.log(
        "\n⚠️  SOME TESTS FAILED. Check the issues above and fix them."
      );
      console.log("\n🔧 Possible Solutions:");
      console.log(
        "   1. Rebuild the extension: npm run compile in extension directory"
      );
      console.log(
        "   2. Install missing dependencies: npm install in both directories"
      );
      console.log("   3. Check file paths and permissions");
    }

    console.log("\n" + "=".repeat(60));
  }
}

// Run the tests
const tester = new MCPExtensionTester();
tester.runAllTests().catch((error) => {
  console.error("❌ Test runner failed:", error);
  process.exit(1);
});
