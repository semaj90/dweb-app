#!/usr/bin/env node

/**
 * MCP Multi-Agent Orchestra Diagnostic & Auto-Fix Script
 * Diagnoses MCP failures and configures auto-switching Ollama setup
 */

import { spawn } from "child_process";
import { promises as fs } from "fs";
import path from "path";

class MCPOrchestrationDiagnostic {
  constructor() {
    this.results = {
      ollama: { native: false, docker: false, port: null },
      mcp: { server: false, extension: false, commands: false },
      agents: { registry: false, orchestrator: false },
      models: { available: [], missing: [] },
      ports: { conflicts: [], available: [] },
    };
    this.recommendations = [];
  }

  log(message, type = "info") {
    const prefix = {
      error: "❌",
      warning: "⚠️",
      success: "✅",
      info: "📋",
    };
    console.log(`${prefix[type]} ${message}`);
  }

  async runDiagnostics() {
    this.log("🔍 MCP MULTI-AGENT ORCHESTRA DIAGNOSTICS", "info");
    this.log("=".repeat(55));
    this.log("");

    await this.checkOllamaConfigurations();
    await this.checkMCPComponents();
    await this.checkAgentOrchestration();
    await this.checkModelsAndPorts();
    await this.generateRecommendations();
    await this.createAutoSwitchSetup();
  }

  async checkOllamaConfigurations() {
    this.log("🔍 Checking Ollama Configurations...");

    // Check native Ollama
    try {
      const nativeCheck = await this.testOllamaEndpoint(
        "http://localhost:11434"
      );
      if (nativeCheck.success) {
        this.results.ollama.native = true;
        this.results.ollama.port = 11434;
        this.log("Native Ollama (11434): Running", "success");
      }
    } catch {
      this.log("Native Ollama (11434): Not available", "warning");
    }

    // Check Docker Ollama on different ports
    const dockerPorts = [11435, 11436, 8080];
    for (const port of dockerPorts) {
      try {
        const dockerCheck = await this.testOllamaEndpoint(
          `http://localhost:${port}`
        );
        if (dockerCheck.success) {
          this.results.ollama.docker = true;
          this.results.ollama.port = port;
          this.log(`Docker Ollama (${port}): Running`, "success");
          break;
        }
      } catch {
        // Continue checking other ports
      }
    }

    if (!this.results.ollama.docker) {
      this.log("Docker Ollama: Not running on any standard port", "warning");
    }

    this.log("");
  }

  async testOllamaEndpoint(url) {
    return new Promise((resolve) => {
      const fetch = require("node-fetch");
      fetch(`${url}/api/tags`, { timeout: 3000 })
        .then((response) => {
          if (response.ok) {
            return response.json();
          }
          throw new Error("Not ok");
        })
        .then((data) => {
          resolve({ success: true, models: data.models || [] });
        })
        .catch(() => {
          resolve({ success: false });
        });
    });
  }

  async checkMCPComponents() {
    this.log("🔍 Checking MCP Components...");

    // Check MCP Server
    try {
      await fs.access("mcp/custom-context7-server.js");
      this.results.mcp.server = true;
      this.log("MCP Server: Found", "success");
    } catch {
      this.log("MCP Server: Missing", "error");
    }

    // Check MCP Extension
    try {
      const extensionPath =
        ".vscode/extensions/mcp-context7-assistant/package.json";
      await fs.access(extensionPath);
      this.results.mcp.extension = true;
      this.log("MCP Extension: Found", "success");
    } catch {
      this.log("MCP Extension: Missing", "error");
    }

    // Check MCP Commands Registration
    try {
      const mcpHelpers = await fs.readFile(
        "sveltekit-frontend/src/lib/utils/mcp-helpers.ts",
        "utf8"
      );
      if (mcpHelpers.includes("copilotOrchestrator")) {
        this.results.mcp.commands = true;
        this.log("MCP Commands: Registered", "success");
      } else {
        this.log("MCP Commands: Missing orchestrator", "warning");
      }
    } catch {
      this.log("MCP Commands: File missing", "error");
    }

    this.log("");
  }

  async checkAgentOrchestration() {
    this.log("🔍 Checking Agent Orchestration...");

    // Check agent orchestrator
    try {
      await fs.access("agent-orchestrator/index.js");
      this.results.agents.orchestrator = true;
      this.log("Agent Orchestrator: Found", "success");
    } catch {
      this.log("Agent Orchestrator: Missing", "error");
    }

    // Check agent registry
    try {
      const helpersPath = "sveltekit-frontend/src/lib/utils/mcp-helpers.ts";
      const content = await fs.readFile(helpersPath, "utf8");
      if (content.includes("agentRegistry")) {
        this.results.agents.registry = true;
        this.log("Agent Registry: Found", "success");
      } else {
        this.log("Agent Registry: Missing", "warning");
      }
    } catch {
      this.log("Agent Registry: File missing", "error");
    }

    this.log("");
  }

  async checkModelsAndPorts() {
    this.log("🔍 Checking Models and Ports...");

    // Check required models
    const requiredModels = ["gemma2:2b", "gemma2-legal", "nomic-embed-text"];
    if (this.results.ollama.port) {
      try {
        const modelsCheck = await this.testOllamaEndpoint(
          `http://localhost:${this.results.ollama.port}`
        );
        if (modelsCheck.success && modelsCheck.models) {
          const availableModels = modelsCheck.models.map((m) => m.name);
          this.results.models.available = availableModels;

          for (const model of requiredModels) {
            if (availableModels.some((m) => m.includes(model))) {
              this.log(`Model ${model}: Available`, "success");
            } else {
              this.log(`Model ${model}: Missing`, "warning");
              this.results.models.missing.push(model);
            }
          }
        }
      } catch {
        this.log("Could not check models", "error");
      }
    }

    this.log("");
  }

  async generateRecommendations() {
    this.log("💡 RECOMMENDATIONS", "info");
    this.log("-".repeat(30));

    if (!this.results.ollama.native && !this.results.ollama.docker) {
      this.recommendations.push("🚀 Start Ollama: Run START-RAG-ALT-PORT.bat");
    }

    if (this.results.models.missing.length > 0) {
      this.recommendations.push(
        "📦 Install missing models: Use service-manager.bat"
      );
    }

    if (!this.results.mcp.server) {
      this.recommendations.push(
        "🔧 Fix MCP Server: Check mcp/custom-context7-server.js"
      );
    }

    if (!this.results.agents.orchestrator) {
      this.recommendations.push(
        "🤖 Setup Agents: Run SETUP-MULTI-AGENT-AI.bat"
      );
    }

    for (const rec of this.recommendations) {
      this.log(rec);
    }

    this.log("");
  }

  async createAutoSwitchSetup() {
    this.log("🔧 Creating Auto-Switch Setup...");

    const autoSwitchScript = `#!/usr/bin/env node

/**
 * Auto-Switch Ollama Configuration
 * Automatically detects and switches between native and Docker Ollama
 */

class OllamaAutoSwitch {
    constructor() {
        this.configurations = [
            { name: 'Native', url: 'http://localhost:11434', priority: 1 },
            { name: 'Docker-Alt', url: 'http://localhost:11435', priority: 2 },
            { name: 'Docker-Std', url: 'http://localhost:8080', priority: 3 }
        ];
        this.activeConfig = null;
    }

    async detectActiveConfiguration() {
        console.log('🔍 Auto-detecting Ollama configuration...');

        for (const config of this.configurations) {
            try {
                const fetch = require('node-fetch');
                const response = await fetch(\`\${config.url}/api/tags\`, { timeout: 3000 });
                if (response.ok) {
                    this.activeConfig = config;
                    console.log(\`✅ Found active Ollama: \${config.name} on \${config.url}\`);
                    return config;
                }
            } catch {
                console.log(\`⚠️ \${config.name} not available on \${config.url}\`);
            }
        }

        console.log('❌ No active Ollama configuration found');
        return null;
    }

    async updateMCPConfiguration() {
        if (!this.activeConfig) {
            console.log('❌ No active configuration to update');
            return;
        }

        console.log(\`🔧 Updating MCP configuration to use \${this.activeConfig.name}...\`);

        // Update environment variables or config files
        const envContent = \`OLLAMA_BASE_URL=\${this.activeConfig.url}
OLLAMA_HOST=\${this.activeConfig.url}
MCP_OLLAMA_URL=\${this.activeConfig.url}
\`;

        const fs = require('fs').promises;
        await fs.writeFile('.env.ollama', envContent);
        console.log('✅ Updated .env.ollama with active configuration');
    }

    async startRecommendedSetup() {
        console.log('🚀 Starting recommended Ollama setup...');

        if (process.platform === 'win32') {
            const { spawn } = require('child_process');

            console.log('Starting Docker Ollama with alternative port...');
            const startScript = spawn('cmd', ['/c', 'START-RAG-ALT-PORT.bat'], {
                stdio: 'inherit',
                detached: true
            });

            console.log('✅ Started Docker Ollama setup');
        } else {
            console.log('Manual setup required for non-Windows platforms');
        }
    }
}

// Auto-run detection
const autoSwitch = new OllamaAutoSwitch();
autoSwitch.detectActiveConfiguration()
    .then(config => {
        if (config) {
            return autoSwitch.updateMCPConfiguration();
        } else {
            console.log('🚀 No active Ollama found, starting recommended setup...');
            return autoSwitch.startRecommendedSetup();
        }
    })
    .catch(console.error);
`;

    await fs.writeFile("auto-switch-ollama.mjs", autoSwitchScript);
    this.log("Created auto-switch-ollama.mjs", "success");
    this.log("");
  }
}

// Run diagnostics
const diagnostic = new MCPOrchestrationDiagnostic();
diagnostic.runDiagnostics().catch(console.error);
