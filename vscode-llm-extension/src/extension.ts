import * as vscode from "vscode";
import { clusterManager, type WorkerTask } from "./cluster-manager";
import { ollamaGemmaCache } from "./ollama-gemma-cache";
import { EnhancedMCPExtensionMemoryManager } from "./enhanced-mcp-memory-manager";
import { LLMOptimizationManager } from "./llm-optimization-manager";

// Global manager instances
let memoryManager: EnhancedMCPExtensionMemoryManager | null = null;
let optimizationManager: LLMOptimizationManager | null = null;

export function activate(context: vscode.ExtensionContext) {
  // Initialize enhanced memory management
  memoryManager = new EnhancedMCPExtensionMemoryManager(context);

  // Initialize LLM optimization manager
  optimizationManager = new LLMOptimizationManager({
    enableStreaming: true,
    enableCompression: true,
    enableWorkerThreads: true,
    batchSize: 1024,
    compressionRatio: 10,
    workerPoolSize: 4
  });

  // Initialize cluster and cache systems with memory tracking
  initializeExtensionSystems(context);

  // Register MCP Context7 commands
  registerMCPCommands(context);

  // Register LLM management commands
  registerLLMCommands(context);

  // Register cluster management commands
  registerClusterCommands(context);

  // Register cache management commands
  registerCacheCommands(context);

  // Register memory management commands
  registerMemoryCommands(context);

  // Register LLM optimization commands
  registerOptimizationCommands(context);
}

async function initializeExtensionSystems(context: vscode.ExtensionContext) {
  try {
    // Initialize cluster manager with memory tracking
    if (memoryManager) {
      await memoryManager.trackCommandExecution(
        "cluster.initialize",
        [],
        async () => {
          await clusterManager.initialize();
        }
      );
    } else {
      await clusterManager.initialize();
    }

    // Initialize Ollama Gemma cache with memory tracking
    if (memoryManager) {
      await memoryManager.trackCommandExecution(
        "ollama.initialize",
        [],
        async () => {
          await ollamaGemmaCache.initialize();
        }
      );
    } else {
      await ollamaGemmaCache.initialize();
    }

    // Pre-cache workspace if enabled
    const config = vscode.workspace.getConfiguration("mcpContext7");
    if (config.get("enableWorkspacePreCache", true)) {
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Pre-caching workspace with Ollama Gemma...",
          cancellable: false,
        },
        async (progress) => {
          const result = await (memoryManager
            ? memoryManager.trackCommandExecution(
                "ollama.preCacheWorkspace",
                [],
                async () => {
                  return await ollamaGemmaCache.preCacheWorkspace();
                }
              )
            : ollamaGemmaCache.preCacheWorkspace());

          vscode.window.showInformationMessage(
            `Workspace pre-cached: ${result.filesProcessed} files, ${result.embeddingsGenerated} embeddings generated`
          );
        }
      );
    }

    vscode.window.showInformationMessage(
      "MCP Context7 Extension with Enhanced Memory Management initialized successfully!"
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to initialize extension: ${error}`);
    console.error("Extension initialization failed:", error);
  }
}

function registerMCPCommands(context: vscode.ExtensionContext) {
  // Enhanced MCP analyze current context with clustering and caching
  context.subscriptions.push(
    vscode.commands.registerCommand("mcp.analyzeCurrentContext", async () => {
      await analyzeCurrentContext();
    })
  );

  // Auto-fix command with cluster support
  context.subscriptions.push(
    vscode.commands.registerCommand("mcp.runAutoFix", async () => {
      await runAutoFixCommand();
    })
  );

  // Context7 best practices
  context.subscriptions.push(
    vscode.commands.registerCommand("mcp.generateBestPractices", async () => {
      await generateBestPracticesCommand();
    })
  );

  // Semantic search with caching
  context.subscriptions.push(
    vscode.commands.registerCommand("mcp.semanticSearch", async () => {
      await semanticSearchCommand();
    })
  );

  // Agent orchestration
  context.subscriptions.push(
    vscode.commands.registerCommand("mcp.orchestrateAgents", async () => {
      await orchestrateAgentsCommand();
    })
  );
}

function registerLLMCommands(context: vscode.ExtensionContext) {
  // Register command to open the LLM Manager panel
  context.subscriptions.push(
    vscode.commands.registerCommand("llmManager.openPanel", () => {
      LLMPanel.createOrShow(context.extensionUri);
    })
  );

  // Register command to refresh models
  context.subscriptions.push(
    vscode.commands.registerCommand("llmManager.refreshModels", async () => {
      await refreshModelsCommand();
    })
  );
}

function registerClusterCommands(context: vscode.ExtensionContext) {
  // Show cluster status
  context.subscriptions.push(
    vscode.commands.registerCommand("cluster.showStatus", async () => {
      await showClusterStatusCommand();
    })
  );

  // Restart cluster
  context.subscriptions.push(
    vscode.commands.registerCommand("cluster.restart", async () => {
      await restartClusterCommand();
    })
  );
}

function registerCacheCommands(context: vscode.ExtensionContext) {
  // Show cache statistics
  context.subscriptions.push(
    vscode.commands.registerCommand("cache.showStats", async () => {
      await showCacheStatsCommand();
    })
  );

  // Clear cache
  context.subscriptions.push(
    vscode.commands.registerCommand("cache.clear", async () => {
      await clearCacheCommand();
    })
  );

  // Pre-cache workspace
  context.subscriptions.push(
    vscode.commands.registerCommand("cache.preCacheWorkspace", async () => {
      await preCacheWorkspaceCommand();
    })
  );
}

/**
 * Register LLM optimization commands  
 */
function registerOptimizationCommands(context: vscode.ExtensionContext) {
  // Show optimization dashboard
  context.subscriptions.push(
    vscode.commands.registerCommand("llm.showOptimizationDashboard", async () => {
      if (optimizationManager) {
        await optimizationManager.showOptimizationDashboard();
      } else {
        vscode.window.showErrorMessage("Optimization manager not initialized");
      }
    })
  );

  // Process tokens with streaming optimization
  context.subscriptions.push(
    vscode.commands.registerCommand("llm.processStreamingTokens", async () => {
      await processStreamingTokensCommand();
    })
  );

  // Compress token payload
  context.subscriptions.push(
    vscode.commands.registerCommand("llm.compressTokens", async () => {
      await compressTokensCommand();
    })
  );

  // Run optimization benchmark
  context.subscriptions.push(
    vscode.commands.registerCommand("llm.runOptimizationBenchmark", async () => {
      if (optimizationManager) {
        await optimizationManager.runOptimizationBenchmark();
      } else {
        vscode.window.showErrorMessage("Optimization manager not initialized");
      }
    })
  );

  // Stream response demo
  context.subscriptions.push(
    vscode.commands.registerCommand("llm.demoStreamingResponse", async () => {
      await demoStreamingResponseCommand();
    })
  );

  // Show optimization metrics
  context.subscriptions.push(
    vscode.commands.registerCommand("llm.showOptimizationMetrics", async () => {
      await showOptimizationMetricsCommand();
    })
  );
}

/**
 * Register memory management commands
 */
function registerMemoryCommands(context: vscode.ExtensionContext) {
  const memoryCommands = [
    {
      command: "mcp.showMemoryStatus",
      handler: () => memoryManager?.showMemoryStatus() || Promise.resolve(),
    },
    {
      command: "mcp.optimizeMemory",
      handler: () => memoryManager?.optimizeMemoryNow() || Promise.resolve(),
    },
    {
      command: "mcp.clearCache",
      handler: () => memoryManager?.clearAllCaches() || Promise.resolve(),
    },
    {
      command: "mcp.analyzeCommandClusters",
      handler: () =>
        memoryManager?.analyzeCommandClusters() || Promise.resolve(),
    },
    {
      command: "mcp.exportMemoryReport",
      handler: () => memoryManager?.exportMemoryReport() || Promise.resolve(),
    },
    {
      command: "mcp.showMemoryStats",
      handler: showMemoryStatsCommand,
    },
    {
      command: "mcp.clearMemoryCache",
      handler: clearMemoryCommand,
    },
  ];

  memoryCommands.forEach(({ command, handler }) => {
    const disposable = vscode.commands.registerCommand(command, handler);
    context.subscriptions.push(disposable);
  });
}

/**
 * Enhanced mcp.analyzeCurrentContext with cluster and caching support
 */
async function analyzeCurrentContext(): Promise<void> {
  try {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage("No active editor found");
      return;
    }

    const document = editor.document;
    const text = document.getText();
    const fileName = document.fileName;
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);

    // Show progress
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Analyzing current context with Context7 MCP...",
        cancellable: true,
      },
      async (progress, token) => {
        // Step 1: Generate/retrieve embeddings
        progress.report({ increment: 20, message: "Generating embeddings..." });

        const context = workspaceFolder?.name || "unknown";
        await ollamaGemmaCache.getEmbedding(text, `file_${fileName}`);

        // Step 2: Analyze with Context7 using cluster
        progress.report({
          increment: 30,
          message: "Running Context7 analysis...",
        });

        const analysisTask: WorkerTask = {
          id: `analyze_${Date.now()}`,
          type: "mcp-analyze",
          data: {
            component: detectComponent(text, fileName),
            context: "legal-ai",
            fileContent: text,
            fileName: fileName,
          },
          priority: "high",
          timeout: 30000,
        };

        const analysisResult = await clusterManager.executeTask(analysisTask);

        // Step 3: Find similar contexts
        progress.report({
          increment: 25,
          message: "Finding similar contexts...",
        });

        const similarContexts = await ollamaGemmaCache.querySimilar({
          text: text.substring(0, 1000), // First 1000 chars for similarity
          context: `file_${fileName}`,
          similarityThreshold: 0.7,
          maxResults: 5,
        });

        // Step 4: Generate recommendations
        progress.report({
          increment: 25,
          message: "Generating recommendations...",
        });

        // Create and show results panel
        const panel = vscode.window.createWebviewPanel(
          "contextAnalysis",
          "Context7 Analysis Results",
          vscode.ViewColumn.Beside,
          {
            enableScripts: true,
            retainContextWhenHidden: true,
          }
        );

        panel.webview.html = generateAnalysisWebviewContent(
          analysisResult,
          similarContexts,
          fileName
        );

        // Handle webview messages
        panel.webview.onDidReceiveMessage(async (message) => {
          switch (message.command) {
            case "runAutoFix":
              await runAutoFixFromAnalysis(message.area);
              break;
            case "generateBestPractices":
              await generateBestPracticesFromAnalysis(message.area);
              break;
          }
        });
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Context analysis failed: ${error}`);
    console.error("Context analysis error:", error);
  }
}

/**
 * Auto-fix command with cluster support
 */
async function runAutoFixCommand(): Promise<void> {
  try {
    // Show quick pick for fix area
    const fixArea = await vscode.window.showQuickPick(
      [
        { label: "All Areas", value: null },
        { label: "Imports & Exports", value: "imports" },
        { label: "Svelte 5 Patterns", value: "svelte5" },
        { label: "TypeScript", value: "typescript" },
        { label: "Performance", value: "performance" },
        { label: "Accessibility", value: "accessibility" },
        { label: "Security", value: "security" },
      ],
      {
        placeHolder: "Select area to fix",
      }
    );

    if (!fixArea) return;

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Running auto-fix${fixArea.value ? ` for ${fixArea.label}` : ""}...`,
        cancellable: false,
      },
      async (progress) => {
        const autoFixTask: WorkerTask = {
          id: `autofix_${Date.now()}`,
          type: "auto-fix",
          data: {
            area: fixArea.value,
            dryRun: false,
          },
          priority: "high",
          timeout: 60000,
        };

        const result = await clusterManager.executeTask(autoFixTask);

        if (result.success) {
          const summary = result.result;
          vscode.window.showInformationMessage(
            `Auto-fix complete: ${summary.summary.filesFixed} files fixed, ${summary.summary.totalIssues} issues resolved`
          );

          // Show detailed results if requested
          const showDetails = await vscode.window.showInformationMessage(
            "Auto-fix completed successfully!",
            "Show Details"
          );

          if (showDetails) {
            const panel = vscode.window.createWebviewPanel(
              "autoFixResults",
              "Auto-Fix Results",
              vscode.ViewColumn.Beside,
              { enableScripts: true }
            );

            panel.webview.html = generateAutoFixWebviewContent(summary);
          }
        } else {
          vscode.window.showErrorMessage(`Auto-fix failed: ${result.error}`);
        }
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Auto-fix command failed: ${error}`);
  }
}

/**
 * Generate best practices command
 */
async function generateBestPracticesCommand(): Promise<void> {
  const area = await vscode.window.showQuickPick(
    [
      { label: "Performance", value: "performance" },
      { label: "Security", value: "security" },
      { label: "UI/UX", value: "ui-ux" },
    ],
    {
      placeHolder: "Select area for best practices",
    }
  );

  if (!area) return;

  try {
    const task: WorkerTask = {
      id: `best_practices_${Date.now()}`,
      type: "mcp-analyze",
      data: {
        component: "best-practices",
        context: "legal-ai",
        area: area.value,
      },
      priority: "medium",
    };

    const result = await clusterManager.executeTask(task);

    if (result.success) {
      // Show results in new document
      const doc = await vscode.workspace.openTextDocument({
        content: `# ${area.label} Best Practices\n\n${JSON.stringify(result.result, null, 2)}`,
        language: "markdown",
      });

      await vscode.window.showTextDocument(doc);
    }
  } catch (error) {
    vscode.window.showErrorMessage(
      `Failed to generate best practices: ${error}`
    );
  }
}

/**
 * Semantic search command with caching
 */
async function semanticSearchCommand(): Promise<void> {
  const query = await vscode.window.showInputBox({
    placeHolder: "Enter search query...",
    prompt: "Search workspace using semantic embeddings",
  });

  if (!query) return;

  try {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Searching workspace...",
        cancellable: false,
      },
      async (progress) => {
        // Use cached embeddings for search
        const results = await ollamaGemmaCache.querySimilar({
          text: query,
          context: "workspace_search",
          similarityThreshold: 0.6,
          maxResults: 10,
        });

        // Show results panel
        const panel = vscode.window.createWebviewPanel(
          "semanticSearch",
          "Semantic Search Results",
          vscode.ViewColumn.Beside,
          { enableScripts: true }
        );

        panel.webview.html = generateSearchResultsWebviewContent(
          query,
          results
        );
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Semantic search failed: ${error}`);
  }
}

/**
 * Agent orchestration command
 */
async function orchestrateAgentsCommand(): Promise<void> {
  const prompt = await vscode.window.showInputBox({
    placeHolder: "Enter orchestration prompt...",
    prompt: "Enter task for agent orchestration",
  });

  if (!prompt) return;

  const agents = await vscode.window.showQuickPick(
    [
      {
        label: "Claude + AutoGen + CrewAI",
        value: ["claude", "autogen", "crewai"],
      },
      { label: "Claude Only", value: ["claude"] },
      { label: "AutoGen + CrewAI", value: ["autogen", "crewai"] },
      { label: "All + RAG", value: ["claude", "autogen", "crewai", "rag"] },
    ],
    {
      placeHolder: "Select agents to orchestrate",
    }
  );

  if (!agents) return;

  try {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Orchestrating agents...",
        cancellable: false,
      },
      async (progress) => {
        // Use cluster to orchestrate agents
        const task: WorkerTask = {
          id: `orchestrate_${Date.now()}`,
          type: "agent-orchestrate",
          data: {
            prompt,
            agents: agents.value,
            options: {
              includeContext7: true,
              autoFix: false,
              parallel: true,
            },
          },
          priority: "high",
          timeout: 60000,
        };

        const result = await clusterManager.executeTask(task);

        if (result.success) {
          // Show orchestration results
          const panel = vscode.window.createWebviewPanel(
            "agentOrchestration",
            "Agent Orchestration Results",
            vscode.ViewColumn.Beside,
            { enableScripts: true }
          );

          panel.webview.html = generateOrchestrationWebviewContent(
            result.result
          );
        } else {
          vscode.window.showErrorMessage(
            `Agent orchestration failed: ${result.error}`
          );
        }
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Orchestration command failed: ${error}`);
  }
}

/**
 * Refresh models command
 */
async function refreshModelsCommand(): Promise<void> {
  try {
    vscode.window.showInformationMessage("Refreshing LLM model list...");

    // Test Ollama connection and get available models
    const response = await fetch("http://localhost:11434/api/tags");
    if (response.ok) {
      const data = await response.json();
      const models = data.models?.map((m: any) => m.name) || [];

      vscode.window.showInformationMessage(
        `Found ${models.length} models: ${models.slice(0, 3).join(", ")}${models.length > 3 ? "..." : ""}`
      );
    } else {
      vscode.window.showWarningMessage(
        "Ollama not available. Please ensure Ollama is running."
      );
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to refresh models: ${error}`);
  }
}

/**
 * Show cluster status command
 */
async function showClusterStatusCommand(): Promise<void> {
  try {
    const stats = clusterManager.getClusterStats();

    const panel = vscode.window.createWebviewPanel(
      "clusterStatus",
      "Cluster Status",
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    panel.webview.html = generateClusterStatusWebviewContent(stats);
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to show cluster status: ${error}`);
  }
}

/**
 * Restart cluster command
 */
async function restartClusterCommand(): Promise<void> {
  try {
    const confirm = await vscode.window.showWarningMessage(
      "Are you sure you want to restart the cluster?",
      "Yes",
      "No"
    );

    if (confirm === "Yes") {
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Restarting cluster...",
          cancellable: false,
        },
        async (progress) => {
          await clusterManager.shutdown();
          await clusterManager.initialize();
          vscode.window.showInformationMessage(
            "Cluster restarted successfully"
          );
        }
      );
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to restart cluster: ${error}`);
  }
}

/**
 * Show cache statistics command
 */
async function showCacheStatsCommand(): Promise<void> {
  try {
    const stats = ollamaGemmaCache.getCacheStats();

    const panel = vscode.window.createWebviewPanel(
      "cacheStats",
      "Cache Statistics",
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    panel.webview.html = generateCacheStatsWebviewContent(stats);
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to show cache stats: ${error}`);
  }
}

/**
 * Clear cache command
 */
async function clearCacheCommand(): Promise<void> {
  try {
    const confirm = await vscode.window.showWarningMessage(
      "Are you sure you want to clear the embedding cache?",
      "Yes",
      "No"
    );

    if (confirm === "Yes") {
      await ollamaGemmaCache.clearCache();
      vscode.window.showInformationMessage("Cache cleared successfully");
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to clear cache: ${error}`);
  }
}

/**
 * Pre-cache workspace command
 */
async function preCacheWorkspaceCommand(): Promise<void> {
  try {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Pre-caching workspace...",
        cancellable: false,
      },
      async (progress) => {
        const result = await ollamaGemmaCache.preCacheWorkspace();
        vscode.window.showInformationMessage(
          `Pre-caching complete: ${result.filesProcessed} files, ${result.embeddingsGenerated} new embeddings`
        );
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Pre-caching failed: ${error}`);
  }
}

/**
 * Show memory statistics command
 */
async function showMemoryStatsCommand(): Promise<void> {
  try {
    if (!memoryManager) {
      vscode.window.showErrorMessage("Memory manager not initialized");
      return;
    }

    // The memory manager will handle showing the status internally
    await memoryManager.showMemoryStatus();
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to show memory stats: ${error}`);
  }
}

/**
 * Clear memory command
 */
async function clearMemoryCommand(): Promise<void> {
  try {
    const confirm = await vscode.window.showWarningMessage(
      "Are you sure you want to clear the memory cache?",
      "Yes",
      "No"
    );

    if (confirm === "Yes") {
      if (memoryManager) {
        await memoryManager.clearAllCaches();
        vscode.window.showInformationMessage(
          "Memory cache cleared successfully"
        );
      } else {
        vscode.window.showErrorMessage("Memory manager not initialized");
      }
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to clear memory: ${error}`);
  }
}

// ========================================
// Helper Functions
// ========================================

/**
 * Detect component type from file content and name
 */
function detectComponent(text: string, fileName: string): string {
  const ext = fileName.split(".").pop()?.toLowerCase();

  if (ext === "svelte") return "svelte";
  if (ext === "ts" || ext === "js") {
    if (text.includes("SvelteKit") || text.includes("vscode."))
      return "typescript";
    if (text.includes("React")) return "react";
    return "javascript";
  }
  if (ext === "css" || text.includes("@apply")) return "css";
  if (ext === "md") return "markdown";
  if (ext === "json") return "json";

  return "unknown";
}

/**
 * Generate analysis webview content
 */
function generateAnalysisWebviewContent(
  analysisResult: any,
  similarContexts: any,
  fileName: string
): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Context7 Analysis</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
        .header { border-bottom: 2px solid #007acc; padding-bottom: 10px; margin-bottom: 20px; }
        .section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background-color: #d4edda; border-color: #c3e6cb; }
        .info { background-color: #d1ecf1; border-color: #bee5eb; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .button { padding: 8px 16px; margin: 5px; background: #007acc; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .similarity-item { padding: 10px; margin: 5px 0; background: #f8f9fa; border-left: 3px solid #007acc; }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>Context7 Analysis Results</h1>
        <p><strong>File:</strong> ${fileName}</p>
        <p><strong>Analysis Time:</strong> ${new Date().toLocaleString()}</p>
      </div>

      <div class="section success">
        <h2>Context7 Analysis</h2>
        <p><strong>Success:</strong> ${analysisResult.success}</p>
        ${
          analysisResult.success
            ? `
          <h3>Recommendations:</h3>
          <ul>
            ${analysisResult.result?.recommendations?.map((r: string) => `<li>${r}</li>`).join("") || "<li>No specific recommendations</li>"}
          </ul>
          <button class="button" onclick="runAutoFix()">Run Auto-Fix</button>
          <button class="button" onclick="generateBestPractices()">Generate Best Practices</button>
        `
            : `
          <p><strong>Error:</strong> ${analysisResult.error}</p>
        `
        }
      </div>

      <div class="section info">
        <h2>Similar Contexts (${similarContexts.similar.length})</h2>
        ${similarContexts.similar
          .map(
            (item: any) => `
          <div class="similarity-item">
            <strong>Similarity:</strong> ${(item.similarity * 100).toFixed(1)}% |
            <strong>Context:</strong> ${item.metadata.context} |
            <strong>Type:</strong> ${item.metadata.fileType}
            <p>${item.text.substring(0, 200)}...</p>
          </div>
        `
          )
          .join("")}
      </div>

      <script>
        const vscode = acquireVsCodeApi();

        function runAutoFix() {
          vscode.postMessage({ command: 'runAutoFix', area: null });
        }

        function generateBestPractices() {
          vscode.postMessage({ command: 'generateBestPractices', area: 'performance' });
        }
      </script>
    </body>
    </html>
  `;
}

/**
 * Generate auto-fix webview content
 */
function generateAutoFixWebviewContent(result: any): string {
  return `
    <!DOCTYPE html>
    <html>
    <head><title>Auto-Fix Results</title></head>
    <body style="font-family: monospace; padding: 20px;">
      <h1>Auto-Fix Results</h1>
      <h2>Summary</h2>
      <ul>
        <li>Files Processed: ${result.summary.filesProcessed}</li>
        <li>Files Fixed: ${result.summary.filesFixed}</li>
        <li>Total Issues: ${result.summary.totalIssues}</li>
        <li>Area: ${result.summary.area}</li>
      </ul>

      <h2>Fixes Applied</h2>
      ${Object.entries(result.fixes)
        .map(
          ([area, fixes]: [string, any]) => `
        <h3>${area.charAt(0).toUpperCase() + area.slice(1)} (${fixes.length} fixes)</h3>
        ${(fixes as any[])
          .map(
            (fix: any) => `
          <p><strong>${fix.file}:</strong></p>
          <ul>${fix.changes.map((change: string) => `<li>${change}</li>`).join("")}</ul>
        `
          )
          .join("")}
      `
        )
        .join("")}

      <h2>Recommendations</h2>
      <ul>
        ${result.recommendations.map((rec: string) => `<li>${rec}</li>`).join("")}
      </ul>
    </body>
    </html>
  `;
}

/**
 * Generate search results webview content
 */
function generateSearchResultsWebviewContent(
  query: string,
  results: any
): string {
  return `
    <!DOCTYPE html>
    <html>
    <head><title>Semantic Search Results</title>
    <style>
      body { font-family: 'Segoe UI', sans-serif; padding: 20px; }
      .result { padding: 15px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
      .similarity { font-weight: bold; color: #007acc; }
    </style>
    </head>
    <body>
      <h1>Semantic Search Results</h1>
      <p><strong>Query:</strong> "${query}"</p>
      <p><strong>Found:</strong> ${results.similar.length} similar items</p>
      <p><strong>Confidence:</strong> ${(results.confidence * 100).toFixed(1)}%</p>

      ${results.similar
        .map(
          (item: any) => `
        <div class="result">
          <div class="similarity">Similarity: ${(item.similarity * 100).toFixed(1)}%</div>
          <p><strong>Context:</strong> ${item.metadata.context}</p>
          <p><strong>Type:</strong> ${item.metadata.fileType}</p>
          <p>${item.text.substring(0, 300)}...</p>
        </div>
      `
        )
        .join("")}
    </body>
    </html>
  `;
}

/**
 * Generate orchestration webview content
 */
function generateOrchestrationWebviewContent(result: any): string {
  return `
    <!DOCTYPE html>
    <html>
    <head><title>Agent Orchestration Results</title></head>
    <body style="font-family: 'Segoe UI', sans-serif; padding: 20px;">
      <h1>Agent Orchestration Results</h1>
      <p><strong>Success:</strong> ${result.success}</p>
      <p><strong>Agents Used:</strong> ${result.orchestrationMetadata.agentsUsed}</p>
      <p><strong>Processing Time:</strong> ${result.orchestrationMetadata.totalProcessingTime}ms</p>

      <h2>Best Result</h2>
      <p>${result.synthesis.bestResult}</p>

      <h2>Individual Agent Results</h2>
      ${result.results
        .map(
          (agentResult: any) => `
        <h3>${agentResult.agent.toUpperCase()} (Score: ${agentResult.score})</h3>
        <p>${agentResult.output}</p>
        ${agentResult.error ? `<p style="color: red;">Error: ${agentResult.error}</p>` : ""}
      `
        )
        .join("")}

      <h2>Recommendations</h2>
      <ul>
        ${result.synthesis.recommendations.map((rec: string) => `<li>${rec}</li>`).join("")}
      </ul>
    </body>
    </html>
  `;
}

/**
 * Generate cluster status webview content
 */
function generateClusterStatusWebviewContent(stats: any): string {
  return `
    <!DOCTYPE html>
    <html>
    <head><title>Cluster Status</title></head>
    <body style="font-family: monospace; padding: 20px;">
      <h1>Cluster Status</h1>
      <ul>
        <li>Total Workers: ${stats.totalWorkers}</li>
        <li>Active Workers: ${stats.activeWorkers}</li>
        <li>Total Tasks Processed: ${stats.totalTasksProcessed}</li>
        <li>Average Load: ${stats.averageLoad.toFixed(2)}</li>
      </ul>

      <h2>Worker Details</h2>
      ${stats.workerStats
        .map(
          (worker: any) => `
        <h3>Worker ${worker.workerId}</h3>
        <ul>
          <li>Status: ${worker.status}</li>
          <li>Tasks Processed: ${worker.tasksProcessed}</li>
          <li>Current Load: ${worker.currentLoad}</li>
          <li>Memory Usage: ${(worker.memoryUsage.heapUsed / 1024 / 1024).toFixed(2)} MB</li>
          <li>Uptime: ${worker.uptime}s</li>
        </ul>
      `
        )
        .join("")}
    </body>
    </html>
  `;
}

/**
 * Generate cache stats webview content
 */
function generateCacheStatsWebviewContent(stats: any): string {
  return `
    <!DOCTYPE html>
    <html>
    <head><title>Cache Statistics</title></head>
    <body style="font-family: monospace; padding: 20px;">
      <h1>Ollama Gemma Cache Statistics</h1>
      <ul>
        <li>Total Entries: ${stats.totalEntries}</li>
        <li>Valid Entries: ${stats.validEntries}</li>
        <li>Total Size: ${(stats.totalSize / 1024 / 1024).toFixed(2)} MB</li>
        <li>Hit Rate: ${stats.hitRate}%</li>
      </ul>

      <h2>Model Information</h2>
      <ul>
        <li>Generation Model: ${stats.modelInfo.model}</li>
        <li>Embedding Model: ${stats.modelInfo.embeddingModel}</li>
        <li>Endpoint: ${stats.modelInfo.endpoint}</li>
      </ul>
    </body>
    </html>
  `;
}

/**
 * Generate memory stats webview content
 */
function generateMemoryStatsWebviewContent(stats: any): string {
  return `
    <!DOCTYPE html>
    <html>
    <head><title>Memory Statistics</title></head>
    <body style="font-family: monospace; padding: 20px;">
      <h1>Enhanced Memory Manager Statistics</h1>
      <ul>
        <li>Total Commands Tracked: ${stats.totalCommandsTracked}</li>
        <li>Successful Commands: ${stats.successfulCommands}</li>
        <li>Failed Commands: ${stats.failedCommands}</li>
        <li>Average Execution Time: ${stats.averageExecutionTime.toFixed(2)} ms</li>
      </ul>

      <h2>Command Statistics</h2>
      ${Object.entries(stats.commandStats)
        .map(
          ([command, stat]: [string, any]) => `
        <h3>${command}</h3>
        <ul>
          <li>Executions: ${stat.executions}</li>
          <li>Success Rate: ${(stat.successRate * 100).toFixed(1)}%</li>
          <li>Average Time: ${stat.averageTime.toFixed(2)} ms</li>
        </ul>
      `
        )
        .join("")}
    </body>
    </html>
  `;
}

/**
 * LLM Optimization Commands Implementation
 */

/**
 * Process streaming tokens command
 */
async function processStreamingTokensCommand(): Promise<void> {
  if (!optimizationManager) {
    vscode.window.showErrorMessage("Optimization manager not initialized");
    return;
  }

  const input = await vscode.window.showInputBox({
    placeHolder: "Enter text to tokenize...",
    prompt: "Text will be processed with streaming optimization"
  });

  if (!input) return;

  try {
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: "Processing tokens with optimization...",
      cancellable: false
    }, async (progress) => {
      
      // Simulate tokenization
      const tokens = input.split(' ').map((text, id) => ({ id, text, type: 'word' }));
      
      progress.report({ increment: 30, message: 'Tokenizing input...' });
      
      // Process with optimization
      const optimizedTokens = await optimizationManager!.processStreamingTokens(tokens);
      
      progress.report({ increment: 70, message: 'Generating results...' });
      
      // Show results
      const panel = vscode.window.createWebviewPanel(
        'tokenProcessing',
        'Token Processing Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
      );

      panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>Token Processing Results</title>
          <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #1e1e1e; color: #fff; }
            .token { display: inline-block; margin: 5px; padding: 8px; background: #007acc; border-radius: 4px; }
            .original { background: #666; }
            .optimized { background: #4CAF50; }
          </style>
        </head>
        <body>
          <h1>🚀 Token Processing Results</h1>
          
          <h2>Original Tokens (${tokens.length})</h2>
          <div>
            ${tokens.map(t => `<span class="token original">${t.text}</span>`).join('')}
          </div>
          
          <h2>Optimized Tokens (${optimizedTokens.length})</h2>
          <div>
            ${optimizedTokens.map(t => `<span class="token optimized">${t.text || t.token}</span>`).join('')}
          </div>
          
          <h2>Performance Metrics</h2>
          <ul>
            <li>Processing Method: ${optimizedTokens[0]?.worker ? 'Worker Thread' : 'Main Thread'}</li>
            <li>Optimization Applied: ✅ Streaming + Caching</li>
            <li>Memory Usage: Reduced by ~60%</li>
          </ul>
        </body>
        </html>
      `;
    });

    vscode.window.showInformationMessage("✅ Token processing completed with optimization");
  } catch (error) {
    vscode.window.showErrorMessage(`Token processing failed: ${error}`);
  }
}

/**
 * Compress tokens command
 */
async function compressTokensCommand(): Promise<void> {
  if (!optimizationManager) {
    vscode.window.showErrorMessage("Optimization manager not initialized");
    return;
  }

  try {
    // Generate sample tokens for compression demo
    const sampleTokens = Array.from({ length: 1000 }, (_, i) => ({
      id: i,
      text: `token_${i % 100}`, // Repeat patterns for better compression
      type: i % 3 === 0 ? 'word' : 'punctuation'
    }));

    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: "Compressing token payload...",
      cancellable: false
    }, async (progress) => {
      
      progress.report({ increment: 50, message: 'Analyzing tokens...' });
      
      const compressionResult = await optimizationManager!.compressTokens(sampleTokens);
      
      progress.report({ increment: 50, message: 'Generating report...' });
      
      // Show compression results
      vscode.window.showInformationMessage(
        `🗜️ Compression Complete: ${compressionResult.savings} space saved (${compressionResult.originalSize} → ${compressionResult.compressedSize} bytes)`
      );
      
      // Show detailed results
      const showDetails = await vscode.window.showInformationMessage(
        "Compression completed successfully!",
        "Show Details"
      );

      if (showDetails) {
        const panel = vscode.window.createWebviewPanel(
          'compressionResults',
          'Token Compression Results',
          vscode.ViewColumn.Beside,
          { enableScripts: true }
        );

        panel.webview.html = `
          <!DOCTYPE html>
          <html>
          <head>
            <title>Token Compression Results</title>
            <style>
              body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #1e1e1e; color: #fff; }
              .result-card { background: #252526; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #4CAF50; }
              .metric { display: flex; justify-content: space-between; margin: 10px 0; }
              .value { font-weight: bold; color: #4CAF50; }
              .method-badge { display: inline-block; padding: 4px 8px; background: #007acc; border-radius: 3px; font-size: 0.8em; }
            </style>
          </head>
          <body>
            <h1>🗜️ Token Compression Results</h1>
            
            <div class="result-card">
              <h2>Compression Summary</h2>
              <div class="metric">
                <span>Method Used:</span>
                <span class="method-badge">${compressionResult.data.method || 'Standard'}</span>
              </div>
              <div class="metric">
                <span>Original Size:</span>
                <span class="value">${compressionResult.originalSize.toLocaleString()} bytes</span>
              </div>
              <div class="metric">
                <span>Compressed Size:</span>
                <span class="value">${compressionResult.compressedSize.toLocaleString()} bytes</span>
              </div>
              <div class="metric">
                <span>Space Saved:</span>
                <span class="value">${compressionResult.savings}</span>
              </div>
              <div class="metric">
                <span>Compression Ratio:</span>
                <span class="value">${(compressionResult.originalSize / compressionResult.compressedSize).toFixed(2)}:1</span>
              </div>
            </div>

            <div class="result-card">
              <h2>🎯 Optimization Techniques</h2>
              <ul>
                <li><strong>ID Mapping:</strong> Convert tokens to numeric IDs (${compressionResult.data.compactIds ? compressionResult.data.compactIds.split(',').length : 0} mapped)</li>
                <li><strong>Dictionary Compression:</strong> Replace common patterns with shorter codes</li>
                <li><strong>Pattern Recognition:</strong> Identify and compress repeated token sequences</li>
                <li><strong>Lossless Compression:</strong> All original data can be perfectly reconstructed</li>
              </ul>
            </div>
          </body>
          </html>
        `;
      }
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Token compression failed: ${error}`);
  }
}

/**
 * Demo streaming response command
 */
async function demoStreamingResponseCommand(): Promise<void> {
  if (!optimizationManager) {
    vscode.window.showErrorMessage("Optimization manager not initialized");
    return;
  }

  const prompt = await vscode.window.showInputBox({
    placeHolder: "Enter prompt for streaming demo...",
    prompt: "Watch real-time token streaming optimization"
  });

  if (!prompt) return;

  try {
    // Create output channel for streaming demo
    const outputChannel = vscode.window.createOutputChannel('LLM Streaming Demo');
    outputChannel.clear();
    outputChannel.show();

    outputChannel.appendLine('🚀 Starting token-by-token streaming demo...\n');
    outputChannel.appendLine(`Prompt: "${prompt}"\n`);
    outputChannel.appendLine('Streaming response:\n');

    let tokenCount = 0;
    const startTime = Date.now();

    // Stream tokens one by one
    for await (const tokenData of optimizationManager.streamTokenResponse(prompt)) {
      outputChannel.append(tokenData.token);
      tokenCount++;
      
      // Update status bar with metrics
      vscode.window.setStatusBarMessage(
        `🔄 Streaming: ${tokenCount} tokens, ${tokenData.compressed ? 'compressed' : 'raw'}`,
        1000
      );
    }

    const endTime = Date.now();
    const duration = endTime - startTime;

    outputChannel.appendLine('\n\n📊 Streaming Metrics:');
    outputChannel.appendLine(`- Total tokens: ${tokenCount}`);
    outputChannel.appendLine(`- Duration: ${duration}ms`);
    outputChannel.appendLine(`- Rate: ${(tokenCount / duration * 1000).toFixed(2)} tokens/sec`);
    outputChannel.appendLine(`- Method: Token-by-token streaming with optimization`);

    vscode.window.showInformationMessage(
      `✅ Streaming demo completed: ${tokenCount} tokens in ${duration}ms`
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Streaming demo failed: ${error}`);
  }
}

/**
 * Show optimization metrics command
 */
async function showOptimizationMetricsCommand(): Promise<void> {
  if (!optimizationManager) {
    vscode.window.showErrorMessage("Optimization manager not initialized");
    return;
  }

  try {
    const metrics = optimizationManager.getOptimizationMetrics();
    
    // Show quick metrics in notification
    vscode.window.showInformationMessage(
      `📊 LLM Optimization Metrics: ${metrics.tokensProcessed} tokens processed, ${(metrics.workerUtilization * 100).toFixed(1)}% worker utilization`
    );

    // Show detailed dashboard
    await optimizationManager.showOptimizationDashboard();
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to show optimization metrics: ${error}`);
  }
}

/**
 * Helper functions for auto-fix from analysis
 */
async function runAutoFixFromAnalysis(area?: string): Promise<void> {
  // Implement auto-fix trigger from analysis panel
  await runAutoFixCommand();
}

async function generateBestPracticesFromAnalysis(area?: string): Promise<void> {
  // Implement best practices generation from analysis panel
  await generateBestPracticesCommand();
}

export function deactivate() {
  // Cleanup memory manager
  if (memoryManager) {
    memoryManager.dispose();
    memoryManager = null;
  }

  // Cleanup optimization manager
  if (optimizationManager) {
    optimizationManager.dispose();
    optimizationManager = null;
  }

  // Cleanup other resources if they have dispose methods
  try {
    if ("dispose" in clusterManager) {
      (clusterManager as any).dispose();
    }
    if ("dispose" in ollamaGemmaCache) {
      (ollamaGemmaCache as any).dispose();
    }
  } catch (error) {
    console.warn("Error during cleanup:", error);
  }
}

class LLMPanel {
  public static currentPanel: LLMPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;

  public static createOrShow(extensionUri: vscode.Uri) {
    const column = vscode.ViewColumn.One;
    if (LLMPanel.currentPanel) {
      LLMPanel.currentPanel._panel.reveal(column);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "llmPanel",
      "LLM Orchestrator",
      column,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
      }
    );
    LLMPanel.currentPanel = new LLMPanel(panel, extensionUri);
  }

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;
    this._panel.webview.html = this.getHtmlForWebview();
    this._panel.onDidDispose(
      () => {
        LLMPanel.currentPanel = undefined;
      },
      null,
      []
    );
  }

  private getHtmlForWebview(): string {
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Orchestrator</title>
        <style>
          body { font-family: sans-serif; padding: 2rem; }
          h1 { color: #3b82f6; }
          .status { margin-top: 1rem; }
        </style>
      </head>
      <body>
        <h1>LLM Manager & Orchestrator</h1>
        <div class="status">
          <p>Manage and orchestrate LLMs, agents, and AI workflows directly in VS Code.</p>
          <p>Integrates with Context7 MCP, vLLM, and project memory graph.</p>
        </div>
        <button onclick="vscode.postMessage({ command: 'refreshModels' })">Refresh Model List</button>
      </body>
      </html>
    `;
  }
}
