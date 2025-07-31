"use strict";
var __createBinding =
  (this && this.__createBinding) ||
  (Object.create
    ? function (o, m, k, k2) {
        if (k2 === undefined) k2 = k;
        var desc = Object.getOwnPropertyDescriptor(m, k);
        if (
          !desc ||
          ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)
        ) {
          desc = {
            enumerable: true,
            get: function () {
              return m[k];
            },
          };
        }
        Object.defineProperty(o, k2, desc);
      }
    : function (o, m, k, k2) {
        if (k2 === undefined) k2 = k;
        o[k2] = m[k];
      });
var __setModuleDefault =
  (this && this.__setModuleDefault) ||
  (Object.create
    ? function (o, v) {
        Object.defineProperty(o, "default", { enumerable: true, value: v });
      }
    : function (o, v) {
        o["default"] = v;
      });
var __importStar =
  (this && this.__importStar) ||
  function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null)
      for (var k in mod)
        if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k))
          __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
  };
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const mcpServerManager_1 = require("./mcpServerManager");
const contextAnalyzer_1 = require("./contextAnalyzer");
const statusBarManager_1 = require("./statusBarManager");
const diagnosticWatcher_1 = require("./diagnosticWatcher");
const stackAnalyzer_1 = require("./stackAnalyzer");
let mcpServerManager;
let contextAnalyzer;
let statusBarManager;
let diagnosticWatcher;
let stackAnalyzer;
function activate(context) {
  console.log("🚀 Context7 MCP Assistant extension is now active!");
  // Initialize core components
  const workspaceRoot =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || "";
  mcpServerManager = new mcpServerManager_1.MCPServerManager(context);
  contextAnalyzer = new contextAnalyzer_1.ContextAnalyzer();
  statusBarManager = new statusBarManager_1.StatusBarManager();
  diagnosticWatcher = new diagnosticWatcher_1.DiagnosticWatcher();
  stackAnalyzer = new stackAnalyzer_1.StackAnalyzer(workspaceRoot);
  // Register commands
  registerCommands(context);
  // Setup event listeners
  setupEventListeners(context);
  // Auto-start server if enabled
  const config = vscode.workspace.getConfiguration("mcpContext7");
  if (config.get("autoStart", true)) {
    mcpServerManager.startServer();
  }
  // Register local agent orchestrator (for multi-agent workflows)
  if (vscode.lm && vscode.lm.registerMcpServerDefinitionProvider) {
    vscode.lm.registerMcpServerDefinitionProvider({
      id: "local-agent-orchestrator",
      label: "Local Agent Orchestrator",
      description:
        "Multi-agent orchestration (Claude, CrewAI, AutoGen, RAG, etc.)",
      start: async () => {
        // Start orchestrator process (if not already running)
        const orchestratorPath = vscode.Uri.joinPath(
          context.extensionPath,
          "../../agent-orchestrator/index.js"
        ).fsPath;
        const cp = require("child_process");
        const proc = cp.spawn("node", [orchestratorPath], {
          detached: true,
          stdio: "ignore",
        });
        proc.unref();
        vscode.window.showInformationMessage(
          "Local Agent Orchestrator started."
        );
      },
      stop: async () => {
        // Optionally: implement orchestrator shutdown logic
        vscode.window.showInformationMessage(
          "Local Agent Orchestrator stopped."
        );
      },
      health: async () => {
        // Check orchestrator health endpoint
        const fetch = require("node-fetch");
        try {
          const res = await fetch("http://localhost:7070/api/agent-health");
          if (res.ok) return { status: "ok" };
        } catch (e) {}
        return { status: "unavailable" };
      },
    });
  }
  // Update status bar
  statusBarManager.updateStatus("ready", "Context7 MCP Ready");
  // Run Agent Orchestrator (Claude Code CLI) with approval prompt
  const runAgentOrchestratorCommand = vscode.commands.registerCommand(
    "mcp.runAgentOrchestrator",
    async () => {
      const approval = await vscode.window.showInformationMessage(
        "Do you want to run the multi-agent orchestrator (Claude, CrewAI, AutoGen, etc.)?",
        { modal: true },
        "Approve",
        "Cancel"
      );
      if (approval === "Approve") {
        // Call orchestrator endpoint (or start orchestrator if needed)
        const fetch = require("node-fetch");
        try {
          const res = await fetch(
            "http://localhost:7070/api/agent-orchestrate",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                prompt: "Run multi-agent workflow",
                context: {},
                options: {},
              }),
            }
          );
          const data = await res.json();
          vscode.window.showInformationMessage(
            "Agent orchestrator result: " +
              JSON.stringify(data.ranked[0] || data.all[0])
          );
        } catch (e) {
          vscode.window.showErrorMessage(
            "Failed to call agent orchestrator: " + e.message
          );
        }
      }
    }
  );
}
exports.activate = activate;
function deactivate() {
  console.log("🛑 Context7 MCP Assistant extension is deactivating...");
  if (mcpServerManager) {
    mcpServerManager.stopServer();
  }
  if (statusBarManager) {
    statusBarManager.dispose();
  }
  if (diagnosticWatcher) {
    diagnosticWatcher.dispose();
  }
}
exports.deactivate = deactivate;
function registerCommands(context) {
    // ...existing commands...

    // Enhanced RAG & Multi-Agent Commands
    context.subscriptions.push(vscode.commands.registerCommand('mcp.runAgentOrchestrator', async () => {
        try {
            const result = await vscode.window.showInformationMessage(
                'Run multi-agent orchestration (Claude, CrewAI, AutoGen)?',
                'Yes', 'No'
            );

            if (result === 'Yes') {
                vscode.window.showInformationMessage('🤖 Starting multi-agent orchestration...');

                // Call the agent orchestrator
                const response = await fetch('http://localhost:7070/api/agent-orchestrate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task: 'analyze_current_context',
                        context: await buildVSCodeContext()
                    })
                });

                const data = await response.json();
                await showResultPanel('Agent Orchestration Results', data);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Agent orchestration failed: ${error.message}`);
        }
    }));

    context.subscriptions.push(vscode.commands.registerCommand('mcp.openRAGStudio', async () => {
        try {
            // Open Enhanced RAG Studio in browser
            const panel = vscode.window.createWebviewPanel(
                'ragStudio',
                'EnhancedRAG Studio',
                vscode.ViewColumn.One,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true
                }
            );

            panel.webview.html = `
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body { margin: 0; padding: 0; }
                        iframe { width: 100%; height: 100vh; border: none; }
                    </style>
                </head>
                <body>
                    <iframe src="http://localhost:5173/rag-studio"></iframe>
                </body>
                </html>
            `;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to open RAG Studio: ${error.message}`);
        }
    }));

    context.subscriptions.push(vscode.commands.registerCommand('mcp.generateBestPractices', async () => {
        try {
            vscode.window.showInformationMessage('📋 Generating best practices report...');

            const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
            if (!workspaceRoot) {
                vscode.window.showErrorMessage('No workspace folder found');
                return;
            }

            const response = await fetch('http://localhost:5173/api/best-practices', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    projectPath: workspaceRoot
                })
            });

            const report = await response.json();
            await showResultPanel('Best Practices Report', report);

        } catch (error) {
            vscode.window.showErrorMessage(`Best practices generation failed: ${error.message}`);
        }
    }));

    context.subscriptions.push(vscode.commands.registerCommand('mcp.uploadDocument', async () => {
        try {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectFiles: true,
                canSelectFolders: false,
                canSelectMany: false,
                filters: {
                    'PDF Files': ['pdf'],
                    'Text Files': ['txt', 'md']
                }
            });

            if (fileUri && fileUri[0]) {
                vscode.window.showInformationMessage('📄 Uploading document to knowledge base...');

                const formData = new FormData();
                formData.append('file', await vscode.workspace.fs.readFile(fileUri[0]));

                const response = await fetch('http://localhost:5173/api/rag?action=upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                vscode.window.showInformationMessage(`✅ Document uploaded: ${result.document.chunks} chunks processed`);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Document upload failed: ${error.message}`);
        }
    }));

    context.subscriptions.push(vscode.commands.registerCommand('mcp.crawlWebsite', async () => {
        try {
            const url = await vscode.window.showInputBox({
                prompt: 'Enter website URL to crawl',
                placeholder: 'https://example.com'
            });

            if (url) {
                vscode.window.showInformationMessage('🌐 Crawling website...');

                const response = await fetch('http://localhost:5173/api/rag?action=crawl', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });

                const result = await response.json();
                vscode.window.showInformationMessage(`✅ Website crawled: ${result.document.chunks} chunks processed`);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Website crawling failed: ${error.message}`);
        }
    }));

    // ...existing commands...

  // Steps 6-10: New Enhanced Features Commands

  // Step 6: Library Sync Commands
  context.subscriptions.push(vscode.commands.registerCommand('mcp.syncLibraries', async () => {
    try {
      vscode.window.showInformationMessage('🔄 Syncing library metadata...');

      const response = await fetch('http://localhost:5173/api/libraries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: null }) // Sync all sources
      });

      const data = await response.json();
      if (data.success) {
        vscode.window.showInformationMessage('✅ Library metadata synced successfully');
      } else {
        vscode.window.showErrorMessage(`Sync failed: ${data.error}`);
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Library sync failed: ${error.message}`);
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('mcp.viewAgentLogs', async () => {
    try {
      const response = await fetch('http://localhost:5173/api/agent-logs?limit=50');
      const data = await response.json();

      if (data.success) {
        await showResultPanel('Agent Call Logs', data.logs);
      } else {
        vscode.window.showErrorMessage(`Failed to get logs: ${data.error}`);
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to view agent logs: ${error.message}`);
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('mcp.searchLibraries', async () => {
    try {
      const query = await vscode.window.showInputBox({
        prompt: 'Enter library search query',
        placeHolder: 'e.g., "svelte", "typescript", "mongodb"'
      });

      if (query) {
        const response = await fetch(`http://localhost:5173/api/libraries?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.success) {
          await showResultPanel(`Library Search Results: ${query}`, data.libraries);
        } else {
          vscode.window.showErrorMessage(`Search failed: ${data.error}`);
        }
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Library search failed: ${error.message}`);
    }
  }));

  // Step 9: Multi-Agent Orchestration Commands
  context.subscriptions.push(vscode.commands.registerCommand('mcp.createWorkflow', async () => {
    try {
      const name = await vscode.window.showInputBox({
        prompt: 'Enter workflow name',
        placeHolder: 'e.g., "Code Analysis Workflow"'
      });

      if (!name) return;

      const capabilities = await vscode.window.showQuickPick(
        [
          'vector_search',
          'code_analysis',
          'best_practices_generation',
          'doc_generation',
          'quality_check'
        ],
        {
          canPickMany: true,
          placeHolder: 'Select required capabilities'
        }
      );

      if (capabilities && capabilities.length > 0) {
        const query = await vscode.window.showInputBox({
          prompt: 'Enter workflow query/task',
          placeHolder: 'What should this workflow accomplish?'
        });

        if (query) {
          const response = await fetch('http://localhost:5173/api/orchestrator', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'create_and_execute',
              name,
              capabilities,
              query,
              context: await buildVSCodeContext()
            })
          });

          const data = await response.json();
          if (data.success) {
            vscode.window.showInformationMessage(`✅ Workflow executed: ${data.workflowId}`);
            await showResultPanel(`Workflow Results: ${name}`, data.result);
          } else {
            vscode.window.showErrorMessage(`Workflow failed: ${data.error}`);
          }
        }
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Workflow creation failed: ${error.message}`);
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('mcp.viewWorkflows', async () => {
    try {
      const response = await fetch('http://localhost:5173/api/orchestrator');
      const data = await response.json();

      if (data.success) {
        await showResultPanel('Active Workflows', data.workflows);
      } else {
        vscode.window.showErrorMessage(`Failed to get workflows: ${data.error}`);
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to view workflows: ${error.message}`);
    }
  }));

  // Step 10: Evaluation & Metrics Commands
  context.subscriptions.push(vscode.commands.registerCommand('mcp.recordFeedback', async () => {
    try {
      const rating = await vscode.window.showQuickPick(
        ['1 - Poor', '2 - Fair', '3 - Good', '4 - Very Good', '5 - Excellent'],
        { placeHolder: 'Rate the last AI response' }
      );

      if (rating) {
        const ratingValue = parseInt(rating.charAt(0));
        const feedback = await vscode.window.showInputBox({
          prompt: 'Enter detailed feedback (optional)',
          placeHolder: 'What worked well? What could be improved?'
        });

        const response = await fetch('http://localhost:5173/api/evaluation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'feedback',
            agentType: 'rag',
            operation: 'user_interaction',
            query: 'VS Code interaction',
            response: {},
            rating: ratingValue,
            feedback: feedback || '',
            sessionId: 'vscode-session'
          })
        });

        const data = await response.json();
        if (data.success) {
          vscode.window.showInformationMessage('✅ Feedback recorded successfully');
        } else {
          vscode.window.showErrorMessage(`Failed to record feedback: ${data.error}`);
        }
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to record feedback: ${error.message}`);
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('mcp.viewMetrics', async () => {
    try {
      const timeWindow = await vscode.window.showQuickPick(
        ['1 - Last Hour', '6 - Last 6 Hours', '24 - Last 24 Hours', '168 - Last Week'],
        { placeHolder: 'Select time window for metrics' }
      );

      if (timeWindow) {
        const hours = parseInt(timeWindow.split(' - ')[0]);
        const response = await fetch(`http://localhost:5173/api/evaluation?action=metrics&timeWindow=${hours}`);
        const data = await response.json();

        if (data.success) {
          await showResultPanel(`Performance Metrics (${hours}h)`, data.metrics);
        } else {
          vscode.window.showErrorMessage(`Failed to get metrics: ${data.error}`);
        }
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to view metrics: ${error.message}`);
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('mcp.getBenchmarks', async () => {
    try {
      const response = await fetch('http://localhost:5173/api/evaluation?action=benchmarks');
      const data = await response.json();

      if (data.success) {
        await showResultPanel('Benchmark Results', {
          metrics: data.metrics,
          trends: data.trends,
          recommendations: data.recommendations
        });
      } else {
        vscode.window.showErrorMessage(`Failed to get benchmarks: ${data.error}`);
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to get benchmarks: ${error.message}`);
    }
  }));

  // ...existing code...
}
  // Analyze Current Context
  const analyzeContextCommand = vscode.commands.registerCommand(
    "mcp.analyzeCurrentContext",
    async () => {
      try {
        statusBarManager.updateStatus(
          "analyzing",
          "Analyzing context (cluster+RAG+GPU+auto-fix)..."
        );
        const vsCodeContext = await buildVSCodeContext();
        // Use service worker for semantic search
        const semanticResults = await runSemanticSearchWithWorkers(
          "recent codebase changes, best practices, architecture, TODO, technical debt",
          vsCodeContext
        );
        // Lazy load context for RAG
        const lazyFiles = await getLazyLoadedContext(vsCodeContext.activeFiles);
        const ragResults = await mcpServerManager.callMCPTool("enhanced_rag", {
          files: lazyFiles,
          context: vsCodeContext,
        });
        // Run best practices
        const bestPractices = await mcpServerManager.callMCPTool(
          "generate-best-practices",
          { area: "ui-ux" }
        );
        // Run error check
        const errors = await mcpServerManager.callMCPTool("get-errors", {
          filePaths: lazyFiles,
        });
        // Auto-fix if possible
        await autoFixIfPossible(errors, bestPractices, lazyFiles);
        // Store context summary for RL
        storeContextSummary({
          semantic: semanticResults,
          rag: ragResults,
          bestPractices,
          errors,
        });
        // Show results
        await showSuggestionsPanel(
          semanticResults,
          "Semantic Search Suggestions"
        );
        vscode.window.showInformationMessage(
          "RAG/Embedding Results: " +
            JSON.stringify(ragResults && ragResults[0])
        );
        statusBarManager.updateStatus(
          "ready",
          `Semantic+RAG+AutoFix analysis complete`
        );
      } catch (error) {
        vscode.window.showErrorMessage(`Context analysis failed: ${error}`);
        statusBarManager.updateStatus("error", "Analysis failed");
      }
    }
  );
  // Suggest Best Practices
  const suggestBestPracticesCommand = vscode.commands.registerCommand(
    "mcp.suggestBestPractices",
    async () => {
      try {
        statusBarManager.updateStatus(
          "analyzing",
          "Generating best practices..."
        );
        const area = await vscode.window.showQuickPick(
          ["performance", "security", "ui-ux"],
          { placeHolder: "Select area for best practices" }
        );
        if (area) {
          const result = await mcpServerManager.callMCPTool(
            "generate-best-practices",
            { area }
          );
          await showResultPanel(`Best Practices: ${area}`, result);
        }
        statusBarManager.updateStatus("ready", "Best practices generated");
      } catch (error) {
        vscode.window.showErrorMessage(
          `Best practices generation failed: ${error}`
        );
        statusBarManager.updateStatus("error", "Generation failed");
      }
    }
  );
  // Get Context-Aware Documentation
  const getContextAwareDocsCommand = vscode.commands.registerCommand(
    "mcp.getContextAwareDocs",
    async () => {
      try {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
          vscode.window.showWarningMessage("No active editor found");
          return;
        }
        const vsCodeContext = await buildVSCodeContext();
        const suggestions = await contextAnalyzer.analyzeCurrentFileForDocs(
          activeEditor.document.fileName,
          vsCodeContext
        );
        if (suggestions.length > 0) {
          const selected = await vscode.window.showQuickPick(
            suggestions.map((s) => ({
              label:
                s.args.context7CompatibleLibraryID ||
                s.args.component ||
                s.tool,
              description: s.reasoning,
              suggestion: s,
            })),
            { placeHolder: "Select documentation to retrieve" }
          );
          if (selected) {
            const result = await mcpServerManager.callMCPTool(
              selected.suggestion.tool,
              selected.suggestion.args
            );
            await showResultPanel(`Documentation: ${selected.label}`, result);
          }
        } else {
          vscode.window.showInformationMessage(
            "No relevant documentation suggestions found"
          );
        }
      } catch (error) {
        vscode.window.showErrorMessage(
          `Documentation retrieval failed: ${error}`
        );
      }
    }
  );
  // Analyze TypeScript Errors
  const analyzeErrorsCommand = vscode.commands.registerCommand(
    "mcp.analyzeErrors",
    async () => {
      try {
        const diagnostics = vscode.languages.getDiagnostics();
        const errors =
          diagnosticWatcher.convertDiagnosticsToErrors(diagnostics);
        if (errors.length === 0) {
          vscode.window.showInformationMessage("No TypeScript errors found!");
          return;
        }
        const suggestions =
          contextAnalyzer.analyzeErrorsForMCPSuggestions(errors);
        await showSuggestionsPanel(
          suggestions,
          `Analyzed ${errors.length} TypeScript errors`
        );
      } catch (error) {
        vscode.window.showErrorMessage(`Error analysis failed: ${error}`);
      }
    }
  );
  // Analyze Full Tech Stack
  const analyzeStackCommand = vscode.commands.registerCommand(
    "mcp.analyzeFullStack",
    async () => {
      try {
        statusBarManager.updateStatus("analyzing", "Analyzing tech stack...");
        const { projectType, detectedStack } =
          await stackAnalyzer.analyzeFullStack();
        const mcpSuggestions =
          stackAnalyzer.getMCPDocSuggestions(detectedStack);
        // Create comprehensive stack report
        const stackReport = generateStackReport(
          projectType,
          detectedStack,
          mcpSuggestions
        );
        await showResultPanel("🔍 Tech Stack Analysis", stackReport);
        statusBarManager.updateStatus("ready", `Project: ${projectType}`);
      } catch (error) {
        vscode.window.showErrorMessage(`Tech stack analysis failed: ${error}`);
        statusBarManager.updateStatus("error", "Analysis failed");
      }
    }
  );
  // Start/Stop Server Commands
  const startServerCommand = vscode.commands.registerCommand(
    "mcp.startServer",
    () => {
      mcpServerManager.startServer();
    }
  );
  const stopServerCommand = vscode.commands.registerCommand(
    "mcp.stopServer",
    () => {
      mcpServerManager.stopServer();
    }
  );
  // Register all commands
  context.subscriptions.push(
    analyzeContextCommand,
    suggestBestPracticesCommand,
    getContextAwareDocsCommand,
    analyzeErrorsCommand,
    analyzeStackCommand,
    startServerCommand,
    stopServerCommand,
    runAgentOrchestratorCommand
  );
}
function setupEventListeners(context) {
  // Watch for diagnostic changes (TypeScript errors)
  const onDiagnosticsChange = vscode.languages.onDidChangeDiagnostics((e) => {
    diagnosticWatcher.onDiagnosticsChanged(e);
  });
  // Watch for active editor changes
  const onActiveEditorChange = vscode.window.onDidChangeActiveTextEditor(
    (editor) => {
      if (editor) {
        contextAnalyzer.onActiveEditorChanged(editor);
      }
    }
  );
  // Watch for workspace folder changes
  const onWorkspaceFoldersChange = vscode.workspace.onDidChangeWorkspaceFolders(
    (e) => {
      mcpServerManager.onWorkspaceChanged(e);
    }
  );

  // === Watch for file changes (Claude code, CLI, or any codebase change) ===
  const watcher = vscode.workspace.createFileSystemWatcher(
    "**/*.{ts,js,svelte,md,json}"
  );
  watcher.onDidChange(async (uri) => {
    await runCopilotSelfPrompt(uri.fsPath, "file-change");
  });
  context.subscriptions.push(
    onDiagnosticsChange,
    onActiveEditorChange,
    onWorkspaceFoldersChange,
    watcher
  );

  // === Monitor: periodic self-autoprompt Copilot agent ===
  let monitorInterval = setInterval(
    async () => {
      await runCopilotSelfPrompt(undefined, "interval");
    },
    1000 * 60 * 10
  ); // every 10 minutes
  context.subscriptions.push({ dispose: () => clearInterval(monitorInterval) });
}

// Self-autoprompt Copilot agent logic
async function runCopilotSelfPrompt(changedFile, trigger) {
  try {
    const vsCodeContext = await buildVSCodeContext();
    // Optionally, include changedFile and trigger in the prompt/context
    const prompt = `Copilot: Analyze the codebase for best practices, errors, and architecture compliance. Trigger: ${trigger}${changedFile ? ", File: " + changedFile : ""}`;
    // Call orchestrator or MCP tool for self-prompting
    const result = await mcpServerManager.callMCPTool("sequentialthinking", {
      thought: prompt,
      nextThoughtNeeded: false,
      thoughtNumber: 1,
      totalThoughts: 1,
    });
    vscode.window.showInformationMessage(
      "Copilot self-prompt result: " + JSON.stringify(result)
    );
    // Optionally: auto-fix or suggest fixes (future enhancement)
  } catch (e) {
    vscode.window.showErrorMessage("Copilot self-prompt failed: " + e.message);
  }
}
async function buildVSCodeContext() {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  const workspaceRoot = workspaceFolders?.[0]?.uri.fsPath || "";
  const activeFiles = vscode.workspace.textDocuments.map((doc) => doc.fileName);
  const currentFile = vscode.window.activeTextEditor?.document.fileName;
  const diagnostics = vscode.languages.getDiagnostics();
  const errors = diagnosticWatcher.convertDiagnosticsToErrors(diagnostics);
  // Get recent prompts from comments in open files
  const recentPrompts = await contextAnalyzer.extractRecentPrompts(activeFiles);
  // Analyze full tech stack and project type
  const { projectType, detectedStack } =
    await detectProjectTypeAndStack(workspaceRoot);
  return {
    workspaceRoot,
    activeFiles,
    currentFile,
    errors,
    userIntent: "debugging",
    recentPrompts,
    projectType,
    detectedStack,
  };
}
async function detectProjectTypeAndStack(workspaceRoot) {
  try {
    // Use the comprehensive stack analyzer
    return await stackAnalyzer.analyzeFullStack();
  } catch (error) {
    console.log("Could not analyze project stack:", error);
    return {
      projectType: "generic",
      detectedStack: {
        frontend: [],
        backend: [],
        databases: [],
        cloud: [],
        aiml: [],
        gpu: [],
        embedded: [],
        systems: [],
        scientific: [],
        gaming: [],
        mobile: [],
        web3: [],
      },
    };
  }
}
async function showSuggestionsPanel(suggestions, title = "MCP Suggestions") {
  if (suggestions.length === 0) {
    vscode.window.showInformationMessage(
      "No MCP suggestions available for current context"
    );
    return;
  }
  const selected = await vscode.window.showQuickPick(
    suggestions.map((s) => ({
      label: `$(${s.priority === "high" ? "warning" : "info"}) ${s.tool}`,
      description: s.reasoning,
      detail: `Confidence: ${(s.confidence * 100).toFixed(0)}% | Expected: ${s.expectedOutput}`,
      suggestion: s,
    })),
    {
      placeHolder: `${title} - Select an MCP tool to execute`,
      matchOnDescription: true,
      matchOnDetail: true,
    }
  );
  if (selected) {
    try {
      statusBarManager.updateStatus(
        "executing",
        `Executing ${selected.suggestion.tool}...`
      );
      const result = await mcpServerManager.callMCPTool(
        selected.suggestion.tool,
        selected.suggestion.args
      );
      await showResultPanel(`${selected.suggestion.tool} Result`, result);
      statusBarManager.updateStatus("ready", "MCP tool executed successfully");
    } catch (error) {
      vscode.window.showErrorMessage(`MCP tool execution failed: ${error}`);
      statusBarManager.updateStatus("error", "Execution failed");
    }
  }
}
function generateStackReport(projectType, stack, mcpSuggestions) {
  let report = `# 🎯 Project Analysis Report\n\n`;
  report += `**Project Type:** \`${projectType}\`\n\n`;
  // Frontend Technologies
  if (stack.frontend.length > 0) {
    report += `## 🎨 Frontend Technologies\n`;
    stack.frontend.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Backend Technologies
  if (stack.backend.length > 0) {
    report += `## ⚙️ Backend Technologies\n`;
    stack.backend.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // GPU/CUDA Computing
  if (stack.gpu.length > 0) {
    report += `## 🚀 GPU/CUDA Computing\n`;
    stack.gpu.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // AI/ML Technologies
  if (stack.aiml.length > 0) {
    report += `## 🤖 AI/ML Technologies\n`;
    stack.aiml.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Embedded/Hardware
  if (stack.embedded.length > 0) {
    report += `## 🔌 Embedded/Hardware\n`;
    stack.embedded.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Systems Programming
  if (stack.systems.length > 0) {
    report += `## 🖥️ Systems Programming\n`;
    stack.systems.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Scientific Computing
  if (stack.scientific.length > 0) {
    report += `## 🔬 Scientific Computing\n`;
    stack.scientific.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Databases
  if (stack.databases.length > 0) {
    report += `## 🗄️ Databases\n`;
    stack.databases.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Cloud & DevOps
  if (stack.cloud.length > 0) {
    report += `## ☁️ Cloud & DevOps\n`;
    stack.cloud.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Gaming
  if (stack.gaming.length > 0) {
    report += `## 🎮 Game Development\n`;
    stack.gaming.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Mobile
  if (stack.mobile.length > 0) {
    report += `## 📱 Mobile Development\n`;
    stack.mobile.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // Web3/Blockchain
  if (stack.web3.length > 0) {
    report += `## ⛓️ Web3/Blockchain\n`;
    stack.web3.forEach((tech) => (report += `- ${tech}\n`));
    report += `\n`;
  }
  // MCP Documentation Suggestions
  if (mcpSuggestions.length > 0) {
    report += `## 📚 Recommended MCP Documentation\n`;
    mcpSuggestions.forEach((suggestion) => {
      report += `### ${suggestion.library}\n`;
      suggestion.topics.forEach((topic) => (report += `- ${topic}\n`));
      report += `\n`;
    });
  }
  // Context-Aware Recommendations
  report += `## 💡 Context-Aware Recommendations\n`;
  if (projectType === "cuda-gpu-computing") {
    report += `- Consider using \`nvcc\` compiler optimizations\n`;
    report += `- Profile GPU memory usage with \`nvidia-smi\` and \`nvprof\`\n`;
    report += `- Leverage CUDA streams for concurrent execution\n`;
    report += `- Consider TensorRT for inference optimization\n`;
  }
  if (projectType === "electrical-embedded") {
    report += `- Set up cross-compilation toolchain\n`;
    report += `- Consider real-time constraints and RTOS usage\n`;
    report += `- Implement proper interrupt handling\n`;
    report += `- Use hardware abstraction layers (HAL)\n`;
  }
  if (projectType === "ml-ai-research") {
    report += `- Set up experiment tracking (MLflow, Weights & Biases)\n`;
    report += `- Consider model versioning and reproducibility\n`;
    report += `- Implement proper data validation and monitoring\n`;
    report += `- Use distributed training for large models\n`;
  }
  if (projectType === "sveltekit-legal-ai") {
    report += `- Implement proper TypeScript configurations\n`;
    report += `- Set up vector database optimization\n`;
    report += `- Consider legal data privacy compliance\n`;
    report += `- Implement proper error handling and logging\n`;
  }
  report += `\n---\n`;
  report += `*Generated by Context7 MCP Assistant - ${new Date().toLocaleString()}*`;
  return report;
}
async function showResultPanel(title, result) {
  const panel = vscode.window.createWebviewPanel(
    "mcpResult",
    title,
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );
  const resultText =
    typeof result === "string" ? result : JSON.stringify(result, null, 2);
  panel.webview.html = `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>${title}</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    padding: 20px;
                    line-height: 1.6;
                    color: var(--vscode-editor-foreground);
                    background-color: var(--vscode-editor-background);
                }
                .header {
                    border-bottom: 2px solid var(--vscode-textSeparator-foreground);
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .content {
                    white-space: pre-wrap;
                    font-family: 'Courier New', monospace;
                    background-color: var(--vscode-textCodeBlock-background);
                    padding: 15px;
                    border-radius: 5px;
                    border: 1px solid var(--vscode-panel-border);
                }
                .copy-button {
                    margin-top: 10px;
                    padding: 8px 16px;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 3px;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 ${title}</h1>
                <p>Generated by Context7 MCP Assistant</p>
            </div>
            <div class="content">${resultText}</div>
            <button class="copy-button" onclick="copyToClipboard()">📋 Copy to Clipboard</button>

            <script>
                function copyToClipboard() {
                    const content = document.querySelector('.content').textContent;
                    navigator.clipboard.writeText(content).then(() => {
                        const button = document.querySelector('.copy-button');
                        const originalText = button.textContent;
                        button.textContent = '✅ Copied!';
                        setTimeout(() => {
                            button.textContent = originalText;
                        }, 2000);
                    });
                }
            </script>
        </body>
        </html>
    `;
}
//# sourceMappingURL=extension.js.map
