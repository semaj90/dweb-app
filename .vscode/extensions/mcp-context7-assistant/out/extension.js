"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
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
const ragBackendClient_1 = require("./ragBackendClient");
const ragCommands_1 = require("./ragCommands");
let mcpServerManager;
let contextAnalyzer;
let statusBarManager;
let diagnosticWatcher;
let stackAnalyzer;
let ragClient;
let ragCommandProvider;
function activate(context) {
    console.log("🚀 Context7 MCP Assistant extension is now active!");
    // Initialize core components
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || "";
    statusBarManager = new statusBarManager_1.StatusBarManager();
    mcpServerManager = new mcpServerManager_1.MCPServerManager(context, statusBarManager);
    contextAnalyzer = new contextAnalyzer_1.ContextAnalyzer();
    diagnosticWatcher = new diagnosticWatcher_1.DiagnosticWatcher();
    stackAnalyzer = new stackAnalyzer_1.StackAnalyzer(workspaceRoot);
    // Initialize Enhanced RAG Backend Integration
    const config = vscode.workspace.getConfiguration("mcpContext7");
    ragClient = new ragBackendClient_1.RAGBackendClient({
        baseUrl: config.get("ragBackendUrl", "http://localhost:8000"),
        timeout: config.get("ragTimeout", 30000),
        retries: config.get("ragRetries", 3),
    });
    ragCommandProvider = new ragCommands_1.RAGCommandProvider(ragClient);
    // Register commands
    registerCommands(context);
    // Register RAG commands
    ragCommandProvider.registerCommands(context);
    // Setup event listeners
    setupEventListeners(context);
    // Auto-start server if enabled
    if (config.get("autoStart", true)) {
        mcpServerManager.startServer();
    }
    // Check RAG backend health
    ragClient
        .healthCheck()
        .then((isHealthy) => {
        if (isHealthy) {
            console.log("✅ Enhanced RAG Backend is healthy");
            statusBarManager.updateStatus("ready", "Context7 MCP + Enhanced RAG Ready");
        }
        else {
            console.log("❌ Enhanced RAG Backend is not available");
            statusBarManager.updateStatus("ready", "Context7 MCP Ready (RAG Offline)");
        }
    })
        .catch((error) => {
        console.log("⚠️ Could not connect to Enhanced RAG Backend:", error);
        statusBarManager.updateStatus("ready", "Context7 MCP Ready (RAG Offline)");
    });
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
    if (ragClient) {
        ragClient.dispose();
    }
}
exports.deactivate = deactivate;
function registerCommands(context) {
    // Enhanced Analyze Current Context with RAG Integration
    const analyzeContextCommand = vscode.commands.registerCommand("mcp.analyzeCurrentContext", async () => {
        try {
            statusBarManager.updateStatus("analyzing", "Enhanced RAG context analysis...");
            // Step 1: Build VS Code context
            const vsCodeContext = await buildVSCodeContext();
            // Step 2: Get enhanced context-aware suggestions
            const suggestions = await contextAnalyzer.getContextAwareSuggestions(vsCodeContext);
            // Step 3: Use Enhanced RAG for deeper analysis
            const activeEditor = vscode.window.activeTextEditor;
            if (activeEditor) {
                const currentText = activeEditor.document.getText();
                const fileName = activeEditor.document.fileName;
                // Enhanced RAG query for current context
                const ragResult = await mcpServerManager.callMCPTool("enhanced_rag_query", {
                    query: `Analyze this ${fileName} file context and suggest improvements: ${currentText.substring(0, 1000)}`,
                    caseId: "context_analysis",
                    maxResults: 10,
                    includeContext7: true,
                });
                // Memory graph update for context tracking
                await mcpServerManager.callMCPTool("mcp_memory2_create_relations", {
                    entities: [
                        {
                            type: "file",
                            id: fileName,
                            properties: {
                                language: activeEditor.document.languageId,
                                lines: activeEditor.document.lineCount,
                                analyzed_at: new Date().toISOString(),
                            },
                        },
                        {
                            type: "analysis",
                            id: `analysis_${Date.now()}`,
                            properties: {
                                suggestions_count: suggestions.length,
                                rag_enhanced: ragResult.success,
                            },
                        },
                    ],
                });
                // Enhanced suggestions with RAG insights
                if (ragResult.success && ragResult.data.sources.length > 0) {
                    const ragSuggestions = ragResult.data.sources.map((source, index) => ({
                        tool: "enhanced-rag-insight",
                        reasoning: `RAG-enhanced suggestion: ${source.content.substring(0, 100)}...`,
                        confidence: source.similarity || 0.8,
                        priority: "high",
                        expectedOutput: "Context-aware code improvement",
                        args: {
                            insight: source.content,
                            relevance: source.similarity,
                            enhanced_rag: true,
                        },
                    }));
                    suggestions.push(...ragSuggestions);
                }
            }
            await showSuggestionsPanel(suggestions, "🤖 Enhanced RAG Context Analysis");
            statusBarManager.updateStatus("ready", `Enhanced analysis: ${suggestions.length} insights`);
            // Show performance metrics
            const metrics = mcpServerManager.getEnhancedMetrics();
            console.log("🔍 Enhanced RAG Analysis Metrics:", metrics);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Enhanced context analysis failed: ${error}`);
            statusBarManager.updateStatus("error", "Enhanced analysis failed");
        }
    });
    // Suggest Best Practices
    const suggestBestPracticesCommand = vscode.commands.registerCommand("mcp.suggestBestPractices", async () => {
        try {
            statusBarManager.updateStatus("analyzing", "Generating best practices...");
            const area = await vscode.window.showQuickPick(["performance", "security", "ui-ux"], { placeHolder: "Select area for best practices" });
            if (area) {
                const result = await mcpServerManager.callMCPTool("generate-best-practices", { area });
                await showResultPanel(`Best Practices: ${area}`, result);
            }
            statusBarManager.updateStatus("ready", "Best practices generated");
        }
        catch (error) {
            vscode.window.showErrorMessage(`Best practices generation failed: ${error}`);
            statusBarManager.updateStatus("error", "Generation failed");
        }
    });
    // Get Context-Aware Documentation
    const getContextAwareDocsCommand = vscode.commands.registerCommand("mcp.getContextAwareDocs", async () => {
        try {
            const activeEditor = vscode.window.activeTextEditor;
            if (!activeEditor) {
                vscode.window.showWarningMessage("No active editor found");
                return;
            }
            const vsCodeContext = await buildVSCodeContext();
            const suggestions = await contextAnalyzer.analyzeCurrentFileForDocs(activeEditor.document.fileName, vsCodeContext);
            if (suggestions.length > 0) {
                const selected = await vscode.window.showQuickPick(suggestions.map((s) => ({
                    label: s.args.context7CompatibleLibraryID ||
                        s.args.component ||
                        s.tool,
                    description: s.reasoning,
                    suggestion: s,
                })), { placeHolder: "Select documentation to retrieve" });
                if (selected) {
                    const result = await mcpServerManager.callMCPTool(selected.suggestion.tool, selected.suggestion.args);
                    await showResultPanel(`Documentation: ${selected.label}`, result);
                }
            }
            else {
                vscode.window.showInformationMessage("No relevant documentation suggestions found");
            }
        }
        catch (error) {
            vscode.window.showErrorMessage(`Documentation retrieval failed: ${error}`);
        }
    });
    // Analyze TypeScript Errors
    const analyzeErrorsCommand = vscode.commands.registerCommand("mcp.analyzeErrors", async () => {
        try {
            const diagnostics = vscode.languages.getDiagnostics();
            const errors = diagnosticWatcher.convertDiagnosticsToErrors(diagnostics);
            if (errors.length === 0) {
                vscode.window.showInformationMessage("No TypeScript errors found!");
                return;
            }
            const suggestions = contextAnalyzer.analyzeErrorsForMCPSuggestions(errors);
            await showSuggestionsPanel(suggestions, `Analyzed ${errors.length} TypeScript errors`);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Error analysis failed: ${error}`);
        }
    });
    // Analyze Full Tech Stack
    const analyzeStackCommand = vscode.commands.registerCommand("mcp.analyzeFullStack", async () => {
        try {
            statusBarManager.updateStatus("analyzing", "Analyzing tech stack...");
            const { projectType, detectedStack } = await stackAnalyzer.analyzeFullStack();
            const mcpSuggestions = stackAnalyzer.getMCPDocSuggestions(detectedStack);
            // Create comprehensive stack report
            const stackReport = generateStackReport(projectType, detectedStack, mcpSuggestions);
            await showResultPanel("🔍 Tech Stack Analysis", stackReport);
            statusBarManager.updateStatus("ready", `Project: ${projectType}`);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Tech stack analysis failed: ${error}`);
            statusBarManager.updateStatus("error", "Analysis failed");
        }
    });
    // Enhanced RAG Commands
    const enhancedRAGQueryCommand = vscode.commands.registerCommand("mcp.enhancedRAGQuery", async () => {
        try {
            const query = await vscode.window.showInputBox({
                prompt: "Enter your enhanced RAG query",
                placeHolder: 'e.g., "How to implement SvelteKit forms with Drizzle ORM?"',
            });
            if (query) {
                statusBarManager.updateStatus("processing", "Processing Enhanced RAG query...");
                const result = await mcpServerManager.callMCPTool("enhanced_rag_query", {
                    query,
                    caseId: "user_query",
                    maxResults: 10,
                    includeContext7: true,
                });
                await showResultPanel(`🤖 Enhanced RAG Query: ${query}`, result.data);
                statusBarManager.updateStatus("ready", "Enhanced RAG query completed");
            }
        }
        catch (error) {
            vscode.window.showErrorMessage(`Enhanced RAG query failed: ${error}`);
        }
    });
    const agentOrchestrationCommand = vscode.commands.registerCommand("mcp.agentOrchestration", async () => {
        try {
            const agentType = await vscode.window.showQuickPick(["claude", "crewai", "autogen"], { placeHolder: "Select agent for orchestration" });
            if (agentType) {
                const prompt = await vscode.window.showInputBox({
                    prompt: `Enter prompt for ${agentType} agent`,
                    placeHolder: 'e.g., "Analyze this codebase for performance improvements"',
                });
                if (prompt) {
                    statusBarManager.updateStatus("processing", `Orchestrating ${agentType} agent...`);
                    const result = await mcpServerManager.callMCPTool(`agent_orchestrate_${agentType}`, {
                        prompt,
                        context: "vscode_extension",
                        autoFix: true,
                    });
                    await showResultPanel(`🤖 ${agentType} Agent Result`, result.data);
                    statusBarManager.updateStatus("ready", `${agentType} agent completed`);
                }
            }
        }
        catch (error) {
            vscode.window.showErrorMessage(`Agent orchestration failed: ${error}`);
        }
    });
    const memoryGraphCommand = vscode.commands.registerCommand("mcp.memoryGraph", async () => {
        try {
            const action = await vscode.window.showQuickPick(["Read Graph", "Search Nodes", "Create Relations"], { placeHolder: "Select memory graph action" });
            if (action) {
                let result;
                switch (action) {
                    case "Read Graph":
                        result = await mcpServerManager.callMCPTool("mcp_memory2_read_graph", {});
                        break;
                    case "Search Nodes":
                        const searchQuery = await vscode.window.showInputBox({
                            prompt: "Enter search query for memory nodes",
                            placeHolder: 'e.g., "files related to authentication"',
                        });
                        if (searchQuery) {
                            result = await mcpServerManager.callMCPTool("mcp_memory2_search_nodes", {
                                query: searchQuery,
                            });
                        }
                        break;
                    case "Create Relations":
                        result = await mcpServerManager.callMCPTool("mcp_memory2_create_relations", {
                            entities: [
                                {
                                    type: "command",
                                    id: "memory_graph_demo",
                                    properties: { created_from: "vscode_extension" },
                                },
                            ],
                        });
                        break;
                }
                if (result) {
                    await showResultPanel(`🧠 Memory Graph: ${action}`, result.data);
                }
            }
        }
        catch (error) {
            vscode.window.showErrorMessage(`Memory graph operation failed: ${error}`);
        }
    });
    const enhancedMetricsCommand = vscode.commands.registerCommand("mcp.enhancedMetrics", async () => {
        try {
            const metrics = mcpServerManager.getEnhancedMetrics();
            await showResultPanel("📊 Enhanced RAG System Metrics", metrics);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Metrics retrieval failed: ${error}`);
        }
    });
    // Start/Stop Server Commands
    const startServerCommand = vscode.commands.registerCommand("mcp.startServer", () => {
        mcpServerManager.startServer();
    });
    const stopServerCommand = vscode.commands.registerCommand("mcp.stopServer", () => {
        mcpServerManager.stopServer();
    });
    // Register all commands
    context.subscriptions.push(analyzeContextCommand, suggestBestPracticesCommand, getContextAwareDocsCommand, analyzeErrorsCommand, analyzeStackCommand, enhancedRAGQueryCommand, agentOrchestrationCommand, memoryGraphCommand, enhancedMetricsCommand, startServerCommand, stopServerCommand);
}
function setupEventListeners(context) {
    // Watch for diagnostic changes (TypeScript errors)
    const onDiagnosticsChange = vscode.languages.onDidChangeDiagnostics((e) => {
        diagnosticWatcher.onDiagnosticsChanged(e);
    });
    // Watch for active editor changes
    const onActiveEditorChange = vscode.window.onDidChangeActiveTextEditor((editor) => {
        if (editor) {
            contextAnalyzer.onActiveEditorChanged(editor);
        }
    });
    // Watch for workspace folder changes
    const onWorkspaceFoldersChange = vscode.workspace.onDidChangeWorkspaceFolders((e) => {
        mcpServerManager.onWorkspaceChanged(e);
    });
    context.subscriptions.push(onDiagnosticsChange, onActiveEditorChange, onWorkspaceFoldersChange);
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
    const { projectType, detectedStack } = await detectProjectTypeAndStack(workspaceRoot);
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
    }
    catch (error) {
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
        vscode.window.showInformationMessage("No MCP suggestions available for current context");
        return;
    }
    const selected = await vscode.window.showQuickPick(suggestions.map((s) => ({
        label: `$(${s.priority === "high" ? "warning" : "info"}) ${s.tool}`,
        description: s.reasoning,
        detail: `Confidence: ${(s.confidence * 100).toFixed(0)}% | Expected: ${s.expectedOutput}`,
        suggestion: s,
    })), {
        placeHolder: `${title} - Select an MCP tool to execute`,
        matchOnDescription: true,
        matchOnDetail: true,
    });
    if (selected) {
        try {
            statusBarManager.updateStatus("executing", `Executing ${selected.suggestion.tool}...`);
            const result = await mcpServerManager.callMCPTool(selected.suggestion.tool, selected.suggestion.args);
            await showResultPanel(`${selected.suggestion.tool} Result`, result);
            statusBarManager.updateStatus("ready", "MCP tool executed successfully");
        }
        catch (error) {
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
    const panel = vscode.window.createWebviewPanel("mcpResult", title, vscode.ViewColumn.Beside, {
        enableScripts: true,
        retainContextWhenHidden: true,
    });
    const resultText = typeof result === "string" ? result : JSON.stringify(result, null, 2);
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