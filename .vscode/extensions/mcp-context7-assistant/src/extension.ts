import * as vscode from 'vscode';
import { MCPServerManager } from './mcpServerManager';
import { ContextAnalyzer } from './contextAnalyzer';
import { StatusBarManager } from './statusBarManager';
import { DiagnosticWatcher } from './diagnosticWatcher';
import { StackAnalyzer } from './stackAnalyzer';
import { VSCodeMCPContext, AutoMCPSuggestion, ProjectType, TechStack } from './types';

let mcpServerManager: MCPServerManager;
let contextAnalyzer: ContextAnalyzer;
let statusBarManager: StatusBarManager;
let diagnosticWatcher: DiagnosticWatcher;
let stackAnalyzer: StackAnalyzer;

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 Context7 MCP Assistant extension is now active!');

    // Initialize core components
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    mcpServerManager = new MCPServerManager(context);
    contextAnalyzer = new ContextAnalyzer();
    statusBarManager = new StatusBarManager();
    diagnosticWatcher = new DiagnosticWatcher();
    stackAnalyzer = new StackAnalyzer(workspaceRoot);

    // Register commands
    registerCommands(context);

    // Setup event listeners
    setupEventListeners(context);

    // Auto-start server if enabled
    const config = vscode.workspace.getConfiguration('mcpContext7');
    if (config.get('autoStart', true)) {
        mcpServerManager.startServer();
    }

    // Update status bar
    statusBarManager.updateStatus('ready', 'Context7 MCP Ready');
}

export function deactivate() {
    console.log('🛑 Context7 MCP Assistant extension is deactivating...');
    
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

function registerCommands(context: vscode.ExtensionContext) {
    // Analyze Current Context
    const analyzeContextCommand = vscode.commands.registerCommand('mcp.analyzeCurrentContext', async () => {
        try {
            statusBarManager.updateStatus('analyzing', 'Analyzing context...');
            
            const vsCodeContext = await buildVSCodeContext();
            const suggestions = await contextAnalyzer.getContextAwareSuggestions(vsCodeContext);
            
            await showSuggestionsPanel(suggestions);
            statusBarManager.updateStatus('ready', `Found ${suggestions.length} suggestions`);
        } catch (error) {
            vscode.window.showErrorMessage(`Context analysis failed: ${error}`);
            statusBarManager.updateStatus('error', 'Analysis failed');
        }
    });

    // Suggest Best Practices
    const suggestBestPracticesCommand = vscode.commands.registerCommand('mcp.suggestBestPractices', async () => {
        try {
            statusBarManager.updateStatus('analyzing', 'Generating best practices...');
            
            const area = await vscode.window.showQuickPick(
                ['performance', 'security', 'ui-ux'],
                { placeHolder: 'Select area for best practices' }
            );
            
            if (area) {
                const result = await mcpServerManager.callMCPTool('generate-best-practices', { area });
                await showResultPanel(`Best Practices: ${area}`, result);
            }
            
            statusBarManager.updateStatus('ready', 'Best practices generated');
        } catch (error) {
            vscode.window.showErrorMessage(`Best practices generation failed: ${error}`);
            statusBarManager.updateStatus('error', 'Generation failed');
        }
    });

    // Get Context-Aware Documentation
    const getContextAwareDocsCommand = vscode.commands.registerCommand('mcp.getContextAwareDocs', async () => {
        try {
            const activeEditor = vscode.window.activeTextEditor;
            if (!activeEditor) {
                vscode.window.showWarningMessage('No active editor found');
                return;
            }

            const vsCodeContext = await buildVSCodeContext();
            const suggestions = await contextAnalyzer.analyzeCurrentFileForDocs(
                activeEditor.document.fileName,
                vsCodeContext
            );

            if (suggestions.length > 0) {
                interface DocQuickPickItem extends vscode.QuickPickItem {
                    suggestion: AutoMCPSuggestion;
                }

                const selected = await vscode.window.showQuickPick<DocQuickPickItem>(
                    suggestions.map(s => ({
                        label: s.args.context7CompatibleLibraryID || s.args.component || s.tool,
                        description: s.reasoning,
                        suggestion: s
                    })),
                    { placeHolder: 'Select documentation to retrieve' }
                );

                if (selected) {
                    const result = await mcpServerManager.callMCPTool(selected.suggestion.tool, selected.suggestion.args);
                    await showResultPanel(`Documentation: ${selected.label}`, result);
                }
            } else {
                vscode.window.showInformationMessage('No relevant documentation suggestions found');
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Documentation retrieval failed: ${error}`);
        }
    });

    // Analyze TypeScript Errors
    const analyzeErrorsCommand = vscode.commands.registerCommand('mcp.analyzeErrors', async () => {
        try {
            const diagnostics = vscode.languages.getDiagnostics();
            const errors = diagnosticWatcher.convertDiagnosticsToErrors(diagnostics);
            
            if (errors.length === 0) {
                vscode.window.showInformationMessage('No TypeScript errors found!');
                return;
            }

            const suggestions = contextAnalyzer.analyzeErrorsForMCPSuggestions(errors);
            await showSuggestionsPanel(suggestions, `Analyzed ${errors.length} TypeScript errors`);
        } catch (error) {
            vscode.window.showErrorMessage(`Error analysis failed: ${error}`);
        }
    });

    // Analyze Full Tech Stack
    const analyzeStackCommand = vscode.commands.registerCommand('mcp.analyzeFullStack', async () => {
        try {
            statusBarManager.updateStatus('analyzing', 'Analyzing tech stack...');
            
            const { projectType, detectedStack } = await stackAnalyzer.analyzeFullStack();
            const mcpSuggestions = stackAnalyzer.getMCPDocSuggestions(detectedStack);
            
            // Create comprehensive stack report
            const stackReport = generateStackReport(projectType, detectedStack, mcpSuggestions);
            await showResultPanel('🔍 Tech Stack Analysis', stackReport);
            
            statusBarManager.updateStatus('ready', `Project: ${projectType}`);
        } catch (error) {
            vscode.window.showErrorMessage(`Tech stack analysis failed: ${error}`);
            statusBarManager.updateStatus('error', 'Analysis failed');
        }
    });

    // Start/Stop Server Commands
    const startServerCommand = vscode.commands.registerCommand('mcp.startServer', () => {
        mcpServerManager.startServer();
    });

    const stopServerCommand = vscode.commands.registerCommand('mcp.stopServer', () => {
        mcpServerManager.stopServer();
    });

    // Register all commands
    context.subscriptions.push(
        analyzeContextCommand,
        suggestBestPracticesCommand,
        getContextAwareDocsCommand,
        analyzeErrorsCommand,
        analyzeStackCommand,
        startServerCommand,
        stopServerCommand
    );
}

function setupEventListeners(context: vscode.ExtensionContext) {
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

    context.subscriptions.push(
        onDiagnosticsChange,
        onActiveEditorChange,
        onWorkspaceFoldersChange
    );
}

async function buildVSCodeContext(): Promise<VSCodeMCPContext> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    const workspaceRoot = workspaceFolders?.[0]?.uri.fsPath || '';
    
    const activeFiles = vscode.workspace.textDocuments.map(doc => doc.fileName);
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
        userIntent: 'debugging', // Could be enhanced with ML
        recentPrompts,
        projectType,
        detectedStack
    };
}

async function detectProjectTypeAndStack(workspaceRoot: string): Promise<{projectType: ProjectType; detectedStack: TechStack}> {
    try {
        // Use the comprehensive stack analyzer
        return await stackAnalyzer.analyzeFullStack();
    } catch (error) {
        console.log('Could not analyze project stack:', error);
        return {
            projectType: 'generic',
            detectedStack: {
                frontend: [], backend: [], databases: [], cloud: [], aiml: [],
                gpu: [], embedded: [], systems: [], scientific: [], gaming: [],
                mobile: [], web3: []
            }
        };
    }
}

async function showSuggestionsPanel(suggestions: AutoMCPSuggestion[], title: string = 'MCP Suggestions') {
    if (suggestions.length === 0) {
        vscode.window.showInformationMessage('No MCP suggestions available for current context');
        return;
    }

    const selected = await vscode.window.showQuickPick(
        suggestions.map(s => ({
            label: `$(${s.priority === 'high' ? 'warning' : 'info'}) ${s.tool}`,
            description: s.reasoning,
            detail: `Confidence: ${(s.confidence * 100).toFixed(0)}% | Expected: ${s.expectedOutput}`,
            suggestion: s
        })),
        { 
            placeHolder: `${title} - Select an MCP tool to execute`,
            matchOnDescription: true,
            matchOnDetail: true
        }
    );

    if (selected) {
        try {
            statusBarManager.updateStatus('executing', `Executing ${selected.suggestion.tool}...`);
            const result = await mcpServerManager.callMCPTool(selected.suggestion.tool, selected.suggestion.args);
            await showResultPanel(`${selected.suggestion.tool} Result`, result);
            statusBarManager.updateStatus('ready', 'MCP tool executed successfully');
        } catch (error) {
            vscode.window.showErrorMessage(`MCP tool execution failed: ${error}`);
            statusBarManager.updateStatus('error', 'Execution failed');
        }
    }
}

function generateStackReport(projectType: ProjectType, stack: TechStack, mcpSuggestions: Array<{library: string; topics: string[]}>): string {
    let report = `# 🎯 Project Analysis Report\n\n`;
    report += `**Project Type:** \`${projectType}\`\n\n`;
    
    // Frontend Technologies
    if (stack.frontend.length > 0) {
        report += `## 🎨 Frontend Technologies\n`;
        stack.frontend.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Backend Technologies
    if (stack.backend.length > 0) {
        report += `## ⚙️ Backend Technologies\n`;
        stack.backend.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // GPU/CUDA Computing
    if (stack.gpu.length > 0) {
        report += `## 🚀 GPU/CUDA Computing\n`;
        stack.gpu.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // AI/ML Technologies
    if (stack.aiml.length > 0) {
        report += `## 🤖 AI/ML Technologies\n`;
        stack.aiml.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Embedded/Hardware
    if (stack.embedded.length > 0) {
        report += `## 🔌 Embedded/Hardware\n`;
        stack.embedded.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Systems Programming
    if (stack.systems.length > 0) {
        report += `## 🖥️ Systems Programming\n`;
        stack.systems.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Scientific Computing
    if (stack.scientific.length > 0) {
        report += `## 🔬 Scientific Computing\n`;
        stack.scientific.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Databases
    if (stack.databases.length > 0) {
        report += `## 🗄️ Databases\n`;
        stack.databases.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Cloud & DevOps
    if (stack.cloud.length > 0) {
        report += `## ☁️ Cloud & DevOps\n`;
        stack.cloud.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Gaming
    if (stack.gaming.length > 0) {
        report += `## 🎮 Game Development\n`;
        stack.gaming.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Mobile
    if (stack.mobile.length > 0) {
        report += `## 📱 Mobile Development\n`;
        stack.mobile.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // Web3/Blockchain
    if (stack.web3.length > 0) {
        report += `## ⛓️ Web3/Blockchain\n`;
        stack.web3.forEach(tech => report += `- ${tech}\n`);
        report += `\n`;
    }
    
    // MCP Documentation Suggestions
    if (mcpSuggestions.length > 0) {
        report += `## 📚 Recommended MCP Documentation\n`;
        mcpSuggestions.forEach(suggestion => {
            report += `### ${suggestion.library}\n`;
            suggestion.topics.forEach(topic => report += `- ${topic}\n`);
            report += `\n`;
        });
    }
    
    // Context-Aware Recommendations
    report += `## 💡 Context-Aware Recommendations\n`;
    
    if (projectType === 'cuda-gpu-computing') {
        report += `- Consider using \`nvcc\` compiler optimizations\n`;
        report += `- Profile GPU memory usage with \`nvidia-smi\` and \`nvprof\`\n`;
        report += `- Leverage CUDA streams for concurrent execution\n`;
        report += `- Consider TensorRT for inference optimization\n`;
    }
    
    if (projectType === 'electrical-embedded') {
        report += `- Set up cross-compilation toolchain\n`;
        report += `- Consider real-time constraints and RTOS usage\n`;
        report += `- Implement proper interrupt handling\n`;
        report += `- Use hardware abstraction layers (HAL)\n`;
    }
    
    if (projectType === 'ml-ai-research') {
        report += `- Set up experiment tracking (MLflow, Weights & Biases)\n`;
        report += `- Consider model versioning and reproducibility\n`;
        report += `- Implement proper data validation and monitoring\n`;
        report += `- Use distributed training for large models\n`;
    }
    
    if (projectType === 'sveltekit-legal-ai') {
        report += `- Implement proper TypeScript configurations\n`;
        report += `- Set up vector database optimization\n`;
        report += `- Consider legal data privacy compliance\n`;
        report += `- Implement proper error handling and logging\n`;
    }
    
    report += `\n---\n`;
    report += `*Generated by Context7 MCP Assistant - ${new Date().toLocaleString()}*`;
    
    return report;
}

async function showResultPanel(title: string, result: any) {
    const panel = vscode.window.createWebviewPanel(
        'mcpResult',
        title,
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true
        }
    );

    const resultText = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
    
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