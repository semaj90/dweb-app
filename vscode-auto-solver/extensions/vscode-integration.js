/**
 * 🔧 VS Code Extension Integration
 * Real-time problem detection and auto-solving with cluster processing
 */

import * as vscode from 'vscode';
import { MultiCoreClusterManager } from '../core/multi-core-solver.js';

class VSCodeAutoSolver {
    constructor() {
        this.solver = null;
        this.diagnosticsCollection = null;
        this.statusBarItem = null;
        this.isProcessing = false;
        this.problemQueue = [];
        this.solverMetrics = {
            problemsSolved: 0,
            processingTime: 0,
            successRate: 0
        };
    }

    async activate(context) {
        console.log('🚀 VS Code Auto-Solver activating...');

        // Initialize multi-core solver
        this.solver = new MultiCoreClusterManager();
        await this.solver.initializeCluster();

        // Create diagnostics collection
        this.diagnosticsCollection = vscode.languages.createDiagnosticCollection('auto-solver');
        context.subscriptions.push(this.diagnosticsCollection);

        // Create status bar item
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
        this.statusBarItem.text = '🧠 Auto-Solver: Ready';
        this.statusBarItem.tooltip = 'Multi-Core Auto-Solver Status';
        this.statusBarItem.show();
        context.subscriptions.push(this.statusBarItem);

        // Register commands
        this.registerCommands(context);

        // Set up event listeners
        this.setupEventListeners(context);

        // Start background processing
        this.startBackgroundProcessing();

        console.log('✅ VS Code Auto-Solver activated!');
    }

    registerCommands(context) {
        // Auto-solve current file command
        const autoSolveCommand = vscode.commands.registerCommand('auto-solver.solveCurrent', async () => {
            await this.solvCurrentFile();
        });

        // Auto-solve workspace command  
        const autoSolveWorkspaceCommand = vscode.commands.registerCommand('auto-solver.solveWorkspace', async () => {
            await this.solveWorkspace();
        });

        // Show metrics command
        const showMetricsCommand = vscode.commands.registerCommand('auto-solver.showMetrics', () => {
            this.showMetricsPanel();
        });

        // Toggle auto-solve command
        const toggleAutoSolveCommand = vscode.commands.registerCommand('auto-solver.toggleAutoSolve', () => {
            this.toggleAutoSolve();
        });

        // Restart solver command
        const restartSolverCommand = vscode.commands.registerCommand('auto-solver.restart', async () => {
            await this.restartSolver();
        });

        context.subscriptions.push(
            autoSolveCommand,
            autoSolveWorkspaceCommand,
            showMetricsCommand,
            toggleAutoSolveCommand,
            restartSolverCommand
        );
    }

    setupEventListeners(context) {
        // Listen to diagnostics changes
        vscode.languages.onDidChangeDiagnostics(async (event) => {
            if (this.isAutoSolveEnabled()) {
                for (const uri of event.uris) {
                    await this.queueProblemAnalysis(uri);
                }
            }
        });

        // Listen to text document changes
        vscode.workspace.onDidChangeTextDocument(async (event) => {
            if (this.isAutoSolveEnabled() && this.shouldProcessDocument(event.document)) {
                // Debounce the processing
                this.debounceProcessing(event.document.uri);
            }
        });

        // Listen to active editor changes
        vscode.window.onDidChangeActiveTextEditor(async (editor) => {
            if (editor && this.isAutoSolveEnabled()) {
                await this.analyzeActiveEditor(editor);
            }
        });

        // Listen to configuration changes
        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('autoSolver')) {
                this.updateConfiguration();
            }
        });
    }

    async solvCurrentFile() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        this.statusBarItem.text = '🧠 Auto-Solver: Processing...';
        this.isProcessing = true;

        try {
            const document = editor.document;
            const problems = await this.extractProblemsFromDocument(document);
            
            if (problems.length === 0) {
                vscode.window.showInformationMessage('No problems found in current file');
                return;
            }

            const startTime = performance.now();
            const results = await this.solver.processProblemBatch(problems);
            const processingTime = performance.now() - startTime;

            // Apply solutions
            await this.applySolutions(document, results.results);

            // Update metrics
            this.updateMetrics(results.results.length, processingTime, results.results);

            vscode.window.showInformationMessage(
                `🎉 Processed ${problems.length} problems in ${processingTime.toFixed(2)}ms`
            );

        } catch (error) {
            console.error('❌ Error solving current file:', error);
            vscode.window.showErrorMessage(`Auto-Solver error: ${error.message}`);
        } finally {
            this.isProcessing = false;
            this.updateStatusBar();
        }
    }

    async solveWorkspace() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders || workspaceFolders.length === 0) {
            vscode.window.showWarningMessage('No workspace folder found');
            return;
        }

        this.statusBarItem.text = '🧠 Auto-Solver: Processing Workspace...';
        this.isProcessing = true;

        try {
            // Show progress
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Auto-Solving Workspace',
                cancellable: true
            }, async (progress, token) => {
                const allProblems = [];
                
                // Find all TypeScript/JavaScript files
                const filePattern = '**/*.{ts,tsx,js,jsx,svelte}';
                const files = await vscode.workspace.findFiles(filePattern, '**/node_modules/**');

                progress.report({ message: `Found ${files.length} files to analyze` });

                // Process files in batches
                const batchSize = 10;
                for (let i = 0; i < files.length; i += batchSize) {
                    if (token.isCancellationRequested) break;

                    const batch = files.slice(i, i + batchSize);
                    const batchProblems = [];

                    for (const uri of batch) {
                        const document = await vscode.workspace.openTextDocument(uri);
                        const problems = await this.extractProblemsFromDocument(document);
                        batchProblems.push(...problems);
                    }

                    allProblems.push(...batchProblems);
                    
                    progress.report({ 
                        increment: (batchSize / files.length) * 100,
                        message: `Analyzed ${Math.min(i + batchSize, files.length)} / ${files.length} files`
                    });
                }

                if (allProblems.length === 0) {
                    vscode.window.showInformationMessage('No problems found in workspace');
                    return;
                }

                progress.report({ message: `Processing ${allProblems.length} problems...` });

                const startTime = performance.now();
                const results = await this.solver.processProblemBatch(allProblems);
                const processingTime = performance.now() - startTime;

                // Apply solutions to documents
                const solvedCount = await this.applySolutionsBatch(results.results);

                this.updateMetrics(allProblems.length, processingTime, results.results);

                vscode.window.showInformationMessage(
                    `🎉 Processed ${allProblems.length} problems, solved ${solvedCount} in ${processingTime.toFixed(2)}ms`
                );
            });

        } catch (error) {
            console.error('❌ Error solving workspace:', error);
            vscode.window.showErrorMessage(`Workspace Auto-Solver error: ${error.message}`);
        } finally {
            this.isProcessing = false;
            this.updateStatusBar();
        }
    }

    async extractProblemsFromDocument(document) {
        const problems = [];
        const diagnostics = vscode.languages.getDiagnostics(document.uri);

        for (const diagnostic of diagnostics) {
            // Skip info-level diagnostics
            if (diagnostic.severity === vscode.DiagnosticSeverity.Information) continue;

            const problemRange = diagnostic.range;
            const problemText = document.getText(problemRange);
            const contextRange = new vscode.Range(
                Math.max(0, problemRange.start.line - 5),
                0,
                Math.min(document.lineCount - 1, problemRange.end.line + 5),
                document.lineAt(Math.min(document.lineCount - 1, problemRange.end.line + 5)).text.length
            );
            const contextText = document.getText(contextRange);

            problems.push({
                filePath: document.fileName,
                content: contextText,
                problemText: problemText,
                diagnostic: {
                    message: diagnostic.message,
                    severity: diagnostic.severity,
                    source: diagnostic.source,
                    code: diagnostic.code,
                    range: {
                        start: { line: problemRange.start.line, character: problemRange.start.character },
                        end: { line: problemRange.end.line, character: problemRange.end.character }
                    }
                },
                language: document.languageId,
                timestamp: Date.now()
            });
        }

        // Also analyze the full document for potential issues
        if (problems.length === 0) {
            // Add general analysis of the document
            problems.push({
                filePath: document.fileName,
                content: document.getText(),
                problemText: '',
                diagnostic: null,
                language: document.languageId,
                timestamp: Date.now(),
                isGeneralAnalysis: true
            });
        }

        return problems;
    }

    async applySolutions(document, results) {
        const edit = new vscode.WorkspaceEdit();
        let appliedCount = 0;

        for (const result of results) {
            if (result.error || !result.suggestedSolutions) continue;

            for (const solution of result.suggestedSolutions) {
                if (solution.confidence < 0.7) continue; // Only apply high-confidence solutions

                const applied = await this.applySingleSolution(document, result, solution, edit);
                if (applied) appliedCount++;
            }
        }

        if (edit.size > 0) {
            const success = await vscode.workspace.applyEdit(edit);
            if (success) {
                vscode.window.showInformationMessage(`Applied ${appliedCount} solutions`);
            } else {
                vscode.window.showWarningMessage('Failed to apply some solutions');
            }
        }

        return appliedCount;
    }

    async applySingleSolution(document, problemResult, solution, edit) {
        // Implementation depends on solution type
        switch (solution.type) {
            case 'syntax-error':
                return this.applySyntaxSolution(document, problemResult, solution, edit);
            case 'type-error':
                return this.applyTypeSolution(document, problemResult, solution, edit);
            case 'import-error':
                return this.applyImportSolution(document, problemResult, solution, edit);
            default:
                return false;
        }
    }

    async applySyntaxSolution(document, problemResult, solution, edit) {
        // Example: Add missing semicolon
        if (solution.template.actions.includes('Verify semicolons')) {
            const diagnostic = problemResult.diagnostic;
            if (diagnostic && diagnostic.message.includes('Expected')) {
                const range = new vscode.Range(
                    diagnostic.range.end.line,
                    diagnostic.range.end.character,
                    diagnostic.range.end.line,
                    diagnostic.range.end.character
                );
                edit.insert(document.uri, range.end, ';');
                return true;
            }
        }
        return false;
    }

    async applyTypeSolution(document, problemResult, solution, edit) {
        // Example: Add type annotation
        if (solution.template.actions.includes('Add type annotations')) {
            // This is a simplified example - real implementation would be more sophisticated
            const diagnostic = problemResult.diagnostic;
            if (diagnostic && diagnostic.message.includes('implicitly has an \'any\' type')) {
                const range = new vscode.Range(
                    diagnostic.range.start.line,
                    diagnostic.range.start.character,
                    diagnostic.range.end.line,
                    diagnostic.range.end.character
                );
                const variableText = document.getText(range);
                edit.replace(document.uri, range, `${variableText}: any`);
                return true;
            }
        }
        return false;
    }

    async applyImportSolution(document, problemResult, solution, edit) {
        // Example: Suggest package installation
        if (solution.template.actions.includes('Install missing package')) {
            const diagnostic = problemResult.diagnostic;
            if (diagnostic && diagnostic.message.includes('Cannot find module')) {
                // Show a code action to install the package
                vscode.window.showInformationMessage(
                    `Missing package detected: ${diagnostic.message}`,
                    'Install Package'
                ).then(selection => {
                    if (selection === 'Install Package') {
                        this.suggestPackageInstallation(diagnostic.message);
                    }
                });
                return true;
            }
        }
        return false;
    }

    async applySolutionsBatch(results) {
        let totalSolved = 0;
        const documentEdits = new Map();

        // Group results by document
        for (const result of results) {
            if (result.error || !result.suggestedSolutions) continue;

            const filePath = result.filePath;
            if (!documentEdits.has(filePath)) {
                documentEdits.set(filePath, []);
            }
            documentEdits.get(filePath).push(result);
        }

        // Apply edits for each document
        for (const [filePath, documentResults] of documentEdits) {
            try {
                const document = await vscode.workspace.openTextDocument(filePath);
                const solved = await this.applySolutions(document, documentResults);
                totalSolved += solved;
            } catch (error) {
                console.error(`Error applying solutions to ${filePath}:`, error);
            }
        }

        return totalSolved;
    }

    showMetricsPanel() {
        const metrics = this.solver.getMetrics();
        const panel = vscode.window.createWebviewPanel(
            'autoSolverMetrics',
            'Auto-Solver Metrics',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true
            }
        );

        panel.webview.html = this.generateMetricsHTML(metrics);
    }

    generateMetricsHTML(metrics) {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Auto-Solver Metrics</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; }
                .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
                .metric-title { font-weight: bold; color: #007acc; }
                .metric-value { font-size: 1.2em; margin: 5px 0; }
                .worker-status { display: flex; gap: 10px; flex-wrap: wrap; }
                .worker { padding: 5px 10px; background: #e8f4fd; border-radius: 3px; }
                .worker.busy { background: #ffeaa7; }
                .gpu-status { color: ${metrics.gpuAvailable ? '#00b894' : '#636e72'}; }
            </style>
        </head>
        <body>
            <h1>🧠 Multi-Core Auto-Solver Metrics</h1>
            
            <div class="metric">
                <div class="metric-title">Performance</div>
                <div class="metric-value">Total Tasks: ${metrics.totalTasks}</div>
                <div class="metric-value">Completed: ${metrics.completedTasks}</div>
                <div class="metric-value">Success Rate: ${((metrics.completedTasks / metrics.totalTasks) * 100 || 0).toFixed(1)}%</div>
                <div class="metric-value">Average Processing Time: ${metrics.averageProcessingTime.toFixed(2)}ms</div>
            </div>

            <div class="metric">
                <div class="metric-title">Workers Status</div>
                <div class="worker-status">
                    ${metrics.workers.map(w => 
                        `<div class="worker ${w.busy ? 'busy' : ''}">${w.id}: ${w.busy ? 'Busy' : 'Idle'} (${w.taskCount})</div>`
                    ).join('')}
                </div>
            </div>

            <div class="metric">
                <div class="metric-title">System Resources</div>
                <div class="metric-value">Memory Usage: ${(metrics.memoryTotal.heapUsed / 1024 / 1024).toFixed(2)} MB</div>
                <div class="metric-value">Memory Total: ${(metrics.memoryTotal.heapTotal / 1024 / 1024).toFixed(2)} MB</div>
                <div class="metric-value gpu-status">GPU: ${metrics.gpuAvailable ? 'Available ✅' : 'Not Available ❌'}</div>
            </div>

            <div class="metric">
                <div class="metric-title">Session Metrics</div>
                <div class="metric-value">Problems Solved: ${this.solverMetrics.problemsSolved}</div>
                <div class="metric-value">Total Processing Time: ${this.solverMetrics.processingTime.toFixed(2)}ms</div>
                <div class="metric-value">Success Rate: ${this.solverMetrics.successRate.toFixed(1)}%</div>
            </div>
        </body>
        </html>`;
    }

    startBackgroundProcessing() {
        // Process queued problems every 5 seconds
        setInterval(async () => {
            if (this.problemQueue.length > 0 && !this.isProcessing) {
                const problems = this.problemQueue.splice(0, 10); // Process up to 10 problems at once
                await this.processProblemsBackground(problems);
            }
        }, 5000);
    }

    async processProblemsBackground(problems) {
        try {
            this.isProcessing = true;
            this.statusBarItem.text = '🧠 Auto-Solver: Background Processing...';
            
            const results = await this.solver.processProblemBatch(problems);
            
            // Apply automatic solutions for high-confidence fixes
            for (const result of results.results) {
                if (result.suggestedSolutions) {
                    const highConfidenceSolutions = result.suggestedSolutions.filter(s => s.confidence > 0.9);
                    if (highConfidenceSolutions.length > 0) {
                        // Auto-apply very high confidence solutions
                        const document = await vscode.workspace.openTextDocument(result.filePath);
                        await this.applySolutions(document, [result]);
                    }
                }
            }
            
        } catch (error) {
            console.error('Background processing error:', error);
        } finally {
            this.isProcessing = false;
            this.updateStatusBar();
        }
    }

    async queueProblemAnalysis(uri) {
        try {
            const document = await vscode.workspace.openTextDocument(uri);
            const problems = await this.extractProblemsFromDocument(document);
            this.problemQueue.push(...problems);
        } catch (error) {
            console.error('Error queuing problem analysis:', error);
        }
    }

    debounceProcessing(uri) {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        this.debounceTimer = setTimeout(() => {
            this.queueProblemAnalysis(uri);
        }, 2000); // 2 second debounce
    }

    shouldProcessDocument(document) {
        const supportedLanguages = ['typescript', 'javascript', 'svelte', 'typescriptreact', 'javascriptreact'];
        return supportedLanguages.includes(document.languageId) && 
               !document.fileName.includes('node_modules');
    }

    isAutoSolveEnabled() {
        const config = vscode.workspace.getConfiguration('autoSolver');
        return config.get('enabled', true);
    }

    toggleAutoSolve() {
        const config = vscode.workspace.getConfiguration('autoSolver');
        const currentValue = config.get('enabled', true);
        config.update('enabled', !currentValue, vscode.ConfigurationTarget.Global);
        
        vscode.window.showInformationMessage(
            `Auto-Solver ${!currentValue ? 'enabled' : 'disabled'}`
        );
        this.updateStatusBar();
    }

    async restartSolver() {
        vscode.window.showInformationMessage('Restarting Auto-Solver...');
        
        this.solver = new MultiCoreClusterManager();
        await this.solver.initializeCluster();
        
        vscode.window.showInformationMessage('Auto-Solver restarted successfully!');
        this.updateStatusBar();
    }

    updateStatusBar() {
        if (!this.isAutoSolveEnabled()) {
            this.statusBarItem.text = '🧠 Auto-Solver: Disabled';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        } else if (this.isProcessing) {
            this.statusBarItem.text = '🧠 Auto-Solver: Processing...';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.prominentBackground');
        } else {
            this.statusBarItem.text = `🧠 Auto-Solver: Ready (${this.solverMetrics.problemsSolved} solved)`;
            this.statusBarItem.backgroundColor = undefined;
        }
    }

    updateConfiguration() {
        // Handle configuration changes
        const config = vscode.workspace.getConfiguration('autoSolver');
        // Update solver configuration if needed
    }

    updateMetrics(problemsCount, processingTime, results) {
        this.solverMetrics.problemsSolved += results.filter(r => !r.error && r.suggestedSolutions?.length > 0).length;
        this.solverMetrics.processingTime += processingTime;
        this.solverMetrics.successRate = (this.solverMetrics.problemsSolved / (this.solverMetrics.problemsSolved + results.filter(r => r.error).length)) * 100;
        this.updateStatusBar();
    }

    suggestPackageInstallation(message) {
        // Extract package name from error message
        const match = message.match(/Cannot find module '([^']+)'/);
        if (match) {
            const packageName = match[1];
            const terminal = vscode.window.createTerminal('Auto-Solver Package Install');
            terminal.sendText(`npm install ${packageName}`);
            terminal.show();
        }
    }

    async analyzeActiveEditor(editor) {
        // Analyze the active editor for potential problems
        const problems = await this.extractProblemsFromDocument(editor.document);
        if (problems.length > 0) {
            this.problemQueue.push(...problems);
        }
    }

    deactivate() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        if (this.solver) {
            // Cleanup solver resources
        }
        
        console.log('🛑 VS Code Auto-Solver deactivated');
    }
}

// Export activation function
export function activate(context) {
    const autoSolver = new VSCodeAutoSolver();
    return autoSolver.activate(context);
}

export function deactivate() {
    // Extension cleanup
}