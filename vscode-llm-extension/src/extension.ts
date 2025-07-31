import * as vscode from "vscode";
import { clusterManager, type WorkerTask } from "./cluster-manager";
import { ollamaGemmaCache } from "./ollama-gemma-cache";

export function activate(context: vscode.ExtensionContext) {
  // Initialize cluster and cache systems
  initializeExtensionSystems(context);

  // Register MCP Context7 commands
  registerMCPCommands(context);

  // Register LLM management commands
  registerLLMCommands(context);

  // Register cluster management commands
  registerClusterCommands(context);

  // Register cache management commands
  registerCacheCommands(context);
}

async function initializeExtensionSystems(context: vscode.ExtensionContext) {
  try {
    // Initialize cluster manager
    await clusterManager.initialize();
    
    // Initialize Ollama Gemma cache
    await ollamaGemmaCache.initialize();
    
    // Pre-cache workspace if enabled
    const config = vscode.workspace.getConfiguration('mcpContext7');
    if (config.get('enableWorkspacePreCache', true)) {
      vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Pre-caching workspace with Ollama Gemma...",
        cancellable: false
      }, async (progress) => {
        const result = await ollamaGemmaCache.preCacheWorkspace();
        vscode.window.showInformationMessage(
          `Workspace pre-cached: ${result.filesProcessed} files, ${result.embeddingsGenerated} embeddings generated`
        );
      });
    }

    vscode.window.showInformationMessage("MCP Context7 Extension initialized successfully!");
    
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to initialize extension: ${error}`);
    console.error('Extension initialization failed:', error);
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

// ========================================
// Command Implementations
// ========================================

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
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: "Analyzing current context with Context7 MCP...",
      cancellable: true
    }, async (progress, token) => {
      
      // Step 1: Generate/retrieve embeddings
      progress.report({ increment: 20, message: "Generating embeddings..." });
      
      const context = workspaceFolder?.name || 'unknown';
      await ollamaGemmaCache.getEmbedding(text, `file_${fileName}`);

      // Step 2: Analyze with Context7 using cluster
      progress.report({ increment: 30, message: "Running Context7 analysis..." });
      
      const analysisTask: WorkerTask = {
        id: `analyze_${Date.now()}`,
        type: 'mcp-analyze',
        data: {
          component: detectComponent(text, fileName),
          context: 'legal-ai',
          fileContent: text,
          fileName: fileName
        },
        priority: 'high',
        timeout: 30000
      };

      const analysisResult = await clusterManager.executeTask(analysisTask);

      // Step 3: Find similar contexts
      progress.report({ increment: 25, message: "Finding similar contexts..." });
      
      const similarContexts = await ollamaGemmaCache.querySimilar({
        text: text.substring(0, 1000), // First 1000 chars for similarity
        context: `file_${fileName}`,
        similarityThreshold: 0.7,
        maxResults: 5
      });

      // Step 4: Generate recommendations
      progress.report({ increment: 25, message: "Generating recommendations..." });

      // Create and show results panel
      const panel = vscode.window.createWebviewPanel(
        'contextAnalysis',
        'Context7 Analysis Results',
        vscode.ViewColumn.Beside,
        {
          enableScripts: true,
          retainContextWhenHidden: true
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
          case 'runAutoFix':
            await runAutoFixFromAnalysis(message.area);
            break;
          case 'generateBestPractices':
            await generateBestPracticesFromAnalysis(message.area);
            break;
        }
      });
    });

  } catch (error) {
    vscode.window.showErrorMessage(`Context analysis failed: ${error}`);
    console.error('Context analysis error:', error);
  }
}

/**
 * Auto-fix command with cluster support
 */
async function runAutoFixCommand(): Promise<void> {
  try {
    // Show quick pick for fix area
    const fixArea = await vscode.window.showQuickPick([
      { label: 'All Areas', value: null },
      { label: 'Imports & Exports', value: 'imports' },
      { label: 'Svelte 5 Patterns', value: 'svelte5' },
      { label: 'TypeScript', value: 'typescript' },
      { label: 'Performance', value: 'performance' },
      { label: 'Accessibility', value: 'accessibility' },
      { label: 'Security', value: 'security' }
    ], {
      placeHolder: 'Select area to fix'
    });

    if (!fixArea) return;

    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: `Running auto-fix${fixArea.value ? ` for ${fixArea.label}` : ''}...`,
      cancellable: false
    }, async (progress) => {
      const autoFixTask: WorkerTask = {
        id: `autofix_${Date.now()}`,
        type: 'auto-fix',
        data: {
          area: fixArea.value,
          dryRun: false
        },
        priority: 'high',
        timeout: 60000
      };

      const result = await clusterManager.executeTask(autoFixTask);

      if (result.success) {
        const summary = result.result;
        vscode.window.showInformationMessage(
          `Auto-fix complete: ${summary.summary.filesFixed} files fixed, ${summary.summary.totalIssues} issues resolved`
        );

        // Show detailed results if requested
        const showDetails = await vscode.window.showInformationMessage(
          'Auto-fix completed successfully!',
          'Show Details'
        );

        if (showDetails) {
          const panel = vscode.window.createWebviewPanel(
            'autoFixResults',
            'Auto-Fix Results',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
          );

          panel.webview.html = generateAutoFixWebviewContent(summary);
        }
      } else {
        vscode.window.showErrorMessage(`Auto-fix failed: ${result.error}`);
      }
    });

  } catch (error) {
    vscode.window.showErrorMessage(`Auto-fix command failed: ${error}`);
  }
}

/**
 * Generate best practices command
 */
async function generateBestPracticesCommand(): Promise<void> {
  const area = await vscode.window.showQuickPick([
    { label: 'Performance', value: 'performance' },
    { label: 'Security', value: 'security' },
    { label: 'UI/UX', value: 'ui-ux' }
  ], {
    placeHolder: 'Select area for best practices'
  });

  if (!area) return;

  try {
    const task: WorkerTask = {
      id: `best_practices_${Date.now()}`,
      type: 'mcp-analyze',
      data: {
        component: 'best-practices',
        context: 'legal-ai',
        area: area.value
      },
      priority: 'medium'
    };

    const result = await clusterManager.executeTask(task);
    
    if (result.success) {
      // Show results in new document
      const doc = await vscode.workspace.openTextDocument({
        content: `# ${area.label} Best Practices\n\n${JSON.stringify(result.result, null, 2)}`,
        language: 'markdown'
      });
      
      await vscode.window.showTextDocument(doc);
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to generate best practices: ${error}`);
  }
}

/**
 * Semantic search command with caching
 */
async function semanticSearchCommand(): Promise<void> {
  const query = await vscode.window.showInputBox({
    placeHolder: 'Enter search query...',
    prompt: 'Search workspace using semantic embeddings'
  });

  if (!query) return;

  try {
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: "Searching workspace...",
      cancellable: false
    }, async (progress) => {
      // Use cached embeddings for search
      const results = await ollamaGemmaCache.querySimilar({
        text: query,
        context: 'workspace_search',
        similarityThreshold: 0.6,
        maxResults: 10
      });

      // Show results panel
      const panel = vscode.window.createWebviewPanel(
        'semanticSearch',
        'Semantic Search Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
      );

      panel.webview.html = generateSearchResultsWebviewContent(query, results);
    });

  } catch (error) {
    vscode.window.showErrorMessage(`Semantic search failed: ${error}`);
  }
}

/**
 * Agent orchestration command
 */
async function orchestrateAgentsCommand(): Promise<void> {
  const prompt = await vscode.window.showInputBox({
    placeHolder: 'Enter orchestration prompt...',
    prompt: 'Enter task for agent orchestration'
  });

  if (!prompt) return;

  const agents = await vscode.window.showQuickPick([
    { label: 'Claude + AutoGen + CrewAI', value: ['claude', 'autogen', 'crewai'] },
    { label: 'Claude Only', value: ['claude'] },
    { label: 'AutoGen + CrewAI', value: ['autogen', 'crewai'] },
    { label: 'All + RAG', value: ['claude', 'autogen', 'crewai', 'rag'] }
  ], {
    placeHolder: 'Select agents to orchestrate'
  });

  if (!agents) return;

  try {
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: "Orchestrating agents...",
      cancellable: false
    }, async (progress) => {
      // Use cluster to orchestrate agents
      const task: WorkerTask = {
        id: `orchestrate_${Date.now()}`,
        type: 'agent-orchestrate',
        data: {
          prompt,
          agents: agents.value,
          options: {
            includeContext7: true,
            autoFix: false,
            parallel: true
          }
        },
        priority: 'high',
        timeout: 60000
      };

      const result = await clusterManager.executeTask(task);

      if (result.success) {
        // Show orchestration results
        const panel = vscode.window.createWebviewPanel(
          'agentOrchestration',
          'Agent Orchestration Results',
          vscode.ViewColumn.Beside,
          { enableScripts: true }
        );

        panel.webview.html = generateOrchestrationWebviewContent(result.result);
      } else {
        vscode.window.showErrorMessage(`Agent orchestration failed: ${result.error}`);
      }
    });

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
    const response = await fetch('http://localhost:11434/api/tags');
    if (response.ok) {
      const data = await response.json();
      const models = data.models?.map((m: any) => m.name) || [];
      
      vscode.window.showInformationMessage(
        `Found ${models.length} models: ${models.slice(0, 3).join(', ')}${models.length > 3 ? '...' : ''}`
      );
    } else {
      vscode.window.showWarningMessage("Ollama not available. Please ensure Ollama is running.");
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
      'clusterStatus',
      'Cluster Status',
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
      'Are you sure you want to restart the cluster?',
      'Yes', 'No'
    );

    if (confirm === 'Yes') {
      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Restarting cluster...",
        cancellable: false
      }, async (progress) => {
        await clusterManager.shutdown();
        await clusterManager.initialize();
        vscode.window.showInformationMessage("Cluster restarted successfully");
      });
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
      'cacheStats',
      'Cache Statistics',
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
      'Are you sure you want to clear the embedding cache?',
      'Yes', 'No'
    );

    if (confirm === 'Yes') {
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
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: "Pre-caching workspace...",
      cancellable: false
    }, async (progress) => {
      const result = await ollamaGemmaCache.preCacheWorkspace();
      vscode.window.showInformationMessage(
        `Pre-caching complete: ${result.filesProcessed} files, ${result.embeddingsGenerated} new embeddings`
      );
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Pre-caching failed: ${error}`);
  }
}

// ========================================
// Helper Functions
// ========================================

/**
 * Detect component type from file content and name
 */
function detectComponent(text: string, fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase();
  
  if (ext === 'svelte') return 'svelte';
  if (ext === 'ts' || ext === 'js') {
    if (text.includes('SvelteKit') || text.includes('vscode.')) return 'typescript';
    if (text.includes('React')) return 'react';
    return 'javascript';
  }
  if (ext === 'css' || text.includes('@apply')) return 'css';
  if (ext === 'md') return 'markdown';
  if (ext === 'json') return 'json';
  
  return 'unknown';
}

/**
 * Generate analysis webview content
 */
function generateAnalysisWebviewContent(analysisResult: any, similarContexts: any, fileName: string): string {
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
        ${analysisResult.success ? `
          <h3>Recommendations:</h3>
          <ul>
            ${analysisResult.result?.recommendations?.map((r: string) => `<li>${r}</li>`).join('') || '<li>No specific recommendations</li>'}
          </ul>
          <button class="button" onclick="runAutoFix()">Run Auto-Fix</button>
          <button class="button" onclick="generateBestPractices()">Generate Best Practices</button>
        ` : `
          <p><strong>Error:</strong> ${analysisResult.error}</p>
        `}
      </div>

      <div class="section info">
        <h2>Similar Contexts (${similarContexts.similar.length})</h2>
        ${similarContexts.similar.map((item: any) => `
          <div class="similarity-item">
            <strong>Similarity:</strong> ${(item.similarity * 100).toFixed(1)}% | 
            <strong>Context:</strong> ${item.metadata.context} |
            <strong>Type:</strong> ${item.metadata.fileType}
            <p>${item.text.substring(0, 200)}...</p>
          </div>
        `).join('')}
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
      ${Object.entries(result.fixes).map(([area, fixes]: [string, any]) => `
        <h3>${area.charAt(0).toUpperCase() + area.slice(1)} (${fixes.length} fixes)</h3>
        ${fixtures.map((fix: any) => `
          <p><strong>${fix.file}:</strong></p>
          <ul>${fix.changes.map((change: string) => `<li>${change}</li>`).join('')}</ul>
        `).join('')}
      `).join('')}
      
      <h2>Recommendations</h2>
      <ul>
        ${result.recommendations.map((rec: string) => `<li>${rec}</li>`).join('')}
      </ul>
    </body>
    </html>
  `;
}

/**
 * Generate search results webview content
 */
function generateSearchResultsWebviewContent(query: string, results: any): string {
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
      
      ${results.similar.map((item: any) => `
        <div class="result">
          <div class="similarity">Similarity: ${(item.similarity * 100).toFixed(1)}%</div>
          <p><strong>Context:</strong> ${item.metadata.context}</p>
          <p><strong>Type:</strong> ${item.metadata.fileType}</p>
          <p>${item.text.substring(0, 300)}...</p>
        </div>
      `).join('')}
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
      ${result.results.map((agentResult: any) => `
        <h3>${agentResult.agent.toUpperCase()} (Score: ${agentResult.score})</h3>
        <p>${agentResult.output}</p>
        ${agentResult.error ? `<p style="color: red;">Error: ${agentResult.error}</p>` : ''}
      `).join('')}
      
      <h2>Recommendations</h2>
      <ul>
        ${result.synthesis.recommendations.map((rec: string) => `<li>${rec}</li>`).join('')}
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
      ${stats.workerStats.map((worker: any) => `
        <h3>Worker ${worker.workerId}</h3>
        <ul>
          <li>Status: ${worker.status}</li>
          <li>Tasks Processed: ${worker.tasksProcessed}</li>
          <li>Current Load: ${worker.currentLoad}</li>
          <li>Memory Usage: ${(worker.memoryUsage.heapUsed / 1024 / 1024).toFixed(2)} MB</li>
          <li>Uptime: ${worker.uptime}s</li>
        </ul>
      `).join('')}
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
  // Shutdown cluster gracefully
  clusterManager.shutdown().catch(console.error);
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
