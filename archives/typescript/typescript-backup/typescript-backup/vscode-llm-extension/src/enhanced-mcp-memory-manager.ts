/**
 * Enhanced MCP Extension Memory Manager
 * Advanced memory optimization for VS Code extensions with ML clustering
 */

import * as vscode from "vscode";
import { AdvancedMemoryOptimizer } from "../optimization/advanced-memory-optimizer.js";

export interface ExtensionMemoryMetrics {
  totalMemory: number;
  usedMemory: number;
  extensionMemory: number;
  cacheMemory: number;
  clusterMemory: number;
  somMemory: number;
  percentage: number;
  pressure: "low" | "medium" | "high" | "critical";
}

export interface CommandExecutionContext {
  commandId: string;
  args: unknown[];
  timestamp: number;
  memoryBefore: number;
  memoryAfter: number;
  duration: number;
  clusterId?: string;
  priority: number;
}

export interface MLModelCache {
  modelId: string;
  data: unknown;
  embeddings: number[];
  lastAccessed: number;
  frequency: number;
  size: number;
  clusterId?: string;
}

export class EnhancedMCPExtensionMemoryManager {
  private context: vscode.ExtensionContext;
  private memoryOptimizer: AdvancedMemoryOptimizer;
  private executionHistory: CommandExecutionContext[] = [];
  private modelCache = new Map<string, MLModelCache>();
  private commandClusters = new Map<string, any>();
  private currentPressure: ExtensionMemoryMetrics["pressure"] = "low";
  private optimizationInterval: NodeJS.Timeout | null = null;
  private statusBarItem: vscode.StatusBarItem;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.memoryOptimizer = new AdvancedMemoryOptimizer();

    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = "mcp.showMemoryStatus";
    this.statusBarItem.show();

    this.initialize();
  }

  /**
   * Initialize the memory manager
   */
  private async initialize(): Promise<void> {
    console.log("🧠 Initializing Enhanced MCP Extension Memory Manager...");

    // Register commands
    this.registerCommands();

    // Start memory monitoring
    this.startMemoryMonitoring();

    // Load persisted execution history
    await this.loadExecutionHistory();

    // Perform initial clustering of commands
    await this.clusterCommands();

    console.log("✅ Enhanced MCP Extension Memory Manager initialized");
  }

  /**
   * Register VS Code commands for memory management
   */
  private registerCommands(): void {
    const commands = [
      {
        command: "mcp.showMemoryStatus",
        handler: () => this.showMemoryStatus(),
      },
      {
        command: "mcp.optimizeMemory",
        handler: () => this.optimizeMemoryNow(),
      },
      {
        command: "mcp.clearCache",
        handler: () => this.clearAllCaches(),
      },
      {
        command: "mcp.analyzeCommandClusters",
        handler: () => this.analyzeCommandClusters(),
      },
      {
        command: "mcp.exportMemoryReport",
        handler: () => this.exportMemoryReport(),
      },
    ];

    commands.forEach(({ command, handler }) => {
      const disposable = vscode.commands.registerCommand(command, handler);
      this.context.subscriptions.push(disposable);
    });
  }

  /**
   * Track command execution with clustering
   */
  async trackCommandExecution(
    commandId: string,
    args: unknown[],
    executor: () => Promise<any>
  ): Promise<any> {
    const startTime = Date.now();
    const memoryBefore = await this.getCurrentMemoryUsage();

    try {
      // Execute the command
      const result = await executor();

      // Record execution metrics
      const memoryAfter = await this.getCurrentMemoryUsage();
      const duration = Date.now() - startTime;

      const executionContext: CommandExecutionContext = {
        commandId,
        args,
        timestamp: startTime,
        memoryBefore,
        memoryAfter,
        duration,
        priority: this.calculateCommandPriority(
          commandId,
          duration,
          memoryAfter - memoryBefore
        ),
      };

      // Add to SOM cluster
      executionContext.clusterId = await this.clusterCommand(executionContext);

      this.executionHistory.push(executionContext);

      // Limit history size
      if (this.executionHistory.length > 1000) {
        this.executionHistory = this.executionHistory.slice(-500);
      }

      // Persist history periodically
      if (this.executionHistory.length % 50 === 0) {
        await this.persistExecutionHistory();
      }

      // Trigger optimization if memory pressure is high
      if (memoryAfter > memoryBefore * 1.2) {
        await this.handleMemoryIncrease(memoryAfter - memoryBefore);
      }

      return result;
    } catch (error) {
      // Still record failed executions for analysis
      const memoryAfter = await this.getCurrentMemoryUsage();
      const duration = Date.now() - startTime;

      this.executionHistory.push({
        commandId,
        args,
        timestamp: startTime,
        memoryBefore,
        memoryAfter,
        duration,
        priority: 0, // Failed commands have lowest priority
      });

      throw error;
    }
  }

  /**
   * Intelligent model caching with clustering
   */
  async cacheMLModel(
    modelId: string,
    data: unknown,
    embeddings: number[] = []
  ): Promise<void> {
    const size = JSON.stringify(data).length;

    // Check if we need to make space
    const currentCacheSize = Array.from(this.modelCache.values()).reduce(
      (total, model) => total + model.size,
      0
    );

    const maxCacheSize = 100 * 1024 * 1024; // 100MB

    if (currentCacheSize + size > maxCacheSize) {
      await this.evictLeastUsedModels(size);
    }

    // Cluster the model
    const clusterId = await this.memoryOptimizer.optimizedMemoryAllocation({
      content: modelId,
      embedding: embeddings,
      type: "ml_model",
      priority: 0.8,
      size,
    });

    const modelCache: MLModelCache = {
      modelId,
      data,
      embeddings,
      lastAccessed: Date.now(),
      frequency: 1,
      size,
      clusterId,
    };

    this.modelCache.set(modelId, modelCache);

    console.log(
      `📦 Cached ML model ${modelId} (${size} bytes) in cluster ${clusterId}`
    );
  }

  /**
   * Retrieve cached model with access tracking
   */
  getCachedMLModel(modelId: string): unknown | null {
    const model = this.modelCache.get(modelId);

    if (model) {
      model.lastAccessed = Date.now();
      model.frequency++;
      return model.data;
    }

    return null;
  }

  /**
   * Cluster commands using K-means and SOM
   */
  private async clusterCommands(): Promise<void> {
    if (this.executionHistory.length < 10) return;

    console.log("🔄 Clustering commands for optimization...");

    // Prepare data for clustering
    const commandData = this.executionHistory.map((cmd) => ({
      commandId: cmd.commandId,
      duration: cmd.duration,
      memoryUsage: cmd.memoryAfter - cmd.memoryBefore,
      frequency: this.executionHistory.filter(
        (c) => c.commandId === cmd.commandId
      ).length,
      timestamp: cmd.timestamp,
      embedding: this.createCommandEmbedding(cmd),
      clusterId: 0,
    }));

    // Perform K-means clustering
    const clusters = await this.memoryOptimizer.performKMeansClustering(
      commandData,
      5
    );

    // Store cluster results
    clusters.forEach((cluster) => {
      this.commandClusters.set(cluster.id, {
        ...cluster,
        commands: commandData.filter(
          (cmd) => cmd.clusterId === parseInt(cluster.id.split("_")[1])
        ),
      });
    });

    console.log(
      `✅ Clustered ${commandData.length} commands into ${clusters.length} clusters`
    );
  }

  /**
   * Assign command to appropriate cluster
   */
  private async clusterCommand(
    context: CommandExecutionContext
  ): Promise<string> {
    const embedding = this.createCommandEmbedding(context);

    return await this.memoryOptimizer.optimizedMemoryAllocation({
      content: context.commandId,
      embedding,
      type: "command",
      priority: context.priority / 10,
      size: JSON.stringify(context).length,
    });
  }

  /**
   * Create embedding for command
   */
  private createCommandEmbedding(context: CommandExecutionContext): number[] {
    // Simple embedding based on command characteristics
    const embedding = new Array(384).fill(0);

    // Command ID hash
    const idHash = this.simpleHash(context.commandId);
    embedding[0] = (idHash % 1000) / 1000;

    // Duration normalization
    embedding[1] = Math.min(context.duration / 10000, 1);

    // Memory usage normalization
    const memoryDelta = context.memoryAfter - context.memoryBefore;
    embedding[2] = Math.min(Math.abs(memoryDelta) / (10 * 1024 * 1024), 1);

    // Time of day
    const hour = new Date(context.timestamp).getHours();
    embedding[3] = hour / 24;

    // Frequency (calculated from history)
    const frequency = this.executionHistory.filter(
      (c) => c.commandId === context.commandId
    ).length;
    embedding[4] = Math.min(frequency / 100, 1);

    // Fill remaining with noise for better clustering
    for (let i = 5; i < 384; i++) {
      embedding[i] = (Math.random() - 0.5) * 0.1;
    }

    return embedding;
  }

  /**
   * Calculate command priority based on usage patterns
   */
  private calculateCommandPriority(
    commandId: string,
    duration: number,
    memoryDelta: number
  ): number {
    let priority = 5; // Base priority

    // Frequent commands get higher priority
    const frequency = this.executionHistory.filter(
      (c) => c.commandId === commandId
    ).length;
    priority += Math.min(frequency / 10, 3);

    // Fast commands get higher priority
    if (duration < 1000) priority += 2;
    else if (duration > 10000) priority -= 2;

    // Low memory usage gets higher priority
    if (memoryDelta < 1024 * 1024)
      priority += 1; // < 1MB
    else if (memoryDelta > 10 * 1024 * 1024) priority -= 3; // > 10MB

    return Math.max(0, Math.min(10, priority));
  }

  /**
   * Handle memory increase events
   */
  private async handleMemoryIncrease(delta: number): Promise<void> {
    const metrics = await this.getMemoryMetrics();

    if (metrics.pressure === "high" || metrics.pressure === "critical") {
      console.warn(`⚠️ Memory pressure ${metrics.pressure}, optimizing...`);
      await this.optimizeMemoryNow();
    }
  }

  /**
   * Evict least used models to make space
   */
  private async evictLeastUsedModels(spaceNeeded: number): Promise<void> {
    const models = Array.from(this.modelCache.values()).sort((a, b) => {
      // Sort by frequency and last accessed time
      const scoreA = a.frequency * (Date.now() - a.lastAccessed);
      const scoreB = b.frequency * (Date.now() - b.lastAccessed);
      return scoreA - scoreB;
    });

    let freedSpace = 0;
    const toRemove: string[] = [];

    for (const model of models) {
      if (freedSpace >= spaceNeeded) break;

      toRemove.push(model.modelId);
      freedSpace += model.size;
    }

    for (const modelId of toRemove) {
      this.modelCache.delete(modelId);
      console.log(`🗑️ Evicted model ${modelId} to free space`);
    }
  }

  /**
   * Get current memory metrics
   */
  private async getMemoryMetrics(): Promise<ExtensionMemoryMetrics> {
    const process = await import("process");
    const memoryUsage = process.memoryUsage();

    const totalMemory = memoryUsage.heapTotal;
    const usedMemory = memoryUsage.heapUsed;
    const extensionMemory = memoryUsage.external;

    const cacheMemory = Array.from(this.modelCache.values()).reduce(
      (total, model) => total + model.size,
      0
    );

    const clusterMemory = Array.from(this.commandClusters.values()).reduce(
      (total, cluster) => total + cluster.memoryUsage,
      0
    );

    const percentage = (usedMemory / totalMemory) * 100;

    let pressure: ExtensionMemoryMetrics["pressure"] = "low";
    if (percentage > 90) pressure = "critical";
    else if (percentage > 75) pressure = "high";
    else if (percentage > 50) pressure = "medium";

    this.currentPressure = pressure;

    return {
      totalMemory,
      usedMemory,
      extensionMemory,
      cacheMemory,
      clusterMemory,
      somMemory: 0, // Would be calculated from SOM network
      percentage,
      pressure,
    };
  }

  /**
   * Get current memory usage
   */
  private async getCurrentMemoryUsage(): Promise<number> {
    const process = await import("process");
    return process.memoryUsage().heapUsed;
  }

  /**
   * Start memory monitoring
   */
  private startMemoryMonitoring(): void {
    this.optimizationInterval = setInterval(async () => {
      const metrics = await this.getMemoryMetrics();

      // Update status bar
      this.updateStatusBar(metrics);

      // Trigger optimization if needed
      if (metrics.pressure === "high" || metrics.pressure === "critical") {
        await this.optimizeMemoryNow();
      }

      // Re-cluster commands periodically
      if (this.executionHistory.length % 100 === 0) {
        await this.clusterCommands();
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Update status bar with memory info
   */
  private updateStatusBar(metrics: ExtensionMemoryMetrics): void {
    const icon = this.getPressureIcon(metrics.pressure);
    const memoryMB = Math.round(metrics.usedMemory / (1024 * 1024));

    this.statusBarItem.text = `${icon} ${memoryMB}MB`;
    this.statusBarItem.tooltip = `Memory: ${metrics.percentage.toFixed(1)}% | Pressure: ${metrics.pressure}`;

    // Change color based on pressure
    switch (metrics.pressure) {
      case "critical":
        this.statusBarItem.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.errorBackground"
        );
        break;
      case "high":
        this.statusBarItem.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.warningBackground"
        );
        break;
      default:
        this.statusBarItem.backgroundColor = undefined;
    }
  }

  /**
   * Get pressure icon
   */
  private getPressureIcon(
    pressure: ExtensionMemoryMetrics["pressure"]
  ): string {
    switch (pressure) {
      case "critical":
        return "🚨";
      case "high":
        return "⚠️";
      case "medium":
        return "📊";
      default:
        return "🧠";
    }
  }

  /**
   * Show memory status in UI
   */
  private async showMemoryStatus(): Promise<void> {
    const metrics = await this.getMemoryMetrics();
    const optimizerStatus = this.memoryOptimizer.getOptimizationStatus();

    const panel = vscode.window.createWebviewPanel(
      "mcpMemoryStatus",
      "MCP Memory Status",
      vscode.ViewColumn.One,
      { enableScripts: true }
    );

    panel.webview.html = this.generateMemoryStatusHTML(
      metrics,
      optimizerStatus
    );
  }

  /**
   * Generate HTML for memory status
   */
  private generateMemoryStatusHTML(
    metrics: ExtensionMemoryMetrics,
    optimizerStatus: unknown
  ): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <title>MCP Memory Status</title>
        <style>
          body { font-family: var(--vscode-font-family); padding: 20px; }
          .metric { margin: 10px 0; padding: 10px; border: 1px solid var(--vscode-panel-border); }
          .critical { background-color: var(--vscode-errorBackground); }
          .high { background-color: var(--vscode-warningBackground); }
          .cluster { margin: 5px 0; padding: 5px; background-color: var(--vscode-editor-background); }
          .progress { width: 100%; height: 20px; background-color: var(--vscode-progressBar-background); }
          .progress-bar { height: 100%; background-color: var(--vscode-progressBar-foreground); }
        </style>
      </head>
      <body>
        <h1>🧠 MCP Extension Memory Status</h1>

        <div class="metric ${metrics.pressure}">
          <h3>Overall Memory Status</h3>
          <p>Pressure: <strong>${metrics.pressure.toUpperCase()}</strong></p>
          <p>Usage: ${Math.round(metrics.usedMemory / (1024 * 1024))}MB / ${Math.round(metrics.totalMemory / (1024 * 1024))}MB</p>
          <div class="progress">
            <div class="progress-bar" style="width: ${metrics.percentage}%"></div>
          </div>
        </div>

        <div class="metric">
          <h3>Cache Breakdown</h3>
          <p>Model Cache: ${Math.round(metrics.cacheMemory / (1024 * 1024))}MB</p>
          <p>Cluster Memory: ${Math.round(metrics.clusterMemory / (1024 * 1024))}MB</p>
          <p>Extension Memory: ${Math.round(metrics.extensionMemory / (1024 * 1024))}MB</p>
        </div>

        <div class="metric">
          <h3>Current LOD Level</h3>
          <p>Level: <strong>${optimizerStatus.currentLOD.detail.toUpperCase()}</strong></p>
          <p>Max Memory: ${optimizerStatus.currentLOD.maxMemoryMB}MB</p>
          <p>Max Objects: ${optimizerStatus.currentLOD.maxObjects}</p>
          <p>Quality: ${(optimizerStatus.currentLOD.quality * 100).toFixed(1)}%</p>
        </div>

        <div class="metric">
          <h3>Memory Pools</h3>
          ${optimizerStatus.pools
            .map(
              (pool: unknown) => `
            <div class="cluster">
              <strong>${pool.id}</strong>: ${pool.usage} (${pool.percentage.toFixed(1)}%) - ${pool.items} items
            </div>
          `
            )
            .join("")}
        </div>

        <div class="metric">
          <h3>Command Clusters</h3>
          ${Array.from(this.commandClusters.values())
            .map(
              (cluster: unknown) => `
            <div class="cluster">
              <strong>${cluster.id}</strong>: ${cluster.size} commands, Cohesion: ${(cluster.cohesion * 100).toFixed(1)}%
            </div>
          `
            )
            .join("")}
        </div>

        <div class="metric">
          <h3>Recent Commands (Last 10)</h3>
          ${this.executionHistory
            .slice(-10)
            .map(
              (cmd) => `
            <div class="cluster">
              <strong>${cmd.commandId}</strong> - ${cmd.duration}ms, ${Math.round((cmd.memoryAfter - cmd.memoryBefore) / 1024)}KB
              ${cmd.clusterId ? `(Cluster: ${cmd.clusterId})` : ""}
            </div>
          `
            )
            .join("")}
        </div>
      </body>
      </html>
    `;
  }

  /**
   * Optimize memory now
   */
  private async optimizeMemoryNow(): Promise<void> {
    console.log("⚡ Optimizing MCP extension memory...");

    // Clear old execution history
    if (this.executionHistory.length > 500) {
      this.executionHistory = this.executionHistory.slice(-250);
    }

    // Evict least used models
    await this.evictLeastUsedModels(50 * 1024 * 1024); // Free 50MB

    // Clear old command clusters
    const now = Date.now();
    for (const [clusterId, cluster] of this.commandClusters) {
      if (now - cluster.lastAccessed > 24 * 60 * 60 * 1000) {
        // 24 hours
        this.commandClusters.delete(clusterId);
      }
    }

    // Force garbage collection
    if (global.gc) {
      global.gc();
    }

    vscode.window.showInformationMessage("MCP Extension memory optimized");
  }

  /**
   * Clear all caches
   */
  private async clearAllCaches(): Promise<void> {
    this.modelCache.clear();
    this.commandClusters.clear();
    this.executionHistory = [];

    // Clear persisted data
    await this.context.workspaceState.update("executionHistory", undefined);
    await this.context.globalState.update("modelCache", undefined);

    if (global.gc) {
      global.gc();
    }

    vscode.window.showInformationMessage("All MCP caches cleared");
  }

  /**
   * Analyze command clusters
   */
  private async analyzeCommandClusters(): Promise<void> {
    await this.clusterCommands();

    const analysis = Array.from(this.commandClusters.values()).map(
      (cluster) => ({
        id: cluster.id,
        size: cluster.size,
        cohesion: cluster.cohesion,
        memoryUsage: cluster.memoryUsage,
        commands: cluster.commands?.map((cmd: unknown) => cmd.commandId) || [],
      })
    );

    const document = await vscode.workspace.openTextDocument({
      content: JSON.stringify(analysis, null, 2),
      language: "json",
    });

    await vscode.window.showTextDocument(document);
  }

  /**
   * Export memory report
   */
  private async exportMemoryReport(): Promise<void> {
    const metrics = await this.getMemoryMetrics();
    const optimizerStatus = this.memoryOptimizer.getOptimizationStatus();

    const report = {
      timestamp: new Date().toISOString(),
      metrics,
      optimizerStatus,
      executionHistory: this.executionHistory.slice(-100),
      commandClusters: Array.from(this.commandClusters.values()),
      modelCache: Array.from(this.modelCache.values()).map((model) => ({
        modelId: model.modelId,
        size: model.size,
        frequency: model.frequency,
        lastAccessed: new Date(model.lastAccessed).toISOString(),
        clusterId: model.clusterId,
      })),
    };

    const document = await vscode.workspace.openTextDocument({
      content: JSON.stringify(report, null, 2),
      language: "json",
    });

    await vscode.window.showTextDocument(document);
    vscode.window.showInformationMessage("Memory report exported");
  }

  /**
   * Load execution history from storage
   */
  private async loadExecutionHistory(): Promise<void> {
    const stored =
      this.context.workspaceState.get<CommandExecutionContext[]>(
        "executionHistory"
      );
    if (stored) {
      this.executionHistory = stored;
      console.log(`📚 Loaded ${stored.length} execution records from storage`);
    }
  }

  /**
   * Persist execution history to storage
   */
  private async persistExecutionHistory(): Promise<void> {
    // Only store last 500 records
    const toStore = this.executionHistory.slice(-500);
    await this.context.workspaceState.update("executionHistory", toStore);
  }

  /**
   * Simple hash function
   */
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.optimizationInterval) {
      clearInterval(this.optimizationInterval);
    }

    this.statusBarItem.dispose();
    this.memoryOptimizer.dispose();
    this.modelCache.clear();
    this.commandClusters.clear();
    this.executionHistory = [];
  }
}
