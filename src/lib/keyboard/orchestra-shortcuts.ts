/**
 * Remote Control Keyboard Shortcuts for Legal AI Platform Orchestration
 * Provides system-wide keyboard shortcuts for service management and monitoring
 */

import { browser } from '$app/environment';

export interface ServiceCommand {
  id: string;
  name: string;
  description: string;
  endpoint?: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  payload?: unknown;
  category: 'system' | 'service' | 'monitoring' | 'encoding' | 'gpu' | 'database';
}

export interface ShortcutConfig {
  key: string;
  command: ServiceCommand;
  enabled: boolean;
  modifier?: 'ctrl' | 'alt' | 'shift';
}

export class OrchestraKeyboardController {
  private shortcuts: Map<string, ShortcutConfig> = new Map();
  private isEnabled: boolean = false;
  private statusDisplay: HTMLElement | null = null;

  constructor() {
    this.initializeShortcuts();
    if (browser) {
      this.setupEventListeners();
      this.createStatusDisplay();
    }
  }

  /**
   * Initialize all keyboard shortcuts
   */
  private initializeShortcuts(): void {
    const shortcuts: ShortcutConfig[] = [
      // System Control
      {
        key: 'ctrl+alt+r',
        command: {
          id: 'restart-all',
          name: 'Restart All Services',
          description: 'Restart all microservices and workers',
          endpoint: '/api/orchestration/restart-all',
          method: 'POST',
          category: 'system'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+s',
        command: {
          id: 'stop-all',
          name: 'Stop All Services',
          description: 'Gracefully stop all services',
          endpoint: '/api/orchestration/stop-all',
          method: 'POST',
          category: 'system'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+d',
        command: {
          id: 'deploy-production',
          name: 'Deploy Production',
          description: 'Deploy to production environment',
          endpoint: '/api/orchestration/deploy',
          method: 'POST',
          payload: { environment: 'production' },
          category: 'system'
        },
        enabled: true
      },

      // Service Management
      {
        key: 'ctrl+shift+1',
        command: {
          id: 'restart-rag',
          name: 'Restart RAG Service',
          description: 'Restart Enhanced RAG Service (Port 8094)',
          endpoint: '/api/services/enhanced-rag/restart',
          method: 'POST',
          category: 'service'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+2',
        command: {
          id: 'restart-upload',
          name: 'Restart Upload Service',
          description: 'Restart Upload Service (Port 8093)',
          endpoint: '/api/services/upload/restart',
          method: 'POST',
          category: 'service'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+3',
        command: {
          id: 'restart-cuda',
          name: 'Restart CUDA Worker',
          description: 'Restart CUDA array processing worker',
          endpoint: '/api/workers/cuda/restart',
          method: 'POST',
          category: 'service'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+4',
        command: {
          id: 'restart-mcp',
          name: 'Restart MCP Server',
          description: 'Restart Context7 MCP multicore server',
          endpoint: '/api/services/mcp/restart',
          method: 'POST',
          category: 'service'
        },
        enabled: true
      },

      // Monitoring & Debugging
      {
        key: 'ctrl+alt+l',
        command: {
          id: 'toggle-logs',
          name: 'Toggle Live Logs',
          description: 'Show/hide live log stream',
          category: 'monitoring'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+m',
        command: {
          id: 'metrics-dashboard',
          name: 'Open Metrics Dashboard',
          description: 'Open real-time metrics dashboard',
          endpoint: '/metrics/dashboard',
          method: 'GET',
          category: 'monitoring'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+h',
        command: {
          id: 'health-status',
          name: 'Show Health Status',
          description: 'Display system health overview',
          endpoint: '/api/health/all',
          method: 'GET',
          category: 'monitoring'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+t',
        command: {
          id: 'toggle-tracing',
          name: 'Toggle Tracing',
          description: 'Enable/disable distributed tracing',
          endpoint: '/api/tracing/toggle',
          method: 'POST',
          category: 'monitoring'
        },
        enabled: true
      },

      // Encoding Control
      {
        key: 'ctrl+shift+c',
        command: {
          id: 'switch-cbor',
          name: 'Switch to CBOR',
          description: 'Set CBOR as default encoding format',
          endpoint: '/api/encoding/format',
          method: 'PUT',
          payload: { format: 'cbor' },
          category: 'encoding'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+p',
        command: {
          id: 'switch-msgpack',
          name: 'Switch to MessagePack',
          description: 'Set MessagePack as default encoding format',
          endpoint: '/api/encoding/format',
          method: 'PUT',
          payload: { format: 'msgpack' },
          category: 'encoding'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+j',
        command: {
          id: 'switch-json',
          name: 'Switch to JSON',
          description: 'Set JSON as default encoding format',
          endpoint: '/api/encoding/format',
          method: 'PUT',
          payload: { format: 'json' },
          category: 'encoding'
        },
        enabled: true
      },

      // GPU & Processing
      {
        key: 'ctrl+alt+g',
        command: {
          id: 'gpu-status',
          name: 'GPU Status',
          description: 'Show GPU utilization and memory',
          endpoint: '/api/gpu/status',
          method: 'GET',
          category: 'gpu'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+a',
        command: {
          id: 'test-array-processing',
          name: 'Test Array Processing',
          description: 'Run CUDA array processing test',
          endpoint: '/api/gpu/test-array',
          method: 'POST',
          payload: { testType: 'som_training' },
          category: 'gpu'
        },
        enabled: true
      },
      {
        key: 'ctrl+alt+v',
        command: {
          id: 'test-vector-search',
          name: 'Test Vector Search',
          description: 'Run vector similarity search test',
          endpoint: '/api/vector/test-search',
          method: 'POST',
          category: 'gpu'
        },
        enabled: true
      },

      // Database Operations
      {
        key: 'ctrl+shift+d',
        command: {
          id: 'database-health',
          name: 'Database Health',
          description: 'Check all database connections',
          endpoint: '/api/database/health',
          method: 'GET',
          category: 'database'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+v',
        command: {
          id: 'vector-db-stats',
          name: 'Vector DB Stats',
          description: 'Show vector database statistics',
          endpoint: '/api/database/vector-stats',
          method: 'GET',
          category: 'database'
        },
        enabled: true
      },
      {
        key: 'ctrl+shift+r',
        command: {
          id: 'cache-stats',
          name: 'Cache Statistics',
          description: 'Show Redis cache performance',
          endpoint: '/api/cache/stats',
          method: 'GET',
          category: 'database'
        },
        enabled: true
      }
    ];

    shortcuts.forEach(shortcut => {
      this.shortcuts.set(shortcut.key, shortcut);
    });
  }

  /**
   * Setup global keyboard event listeners
   */
  private setupEventListeners(): void {
    if (!browser) return;

    // Global keydown listener
    document.addEventListener('keydown', this.handleKeydown.bind(this));

    // Help overlay toggle
    document.addEventListener('keydown', (e) => {
      if (e.key === 'F1' || (e.ctrlKey && e.key === '?')) {
        e.preventDefault();
        this.showHelpOverlay();
      }

      if (e.key === 'Escape') {
        this.hideHelpOverlay();
        this.hideStatusDisplay();
      }
    });
  }

  /**
   * Handle keydown events and execute commands
   */
  private handleKeydown(event: KeyboardEvent): void {
    if (!this.isEnabled) return;

    const key = this.formatKeyCombo(event);
    const shortcut = this.shortcuts.get(key);

    if (shortcut && shortcut.enabled) {
      event.preventDefault();
      this.executeCommand(shortcut.command);
      this.showCommandExecuted(shortcut.command);
    }
  }

  /**
   * Format key combination string
   */
  private formatKeyCombo(event: KeyboardEvent): string {
    const parts: string[] = [];
    
    if (event.ctrlKey) parts.push('ctrl');
    if (event.altKey) parts.push('alt');
    if (event.shiftKey) parts.push('shift');
    
    parts.push(event.key.toLowerCase());
    
    return parts.join('+');
  }

  /**
   * Execute a service command
   */
  private async executeCommand(command: ServiceCommand): Promise<void> {
    try {
      if (command.endpoint) {
        const response = await fetch(command.endpoint, {
          method: command.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          body: command.payload ? JSON.stringify(command.payload) : undefined
        });

        if (!response.ok) {
          throw new Error(`Command failed: ${response.statusText}`);
        }

        const result = await response.json();
        this.showSuccess(command.name, result);
      } else {
        // Handle client-side commands
        this.handleClientCommand(command);
      }
    } catch (error) {
      this.showError(command.name, error as Error);
    }
  }

  /**
   * Handle client-side commands
   */
  private handleClientCommand(command: ServiceCommand): void {
    switch (command.id) {
      case 'toggle-logs':
        this.toggleLogDisplay();
        break;
      case 'metrics-dashboard':
        window.open('/metrics/dashboard', '_blank');
        break;
      default:
        console.log(`Executing client command: ${command.name}`);
    }
  }

  /**
   * Toggle log display
   */
  private toggleLogDisplay(): void {
    let logDisplay = document.getElementById('orchestra-logs');
    
    if (!logDisplay) {
      logDisplay = document.createElement('div');
      logDisplay.id = 'orchestra-logs';
      logDisplay.style.cssText = `
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 300px;
        background: rgba(0, 0, 0, 0.9);
        color: #00ff00;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        padding: 10px;
        overflow-y: auto;
        z-index: 10000;
        border-top: 2px solid #00ff00;
      `;
      document.body.appendChild(logDisplay);
      
      // Start log streaming
      this.startLogStreaming(logDisplay);
    } else {
      logDisplay.style.display = logDisplay.style.display === 'none' ? 'block' : 'none';
    }
  }

  /**
   * Start streaming logs
   */
  private async startLogStreaming(display: HTMLElement): Promise<void> {
    try {
      const response = await fetch('/api/logs/stream');
      const reader = response.body?.getReader();
      
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const text = new TextDecoder().decode(value);
          display.innerHTML += text + '\n';
          display.scrollTop = display.scrollHeight;
        }
      }
    } catch (error) {
      display.innerHTML += `Log streaming error: ${error}\n`;
    }
  }

  /**
   * Create status display
   */
  private createStatusDisplay(): void {
    this.statusDisplay = document.createElement('div');
    this.statusDisplay.id = 'orchestra-status';
    this.statusDisplay.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 10px 15px;
      border-radius: 5px;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      font-size: 14px;
      z-index: 10001;
      min-width: 200px;
      display: none;
    `;
    document.body.appendChild(this.statusDisplay);
  }

  /**
   * Show command executed notification
   */
  private showCommandExecuted(command: ServiceCommand): void {
    if (this.statusDisplay) {
      this.statusDisplay.innerHTML = `
        <div style="color: #4CAF50;">✓ ${command.name}</div>
        <div style="font-size: 12px; color: #ccc;">${command.description}</div>
      `;
      this.statusDisplay.style.display = 'block';
      
      setTimeout(() => {
        this.hideStatusDisplay();
      }, 3000);
    }
  }

  /**
   * Show success message
   */
  private showSuccess(commandName: string, result: unknown): void {
    if (this.statusDisplay) {
      this.statusDisplay.innerHTML = `
        <div style="color: #4CAF50;">✓ ${commandName} - Success</div>
        <div style="font-size: 12px; color: #ccc;">${JSON.stringify(result, null, 2).substring(0, 100)}...</div>
      `;
      this.statusDisplay.style.display = 'block';
      
      setTimeout(() => {
        this.hideStatusDisplay();
      }, 5000);
    }
  }

  /**
   * Show error message
   */
  private showError(commandName: string, error: Error): void {
    if (this.statusDisplay) {
      this.statusDisplay.innerHTML = `
        <div style="color: #f44336;">✗ ${commandName} - Error</div>
        <div style="font-size: 12px; color: #ccc;">${error.message}</div>
      `;
      this.statusDisplay.style.display = 'block';
      
      setTimeout(() => {
        this.hideStatusDisplay();
      }, 5000);
    }
  }

  /**
   * Hide status display
   */
  private hideStatusDisplay(): void {
    if (this.statusDisplay) {
      this.statusDisplay.style.display = 'none';
    }
  }

  /**
   * Show help overlay
   */
  private showHelpOverlay(): void {
    let helpOverlay = document.getElementById('orchestra-help');
    
    if (!helpOverlay) {
      helpOverlay = document.createElement('div');
      helpOverlay.id = 'orchestra-help';
      helpOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 40px;
        overflow-y: auto;
        z-index: 10002;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      `;
      
      const categories = this.groupShortcutsByCategory();
      helpOverlay.innerHTML = this.generateHelpHTML(categories);
      
      document.body.appendChild(helpOverlay);
    } else {
      helpOverlay.style.display = 'block';
    }
  }

  /**
   * Hide help overlay
   */
  private hideHelpOverlay(): void {
    const helpOverlay = document.getElementById('orchestra-help');
    if (helpOverlay) {
      helpOverlay.style.display = 'none';
    }
  }

  /**
   * Group shortcuts by category
   */
  private groupShortcutsByCategory(): Map<string, ShortcutConfig[]> {
    const categories = new Map<string, ShortcutConfig[]>();
    
    this.shortcuts.forEach(shortcut => {
      const category = shortcut.command.category;
      if (!categories.has(category)) {
        categories.set(category, []);
      }
      categories.get(category)!.push(shortcut);
    });
    
    return categories;
  }

  /**
   * Generate help HTML
   */
  private generateHelpHTML(categories: Map<string, ShortcutConfig[]>): string {
    let html = `
      <h1>🎹 Orchestra Keyboard Shortcuts</h1>
      <p>Press <strong>Esc</strong> to close this help, <strong>F1</strong> to show it again.</p>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 30px;">
    `;

    categories.forEach((shortcuts, category) => {
      html += `
        <div>
          <h2 style="color: #4CAF50; text-transform: capitalize;">${category.replace('_', ' ')} Commands</h2>
          <table style="width: 100%; border-collapse: collapse;">
      `;

      shortcuts.forEach(shortcut => {
        html += `
          <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 8px; font-family: monospace; color: #ff9800;">${shortcut.key.toUpperCase()}</td>
            <td style="padding: 8px;">
              <strong>${shortcut.command.name}</strong><br>
              <small style="color: #ccc;">${shortcut.command.description}</small>
            </td>
          </tr>
        `;
      });

      html += `
          </table>
        </div>
      `;
    });

    html += `
      </div>
      <div style="margin-top: 40px; text-align: center; color: #666;">
        <p>Legal AI Platform Orchestra Control System</p>
      </div>
    `;

    return html;
  }

  /**
   * Enable keyboard shortcuts
   */
  public enable(): void {
    this.isEnabled = true;
    console.log('Orchestra keyboard shortcuts enabled');
  }

  /**
   * Disable keyboard shortcuts
   */
  public disable(): void {
    this.isEnabled = false;
    console.log('Orchestra keyboard shortcuts disabled');
  }

  /**
   * Get all shortcuts
   */
  public getShortcuts(): ShortcutConfig[] {
    return Array.from(this.shortcuts.values());
  }

  /**
   * Add custom shortcut
   */
  public addShortcut(key: string, command: ServiceCommand): void {
    this.shortcuts.set(key, {
      key,
      command,
      enabled: true
    });
  }

  /**
   * Remove shortcut
   */
  public removeShortcut(key: string): void {
    this.shortcuts.delete(key);
  }
}

// Global instance
export const orchestraKeyboard = new OrchestraKeyboardController();

// Auto-enable in browser
if (browser) {
  orchestraKeyboard.enable();
}