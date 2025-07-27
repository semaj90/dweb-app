import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext) {
  // Register command to open the LLM Manager panel
  context.subscriptions.push(
    vscode.commands.registerCommand("llmManager.openPanel", () => {
      LLMPanel.createOrShow(context.extensionUri);
    })
  );

  // Register command to refresh models
  context.subscriptions.push(
    vscode.commands.registerCommand("llmManager.refreshModels", () => {
      vscode.window.showInformationMessage("Refreshing LLM model list...");
      // TODO: Integrate with Context7 MCP/vLLM backend
    })
  );
}

export function deactivate() {}

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
