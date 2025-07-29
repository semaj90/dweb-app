import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import { MCPServerStatus, MCPToolResult } from './types';

export class MCPServerManager {
    private server: ChildProcess | null = null;
    private context: vscode.ExtensionContext;
    private status: MCPServerStatus = {
        running: false,
        port: 3000
    };

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    async startServer(): Promise<void> {
        const config = vscode.workspace.getConfiguration('mcpContext7');
        const port = config.get('serverPort', 3000);
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

        if (!workspaceRoot) {
            vscode.window.showErrorMessage('No workspace folder found');
            return;
        }

        if (this.server) {
            vscode.window.showWarningMessage('MCP Server is already running');
            return;
        }

        try {
            // Check if MCP server exists in workspace
            const mcpServerPath = `${workspaceRoot}/mcp-server.js`;
            
            this.server = spawn('node', [mcpServerPath], {
                cwd: workspaceRoot,
                env: { 
                    ...process.env, 
                    PORT: port.toString(),
                    WORKSPACE_ROOT: workspaceRoot 
                }
            });

            this.server.on('spawn', () => {
                this.status = {
                    running: true,
                    port,
                    pid: this.server?.pid,
                    startTime: new Date()
                };
                vscode.window.showInformationMessage(`MCP Server started on port ${port}`);
            });

            this.server.on('error', (error) => {
                vscode.window.showErrorMessage(`MCP Server failed to start: ${error.message}`);
                this.status.running = false;
                this.server = null;
            });

            this.server.on('exit', (code) => {
                this.status.running = false;
                this.server = null;
                if (code !== 0) {
                    vscode.window.showWarningMessage(`MCP Server exited with code ${code}`);
                }
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Failed to start MCP Server: ${error}`);
        }
    }

    stopServer(): void {
        if (this.server) {
            this.server.kill();
            this.server = null;
            this.status.running = false;
            vscode.window.showInformationMessage('MCP Server stopped');
        } else {
            vscode.window.showWarningMessage('MCP Server is not running');
        }
    }

    async callMCPTool(toolName: string, args: Record<string, any>): Promise<MCPToolResult> {
        if (!this.status.running) {
            throw new Error('MCP Server is not running');
        }

        const startTime = Date.now();

        try {
            const response = await (globalThis as any).fetch(`http://localhost:${this.status.port}/mcp/call`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tool: toolName,
                    arguments: args
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            this.status.lastActivity = new Date();

            return {
                success: true,
                data,
                executionTime: Date.now() - startTime
            };

        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                executionTime: Date.now() - startTime
            };
        }
    }

    getStatus(): MCPServerStatus {
        return { ...this.status };
    }

    onWorkspaceChanged(event: vscode.WorkspaceFoldersChangeEvent): void {
        // Restart server with new workspace context
        if (this.status.running) {
            this.stopServer();
            setTimeout(() => this.startServer(), 1000);
        }
    }
}