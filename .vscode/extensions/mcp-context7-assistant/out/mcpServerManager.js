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
exports.MCPServerManager = void 0;
const vscode = __importStar(require("vscode"));
const child_process_1 = require("child_process");
class MCPServerManager {
    constructor(context) {
        this.server = null;
        this.status = {
            running: false,
            port: 3000
        };
        this.context = context;
    }
    async startServer() {
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
            this.server = (0, child_process_1.spawn)('node', [mcpServerPath], {
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
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to start MCP Server: ${error}`);
        }
    }
    stopServer() {
        if (this.server) {
            this.server.kill();
            this.server = null;
            this.status.running = false;
            vscode.window.showInformationMessage('MCP Server stopped');
        }
        else {
            vscode.window.showWarningMessage('MCP Server is not running');
        }
    }
    async callMCPTool(toolName, args) {
        if (!this.status.running) {
            throw new Error('MCP Server is not running');
        }
        const startTime = Date.now();
        try {
            const response = await globalThis.fetch(`http://localhost:${this.status.port}/mcp/call`, {
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
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                executionTime: Date.now() - startTime
            };
        }
    }
    getStatus() {
        return { ...this.status };
    }
    onWorkspaceChanged(event) {
        // Restart server with new workspace context
        if (this.status.running) {
            this.stopServer();
            setTimeout(() => this.startServer(), 1000);
        }
    }
}
exports.MCPServerManager = MCPServerManager;
//# sourceMappingURL=mcpServerManager.js.map