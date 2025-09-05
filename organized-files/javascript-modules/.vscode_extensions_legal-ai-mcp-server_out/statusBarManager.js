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
exports.StatusBarManager = void 0;
const vscode = __importStar(require("vscode"));
class StatusBarManager {
    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.currentState = {
            status: 'offline',
            message: 'MCP Offline'
        };
        this.statusBarItem.command = 'mcp.analyzeCurrentContext';
        this.updateDisplay();
        this.statusBarItem.show();
    }
    updateStatus(status, message, tooltip) {
        this.currentState = {
            status,
            message,
            tooltip
        };
        this.updateDisplay();
    }
    updateDisplay() {
        const { status, message, tooltip } = this.currentState;
        // Set icon based on status
        let icon = '$(question)';
        switch (status) {
            case 'ready':
                icon = '$(check)';
                break;
            case 'analyzing':
                icon = '$(sync~spin)';
                break;
            case 'executing':
                icon = '$(gear~spin)';
                break;
            case 'error':
                icon = '$(error)';
                break;
            case 'offline':
                icon = '$(circle-slash)';
                break;
        }
        this.statusBarItem.text = `${icon} Context7 MCP: ${message}`;
        this.statusBarItem.tooltip = tooltip || `Context7 MCP Assistant - ${message}`;
        // Set color based on status
        switch (status) {
            case 'ready':
                this.statusBarItem.color = undefined; // Default color
                break;
            case 'analyzing':
            case 'executing':
                this.statusBarItem.color = new vscode.ThemeColor('statusBarItem.prominentForeground');
                break;
            case 'error':
                this.statusBarItem.color = new vscode.ThemeColor('statusBarItem.errorForeground');
                break;
            case 'offline':
                this.statusBarItem.color = new vscode.ThemeColor('statusBarItem.warningForeground');
                break;
        }
    }
    getState() {
        return { ...this.currentState };
    }
    dispose() {
        this.statusBarItem.dispose();
    }
}
exports.StatusBarManager = StatusBarManager;
//# sourceMappingURL=statusBarManager.js.map