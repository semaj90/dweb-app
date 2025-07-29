import * as vscode from 'vscode';
import { StatusBarState } from './types';

export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private currentState: StatusBarState;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right, 
            100
        );
        
        this.currentState = {
            status: 'offline',
            message: 'MCP Offline'
        };

        this.statusBarItem.command = 'mcp.analyzeCurrentContext';
        this.updateDisplay();
        this.statusBarItem.show();
    }

    updateStatus(status: StatusBarState['status'], message: string, tooltip?: string): void {
        this.currentState = {
            status,
            message,
            tooltip
        };
        this.updateDisplay();
    }

    private updateDisplay(): void {
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

    getState(): StatusBarState {
        return { ...this.currentState };
    }

    dispose(): void {
        this.statusBarItem.dispose();
    }
}