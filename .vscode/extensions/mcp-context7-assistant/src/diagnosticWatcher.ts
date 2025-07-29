import * as vscode from 'vscode';
import { DiagnosticError } from './types';

export class DiagnosticWatcher {
    private diagnosticCollection: vscode.DiagnosticCollection;

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('mcpContext7');
    }

    onDiagnosticsChanged(event: vscode.DiagnosticChangeEvent): void {
        // Could be used to automatically trigger MCP suggestions when errors change
        const errorCount = this.getTotalErrorCount();
        console.log(`Diagnostics changed - ${errorCount} total errors`);
    }

    convertDiagnosticsToErrors(diagnostics: readonly [vscode.Uri, vscode.Diagnostic[]][]): DiagnosticError[] {
        const errors: DiagnosticError[] = [];

        for (const [uri, uriDiagnostics] of diagnostics) {
            for (const diagnostic of uriDiagnostics) {
                // Only include errors and warnings, skip info
                if (diagnostic.severity === vscode.DiagnosticSeverity.Error || 
                    diagnostic.severity === vscode.DiagnosticSeverity.Warning) {
                    
                    errors.push({
                        file: uri.fsPath,
                        line: diagnostic.range.start.line + 1, // Convert to 1-based line numbers
                        message: diagnostic.message,
                        severity: this.convertSeverity(diagnostic.severity),
                        source: (diagnostic.source as 'typescript' | 'eslint' | 'svelte' | 'other') || 'other'
                    });
                }
            }
        }

        return errors;
    }

    private convertSeverity(severity: vscode.DiagnosticSeverity): 'error' | 'warning' | 'info' {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return 'error';
            case vscode.DiagnosticSeverity.Warning:
                return 'warning';
            case vscode.DiagnosticSeverity.Information:
            case vscode.DiagnosticSeverity.Hint:
            default:
                return 'info';
        }
    }

    getTotalErrorCount(): number {
        const diagnostics = vscode.languages.getDiagnostics();
        let totalErrors = 0;

        for (const [, uriDiagnostics] of diagnostics) {
            totalErrors += uriDiagnostics.filter(d => 
                d.severity === vscode.DiagnosticSeverity.Error
            ).length;
        }

        return totalErrors;
    }

    getErrorsByFile(filePath: string): DiagnosticError[] {
        const uri = vscode.Uri.file(filePath);
        const diagnostics = vscode.languages.getDiagnostics(uri);
        
        return this.convertDiagnosticsToErrors([[uri, diagnostics]]);
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
    }
}