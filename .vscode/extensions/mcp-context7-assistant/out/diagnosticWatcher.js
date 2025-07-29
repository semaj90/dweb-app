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
exports.DiagnosticWatcher = void 0;
const vscode = __importStar(require("vscode"));
class DiagnosticWatcher {
    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('mcpContext7');
    }
    onDiagnosticsChanged(event) {
        // Could be used to automatically trigger MCP suggestions when errors change
        const errorCount = this.getTotalErrorCount();
        console.log(`Diagnostics changed - ${errorCount} total errors`);
    }
    convertDiagnosticsToErrors(diagnostics) {
        const errors = [];
        for (const [uri, uriDiagnostics] of diagnostics) {
            for (const diagnostic of uriDiagnostics) {
                // Only include errors and warnings, skip info
                if (diagnostic.severity === vscode.DiagnosticSeverity.Error ||
                    diagnostic.severity === vscode.DiagnosticSeverity.Warning) {
                    errors.push({
                        file: uri.fsPath,
                        line: diagnostic.range.start.line + 1,
                        message: diagnostic.message,
                        severity: this.convertSeverity(diagnostic.severity),
                        source: diagnostic.source || 'other'
                    });
                }
            }
        }
        return errors;
    }
    convertSeverity(severity) {
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
    getTotalErrorCount() {
        const diagnostics = vscode.languages.getDiagnostics();
        let totalErrors = 0;
        for (const [, uriDiagnostics] of diagnostics) {
            totalErrors += uriDiagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Error).length;
        }
        return totalErrors;
    }
    getErrorsByFile(filePath) {
        const uri = vscode.Uri.file(filePath);
        const diagnostics = vscode.languages.getDiagnostics(uri);
        return this.convertDiagnosticsToErrors([[uri, diagnostics]]);
    }
    dispose() {
        this.diagnosticCollection.dispose();
    }
}
exports.DiagnosticWatcher = DiagnosticWatcher;
//# sourceMappingURL=diagnosticWatcher.js.map