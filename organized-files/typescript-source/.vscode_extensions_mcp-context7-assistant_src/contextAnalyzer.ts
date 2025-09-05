import * as vscode from 'vscode';
import * as fs from 'fs';
import { VSCodeMCPContext, AutoMCPSuggestion, DiagnosticError } from './types';
// Note: These functions would come from the main application when this extension is running in context
// For standalone compilation, we'll implement minimal versions

// Minimal implementations for standalone compilation
function getContextAwareSuggestions(context: VSCodeMCPContext): Promise<AutoMCPSuggestion[]> {
    return Promise.resolve([]);
}

function analyzeErrorsForMCPSuggestions(errors: DiagnosticError[]): AutoMCPSuggestion[] {
    return [];
}

function analyzeFilesForStackSuggestions(files: string[]): AutoMCPSuggestion[] {
    return [];
}

function analyzePromptIntent(prompts: string[]): AutoMCPSuggestion[] {
    return [];
}

export class ContextAnalyzer {
    
    async getContextAwareSuggestions(vsCodeContext: VSCodeMCPContext): Promise<AutoMCPSuggestion[]> {
        return await getContextAwareSuggestions(vsCodeContext);
    }

    analyzeErrorsForMCPSuggestions(errors: DiagnosticError[]): AutoMCPSuggestion[] {
        return analyzeErrorsForMCPSuggestions(errors);
    }

    analyzeFilesForStackSuggestions(activeFiles: string[]): AutoMCPSuggestion[] {
        return analyzeFilesForStackSuggestions(activeFiles);
    }

    analyzePromptIntent(recentPrompts: string[]): AutoMCPSuggestion[] {
        return analyzePromptIntent(recentPrompts);
    }

    async analyzeCurrentFileForDocs(fileName: string, vsCodeContext: VSCodeMCPContext): Promise<AutoMCPSuggestion[]> {
        const suggestions: AutoMCPSuggestion[] = [];
        
        try {
            const content = fs.readFileSync(fileName, 'utf8');
            const extension = fileName.split('.').pop()?.toLowerCase();

            // Analyze file extension and content for documentation suggestions
            switch (extension) {
                case 'ts':
                case 'js':
                    if (content.includes('xstate') || content.includes('createMachine')) {
                        suggestions.push({
                            tool: 'get-library-docs',
                            confidence: 0.9,
                            reasoning: 'XState usage detected in current file',
                            args: { context7CompatibleLibraryID: 'xstate', topic: 'machines' },
                            priority: 'high',
                            expectedOutput: 'XState machine creation and management documentation'
                        });
                    }
                    if (content.includes('drizzle') || content.includes('pgTable')) {
                        suggestions.push({
                            tool: 'get-library-docs',
                            confidence: 0.85,
                            reasoning: 'Drizzle ORM usage detected in current file',
                            args: { context7CompatibleLibraryID: 'drizzle', topic: 'schema' },
                            priority: 'high',
                            expectedOutput: 'Drizzle ORM schema definition documentation'
                        });
                    }
                    break;

                case 'svelte':
                    suggestions.push({
                        tool: 'get-library-docs',
                        confidence: 0.8,
                        reasoning: 'Svelte component file detected',
                        args: { context7CompatibleLibraryID: 'svelte', topic: 'components' },
                        priority: 'medium',
                        expectedOutput: 'Svelte component development documentation'
                    });
                    break;

                case 'py':
                    if (content.includes('import torch') || content.includes('import tensorflow')) {
                        suggestions.push({
                            tool: 'get-library-docs',
                            confidence: 0.9,
                            reasoning: 'AI/ML framework detected in Python file',
                            args: { context7CompatibleLibraryID: 'pytorch', topic: 'neural-networks' },
                            priority: 'high',
                            expectedOutput: 'PyTorch neural network documentation'
                        });
                    }
                    break;

                case 'cu':
                case 'cuh':
                    suggestions.push({
                        tool: 'get-library-docs',
                        confidence: 0.95,
                        reasoning: 'CUDA source file detected',
                        args: { context7CompatibleLibraryID: 'cuda', topic: 'programming-guide' },
                        priority: 'high',
                        expectedOutput: 'CUDA programming guide and best practices'
                    });
                    break;

                case 'cpp':
                case 'c':
                    if (content.includes('#include <cuda') || content.includes('__global__')) {
                        suggestions.push({
                            tool: 'get-library-docs',
                            confidence: 0.9,
                            reasoning: 'CUDA C++ code detected',
                            args: { context7CompatibleLibraryID: 'cuda', topic: 'runtime-api' },
                            priority: 'high',
                            expectedOutput: 'CUDA runtime API documentation'
                        });
                    }
                    break;
            }

            // Analyze comments for explicit documentation requests
            const comments = this.extractCommentsFromFile(content, extension || '');
            const commentSuggestions = this.analyzePromptIntent(comments);
            suggestions.push(...commentSuggestions);

        } catch (error) {
            console.log('Error analyzing file for docs:', error);
        }

        return suggestions;
    }

    private extractCommentsFromFile(content: string, extension: string): string[] {
        const comments: string[] = [];
        
        try {
            const lines = content.split('\n');
            
            for (const line of lines) {
                const trimmed = line.trim();
                
                // Extract different comment styles
                if (extension === 'py' && trimmed.startsWith('#')) {
                    comments.push(trimmed.substring(1).trim());
                } else if (['ts', 'js', 'cpp', 'c', 'cu'].includes(extension)) {
                    if (trimmed.startsWith('//')) {
                        comments.push(trimmed.substring(2).trim());
                    } else if (trimmed.startsWith('/*') && trimmed.endsWith('*/')) {
                        comments.push(trimmed.substring(2, trimmed.length - 2).trim());
                    }
                } else if (extension === 'html' && trimmed.startsWith('<!--') && trimmed.endsWith('-->')) {
                    comments.push(trimmed.substring(4, trimmed.length - 3).trim());
                }
            }
        } catch (error) {
            console.log('Error extracting comments:', error);
        }
        
        return comments.filter(comment => comment.length > 0);
    }

    async extractRecentPrompts(activeFiles: string[]): Promise<string[]> {
        const allPrompts: string[] = [];
        
        // Limit to most recently opened files to avoid performance issues
        const recentFiles = activeFiles.slice(0, 10);
        
        for (const filePath of recentFiles) {
            try {
                const content = fs.readFileSync(filePath, 'utf8');
                const extension = filePath.split('.').pop()?.toLowerCase() || '';
                const comments = this.extractCommentsFromFile(content, extension);
                
                // Filter for prompts that look like requests or questions
                const prompts = comments.filter(comment => 
                    comment.toLowerCase().includes('todo') ||
                    comment.toLowerCase().includes('fix') ||
                    comment.toLowerCase().includes('implement') ||
                    comment.toLowerCase().includes('add') ||
                    comment.toLowerCase().includes('how') ||
                    comment.toLowerCase().includes('why') ||
                    comment.toLowerCase().includes('best practice') ||
                    comment.toLowerCase().includes('optimize')
                );
                
                allPrompts.push(...prompts);
            } catch (error) {
                // Skip files that can't be read
                continue;
            }
        }
        
        // Return most recent unique prompts
        return [...new Set(allPrompts)].slice(0, 20);
    }

    onActiveEditorChanged(editor: vscode.TextEditor): void {
        // Could be used to update context when active editor changes
        console.log('Active editor changed:', editor.document.fileName);
    }
}