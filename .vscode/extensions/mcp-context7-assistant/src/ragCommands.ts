/**
 * Enhanced RAG Commands for VS Code Extension
 * Provides RAG-powered commands integrated with the backend
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { RAGBackendClient, SearchQuery, WorkflowRequest } from './ragBackendClient';

export class RAGCommandProvider {
  private ragClient: RAGBackendClient;
  private outputChannel: vscode.OutputChannel;

  constructor(ragClient: RAGBackendClient) {
    this.ragClient = ragClient;
    this.outputChannel = vscode.window.createOutputChannel('RAG Commands');
  }

  /**
   * Register all RAG commands
   */
  registerCommands(context: vscode.ExtensionContext): void {
    // Document search and analysis commands
    context.subscriptions.push(
      vscode.commands.registerCommand('rag.searchDocuments', () => this.searchDocuments()),
      vscode.commands.registerCommand('rag.analyzeCurrentFile', () => this.analyzeCurrentFile()),
      vscode.commands.registerCommand('rag.summarizeSelection', () => this.summarizeSelection()),
      vscode.commands.registerCommand('rag.uploadDocument', () => this.uploadDocument()),
      vscode.commands.registerCommand('rag.chatWithAI', () => this.chatWithAI()),
      
      // Workflow commands
      vscode.commands.registerCommand('rag.documentAnalysisWorkflow', () => this.documentAnalysisWorkflow()),
      vscode.commands.registerCommand('rag.legalResearchWorkflow', () => this.legalResearchWorkflow()),
      vscode.commands.registerCommand('rag.casePreparationWorkflow', () => this.casePreparationWorkflow()),
      vscode.commands.registerCommand('rag.contractReviewWorkflow', () => this.contractReviewWorkflow()),
      
      // System commands
      vscode.commands.registerCommand('rag.showSystemStatus', () => this.showSystemStatus()),
      vscode.commands.registerCommand('rag.showRAGStats', () => this.showRAGStats()),
      vscode.commands.registerCommand('rag.clearCache', () => this.clearCache()),
      vscode.commands.registerCommand('rag.findSimilarFiles', () => this.findSimilarFiles())
    );

    // Register context menu commands
    context.subscriptions.push(
      vscode.commands.registerCommand('rag.analyzeSelectedText', () => this.analyzeSelectedText()),
      vscode.commands.registerCommand('rag.findSimilarContent', () => this.findSimilarContent())
    );
  }

  /**
   * Search documents using semantic search
   */
  private async searchDocuments(): Promise<void> {
    try {
      const query = await vscode.window.showInputBox({
        prompt: 'Enter search query for documents',
        placeholder: 'e.g., contract liability clauses, legal precedents, evidence analysis',
        value: ''
      });

      if (!query) return;

      // Show search type options
      const searchType = await vscode.window.showQuickPick([
        { label: 'Hybrid Search', value: 'hybrid', description: 'Best of vector and text search' },
        { label: 'Vector Search', value: 'vector', description: 'Semantic similarity search' },
        { label: 'Chunk Search', value: 'chunks', description: 'Search document chunks' }
      ], {
        placeHolder: 'Select search type'
      });

      if (!searchType) return;

      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Searching documents...',
        cancellable: false
      }, async (progress) => {
        progress.report({ increment: 0, message: 'Preparing search...' });

        const searchQuery: SearchQuery = {
          query,
          searchType: searchType.value as any,
          limit: 10,
          includeContent: true
        };

        progress.report({ increment: 50, message: 'Executing search...' });
        const results = await this.ragClient.searchDocuments(searchQuery);

        progress.report({ increment: 100, message: 'Displaying results...' });
        await this.showSearchResults(results, query);
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Search failed: ${error}`);
      this.outputChannel.appendLine(`Search error: ${error}`);
    }
  }

  /**
   * Analyze the currently active file
   */
  private async analyzeCurrentFile(): Promise<void> {
    try {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active file to analyze');
        return;
      }

      const document = editor.document;
      const text = document.getText();

      if (text.length === 0) {
        vscode.window.showWarningMessage('Current file is empty');
        return;
      }

      if (text.length > 50000) {
        vscode.window.showWarningMessage('File is too long for analysis (max 50,000 characters)');
        return;
      }

      // Determine analysis type based on file extension
      const ext = path.extname(document.fileName).toLowerCase();
      let analysisType = 'general';
      
      if (['.ts', '.js', '.py', '.java', '.c', '.cpp'].includes(ext)) {
        analysisType = 'general';
      } else if (['.md', '.txt', '.doc', '.docx'].includes(ext)) {
        analysisType = 'general';
      }

      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Analyzing file with AI...',
        cancellable: false
      }, async (progress) => {
        progress.report({ increment: 0, message: 'Preparing analysis...' });

        const analysis = await this.ragClient.analyzeText(text, analysisType);

        progress.report({ increment: 100, message: 'Analysis complete' });
        await this.showAnalysisResults(analysis, document.fileName);
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Analysis failed: ${error}`);
      this.outputChannel.appendLine(`Analysis error: ${error}`);
    }
  }

  /**
   * Summarize selected text
   */
  private async summarizeSelection(): Promise<void> {
    try {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
      }

      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);

      if (!selectedText) {
        vscode.window.showWarningMessage('No text selected');
        return;
      }

      const summaryLength = await vscode.window.showQuickPick([
        { label: 'Short', value: 'short', description: 'Brief 2-3 sentence summary' },
        { label: 'Medium', value: 'medium', description: 'Comprehensive paragraph summary' },
        { label: 'Long', value: 'long', description: 'Detailed multi-paragraph summary' }
      ], {
        placeHolder: 'Select summary length'
      });

      if (!summaryLength) return;

      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Summarizing text...',
        cancellable: false
      }, async (progress) => {
        progress.report({ increment: 0, message: 'Generating summary...' });

        const summary = await this.ragClient.summarizeText(selectedText, summaryLength.value);

        progress.report({ increment: 100, message: 'Summary complete' });
        await this.showSummaryResults(summary, selectedText.length);
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Summarization failed: ${error}`);
      this.outputChannel.appendLine(`Summarization error: ${error}`);
    }
  }

  /**
   * Upload document to RAG system
   */
  private async uploadDocument(): Promise<void> {
    try {
      const fileUri = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: false,
        filters: {
          'Documents': ['pdf', 'docx', 'doc', 'txt', 'html', 'htm'],
          'Images': ['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
          'All Files': ['*']
        }
      });

      if (!fileUri || fileUri.length === 0) return;

      const filePath = fileUri[0].fsPath;
      const fileName = path.basename(filePath);

      // Get document metadata
      const title = await vscode.window.showInputBox({
        prompt: 'Enter document title (optional)',
        value: path.parse(fileName).name
      });

      const documentType = await vscode.window.showQuickPick([
        { label: 'General', value: 'general' },
        { label: 'Contract', value: 'contract' },
        { label: 'Evidence', value: 'evidence' },
        { label: 'Legal Document', value: 'legal' },
        { label: 'Research', value: 'research' }
      ], {
        placeHolder: 'Select document type'
      });

      const caseId = await vscode.window.showInputBox({
        prompt: 'Enter case ID (optional)',
        placeholder: 'e.g., CASE-2025-001'
      });

      if (!documentType) return;

      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Uploading document...',
        cancellable: false
      }, async (progress) => {
        progress.report({ increment: 0, message: 'Reading file...' });

        const fileBuffer = fs.readFileSync(filePath);

        progress.report({ increment: 30, message: 'Uploading and processing...' });

        const result = await this.ragClient.uploadDocument({
          file: fileBuffer,
          fileName,
          title: title || undefined,
          documentType: documentType.value,
          caseId: caseId || undefined
        });

        progress.report({ increment: 100, message: 'Upload complete' });
        
        vscode.window.showInformationMessage(
          `Document uploaded successfully!\nID: ${result.document.id}\nChunks: ${result.document.chunkCount}`
        );
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Upload failed: ${error}`);
      this.outputChannel.appendLine(`Upload error: ${error}`);
    }
  }

  /**
   * Interactive chat with AI
   */
  private async chatWithAI(): Promise<void> {
    try {
      const messages: Array<{role: string, content: string}> = [];
      
      while (true) {
        const userInput = await vscode.window.showInputBox({
          prompt: 'Chat with AI (press Escape to exit)',
          placeholder: 'Ask about documents, code, or legal questions...'
        });

        if (!userInput) break;

        messages.push({ role: 'user', content: userInput });

        await vscode.window.withProgress({
          location: vscode.ProgressLocation.Notification,
          title: 'AI is thinking...',
          cancellable: false
        }, async (progress) => {
          const response = await this.ragClient.chatWithAgent(messages);
          
          if (response.response?.content) {
            messages.push({ role: 'assistant', content: response.response.content });
            
            // Show response in a new document
            const doc = await vscode.workspace.openTextDocument({
              content: `AI Response:\n\n${response.response.content}`,
              language: 'markdown'
            });
            await vscode.window.showTextDocument(doc);
          }
        });
      }

    } catch (error) {
      vscode.window.showErrorMessage(`Chat failed: ${error}`);
      this.outputChannel.appendLine(`Chat error: ${error}`);
    }
  }

  /**
   * Execute document analysis workflow
   */
  private async documentAnalysisWorkflow(): Promise<void> {
    try {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active file for analysis');
        return;
      }

      const document = editor.document;
      const text = document.getText();

      if (text.length === 0) {
        vscode.window.showWarningMessage('Current file is empty');
        return;
      }

      const workflowRequest: WorkflowRequest = {
        workflowType: 'document_analysis',
        input: {
          content: text,
          title: path.basename(document.fileName),
          documentType: 'general'
        },
        options: {
          priority: 'medium',
          context: {
            fileName: document.fileName,
            language: document.languageId
          }
        }
      };

      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Running document analysis workflow...',
        cancellable: false
      }, async (progress) => {
        progress.report({ increment: 0, message: 'Starting multi-agent analysis...' });

        const result = await this.ragClient.executeWorkflow(workflowRequest);

        progress.report({ increment: 100, message: 'Analysis complete' });
        await this.showWorkflowResults(result, 'Document Analysis');
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Workflow failed: ${error}`);
      this.outputChannel.appendLine(`Workflow error: ${error}`);
    }
  }

  /**
   * Execute legal research workflow
   */
  private async legalResearchWorkflow(): Promise<void> {
    try {
      const query = await vscode.window.showInputBox({
        prompt: 'Enter legal research query',
        placeholder: 'e.g., contract liability precedents, employment law cases',
        value: ''
      });

      if (!query) return;

      const jurisdiction = await vscode.window.showQuickPick([
        { label: 'Federal', value: 'federal' },
        { label: 'State', value: 'state' },
        { label: 'International', value: 'international' },
        { label: 'All Jurisdictions', value: 'all' }
      ], {
        placeHolder: 'Select jurisdiction'
      });

      if (!jurisdiction) return;

      const workflowRequest: WorkflowRequest = {
        workflowType: 'legal_research',
        input: {
          text: query,
          context: 'Legal research workflow',
          jurisdiction: jurisdiction.value
        },
        options: {
          priority: 'high',
          timeout: 180000 // 3 minutes for legal research
        }
      };

      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Conducting legal research...',
        cancellable: false
      }, async (progress) => {
        progress.report({ increment: 0, message: 'Planning research strategy...' });

        const result = await this.ragClient.executeWorkflow(workflowRequest);

        progress.report({ increment: 100, message: 'Research complete' });
        await this.showWorkflowResults(result, 'Legal Research');
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Legal research failed: ${error}`);
      this.outputChannel.appendLine(`Legal research error: ${error}`);
    }
  }

  /**
   * Show system status
   */
  private async showSystemStatus(): Promise<void> {
    try {
      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Checking system status...',
        cancellable: false
      }, async (progress) => {
        const [isHealthy, metrics] = await Promise.all([
          this.ragClient.healthCheck(),
          this.ragClient.getSystemMetrics()
        ]);

        const statusDoc = await vscode.workspace.openTextDocument({
          content: this.formatSystemStatus(isHealthy, metrics),
          language: 'markdown'
        });
        
        await vscode.window.showTextDocument(statusDoc);
      });

    } catch (error) {
      vscode.window.showErrorMessage(`Failed to get system status: ${error}`);
    }
  }

  /**
   * Show RAG statistics
   */
  private async showRAGStats(): Promise<void> {
    try {
      const stats = await this.ragClient.getRAGStats();
      
      const statsDoc = await vscode.workspace.openTextDocument({
        content: this.formatRAGStats(stats),
        language: 'markdown'
      });
      
      await vscode.window.showTextDocument(statsDoc);

    } catch (error) {
      vscode.window.showErrorMessage(`Failed to get RAG stats: ${error}`);
    }
  }

  /**
   * Clear system cache
   */
  private async clearCache(): Promise<void> {
    try {
      const pattern = await vscode.window.showInputBox({
        prompt: 'Enter cache pattern to clear (leave empty for all)',
        placeholder: 'e.g., rag:*, search:*, document:*'
      });

      const result = await this.ragClient.clearCache(pattern || undefined);
      
      vscode.window.showInformationMessage(
        `Cache cleared: ${result.message || 'Success'}`
      );

    } catch (error) {
      vscode.window.showErrorMessage(`Failed to clear cache: ${error}`);
    }
  }

  // Helper methods for displaying results

  private async showSearchResults(results: any, query: string): Promise<void> {
    const content = `# Search Results: "${query}"\n\n` +
      `Found **${results.results?.length || 0}** results in **${results.metadata?.processingTime || 0}ms**\n\n` +
      (results.results || []).map((result: any, index: number) => 
        `## ${index + 1}. ${result.title || 'Untitled'}\n\n` +
        `**Type:** ${result.document_type || 'general'}  \n` +
        `**Similarity:** ${(result.similarity_score * 100).toFixed(1)}%  \n` +
        `**Created:** ${new Date(result.created_at).toLocaleDateString()}  \n\n` +
        (result.content ? `**Content Preview:**\n\`\`\`\n${result.content.substring(0, 500)}...\n\`\`\`\n\n` : '') +
        `---\n\n`
      ).join('');

    const doc = await vscode.workspace.openTextDocument({
      content,
      language: 'markdown'
    });
    
    await vscode.window.showTextDocument(doc);
  }

  private async showAnalysisResults(analysis: any, fileName: string): Promise<void> {
    const content = `# AI Analysis: ${fileName}\n\n` +
      `**Analysis Type:** ${analysis.analysis?.analysisType || 'general'}  \n` +
      `**Processing Time:** ${analysis.analysis?.processingTime || 0}ms  \n` +
      `**Model:** ${analysis.analysis?.model || 'unknown'}  \n\n` +
      `## Analysis Results\n\n${analysis.analysis?.analysis || 'No analysis available'}\n\n`;

    const doc = await vscode.workspace.openTextDocument({
      content,
      language: 'markdown'
    });
    
    await vscode.window.showTextDocument(doc);
  }

  private async showSummaryResults(summary: any, originalLength: number): Promise<void> {
    const content = `# Text Summary\n\n` +
      `**Original Length:** ${originalLength.toLocaleString()} characters  \n` +
      `**Summary Length:** ${summary.summary?.summaryLength || 0} characters  \n` +
      `**Compression Ratio:** ${summary.summary?.compressionRatio || 'N/A'}  \n` +
      `**Model:** ${summary.summary?.model || 'unknown'}  \n\n` +
      `## Summary\n\n${summary.summary?.summary || 'No summary available'}\n\n`;

    const doc = await vscode.workspace.openTextDocument({
      content,
      language: 'markdown'
    });
    
    await vscode.window.showTextDocument(doc);
  }

  private async showWorkflowResults(result: any, workflowName: string): Promise<void> {
    const workflow = result.result;
    const content = `# ${workflowName} Results\n\n` +
      `**Processing Time:** ${workflow.metadata?.processingTime || 0}ms  \n` +
      `**Agents Used:** ${workflow.metadata?.agentsUsed?.join(', ') || 'unknown'}  \n` +
      `**Timestamp:** ${workflow.metadata?.timestamp || new Date().toISOString()}  \n\n` +
      `## Synthesis\n\n${workflow.synthesis?.content || 'No synthesis available'}\n\n` +
      `**Confidence:** ${((workflow.synthesis?.confidence || 0) * 100).toFixed(1)}%\n\n`;

    const doc = await vscode.workspace.openTextDocument({
      content,
      language: 'markdown'
    });
    
    await vscode.window.showTextDocument(doc);
  }

  private formatSystemStatus(isHealthy: boolean, metrics: any): string {
    return `# Enhanced RAG System Status\n\n` +
      `**Overall Status:** ${isHealthy ? '✅ Healthy' : '❌ Unhealthy'}  \n` +
      `**Last Check:** ${new Date().toISOString()}  \n\n` +
      `## Component Status\n\n` +
      Object.entries(metrics.health?.components || {}).map(([name, component]: [string, any]) =>
        `- **${name}:** ${component.status === 'healthy' ? '✅' : '❌'} ${component.status} ` +
        `(${component.responseTime || 0}ms)\n`
      ).join('') + '\n' +
      `## System Resources\n\n` +
      `- **Uptime:** ${metrics.health?.components?.system?.details?.uptime?.formatted || 'unknown'}\n` +
      `- **Memory Used:** ${metrics.health?.components?.system?.details?.memory?.heapUsed || 0}MB\n` +
      `- **Node Version:** ${metrics.health?.components?.system?.details?.nodeVersion || 'unknown'}\n`;
  }

  private formatRAGStats(stats: any): string {
    return `# RAG System Statistics\n\n` +
      `## Documents\n\n` +
      `- **Total Documents:** ${stats.stats?.documents?.total || 0}\n` +
      `- **Indexed Documents:** ${stats.stats?.documents?.indexed || 0}\n` +
      `- **Pending Processing:** ${stats.stats?.documents?.pendingProcessing || 0}\n\n` +
      `## Search Performance\n\n` +
      `- **Queries (24h):** ${stats.stats?.queries?.last24h || 0}\n` +
      `- **Average Processing Time:** ${stats.stats?.queries?.avgProcessingTime || 0}ms\n\n` +
      `## System Jobs\n\n` +
      `- **Pending Jobs:** ${stats.stats?.jobs?.pending || 0}\n` +
      `- **Running Jobs:** ${stats.stats?.jobs?.running || 0}\n`;
  }

  // Additional helper methods...
  private async analyzeSelectedText(): Promise<void> {
    // Implementation for context menu command
    await this.summarizeSelection();
  }

  private async findSimilarContent(): Promise<void> {
    // Implementation for finding similar content
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);

    if (!selectedText) {
      vscode.window.showWarningMessage('No text selected');
      return;
    }

    // Use the selected text as a search query
    const searchQuery: SearchQuery = {
      query: selectedText,
      searchType: 'vector',
      limit: 5,
      includeContent: true
    };

    try {
      const results = await this.ragClient.searchDocuments(searchQuery);
      await this.showSearchResults(results, selectedText.substring(0, 100) + '...');
    } catch (error) {
      vscode.window.showErrorMessage(`Similar content search failed: ${error}`);
    }
  }

  private async findSimilarFiles(): Promise<void> {
    // Implementation for finding similar files based on current file
    await this.analyzeCurrentFile();
  }

  private async casePreparationWorkflow(): Promise<void> {
    // Implementation for case preparation workflow
    const caseTitle = await vscode.window.showInputBox({
      prompt: 'Enter case title',
      placeholder: 'e.g., Smith v. Jones Contract Dispute'
    });

    if (!caseTitle) return;

    const caseDescription = await vscode.window.showInputBox({
      prompt: 'Enter case description',
      placeholder: 'Brief description of the case'
    });

    const workflowRequest: WorkflowRequest = {
      workflowType: 'case_preparation',
      input: {
        title: caseTitle,
        description: caseDescription || '',
        evidenceCount: 0,
        caseType: 'general'
      },
      options: {
        priority: 'high',
        timeout: 240000 // 4 minutes
      }
    };

    try {
      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Preparing case analysis...',
        cancellable: false
      }, async (progress) => {
        const result = await this.ragClient.executeWorkflow(workflowRequest);
        await this.showWorkflowResults(result, 'Case Preparation');
      });
    } catch (error) {
      vscode.window.showErrorMessage(`Case preparation failed: ${error}`);
    }
  }

  private async contractReviewWorkflow(): Promise<void> {
    // Implementation for contract review workflow
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('No active contract file for review');
      return;
    }

    const document = editor.document;
    const contractText = document.getText();

    if (contractText.length === 0) {
      vscode.window.showWarningMessage('Contract file is empty');
      return;
    }

    const workflowRequest: WorkflowRequest = {
      workflowType: 'contract_review',
      input: {
        content: contractText,
        title: path.basename(document.fileName)
      },
      options: {
        priority: 'high',
        timeout: 300000 // 5 minutes
      }
    };

    try {
      await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Reviewing contract...',
        cancellable: false
      }, async (progress) => {
        const result = await this.ragClient.executeWorkflow(workflowRequest);
        await this.showWorkflowResults(result, 'Contract Review');
      });
    } catch (error) {
      vscode.window.showErrorMessage(`Contract review failed: ${error}`);
    }
  }
}