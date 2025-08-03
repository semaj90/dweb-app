import * as vscode from 'vscode';
import { NomicEmbedService } from './nomic-embed-service';
import { Neo4jService } from './neo4j-service';
import { SIMDJSONParser } from './simd-json-parser';

// Global service instances
let nomicEmbedService: NomicEmbedService | null = null;
let neo4jService: Neo4jService | null = null;
let simdJsonParser: SIMDJSONParser | null = null;

export function activate(context: vscode.ExtensionContext) {
  console.log('MCP Context7 Assistant extension is being activated...');

  try {
    // Initialize services
    nomicEmbedService = new NomicEmbedService();
    neo4jService = new Neo4jService();
    simdJsonParser = new SIMDJSONParser({
      enableWorkerThreads: true,
      workerPoolSize: 4,
      validationLevel: 'basic'
    });

    // Register new MCP Context7 commands
    registerMCPContext7Commands(context);

    vscode.window.showInformationMessage(
      'MCP Context7 Assistant extension activated successfully!'
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to activate extension: ${error}`);
    console.error('Extension activation failed:', error);
  }
}

function registerMCPContext7Commands(context: vscode.ExtensionContext) {
  // MCP: Embed VSCode Markdown Files
  context.subscriptions.push(
    vscode.commands.registerCommand('mcp.embedMarkdownFiles', async () => {
      await embedMarkdownFilesCommand();
    })
  );

  // MCP: Search Evidence
  context.subscriptions.push(
    vscode.commands.registerCommand('mcp.searchEvidence', async () => {
      await searchEvidenceCommand();
    })
  );

  // MCP: Parse Evidence SIMD JSON
  context.subscriptions.push(
    vscode.commands.registerCommand('mcp.parseEvidenceSimdJson', async () => {
      await parseEvidenceSimdJsonCommand();
    })
  );
}

/**
 * Embed markdown files command
 */
async function embedMarkdownFilesCommand(): Promise<void> {
  if (!nomicEmbedService || !neo4jService) {
    vscode.window.showErrorMessage('Required services not initialized');
    return;
  }

  try {
    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'Embedding VSCode markdown files...',
      cancellable: false
    }, async (progress) => {
      
      progress.report({ increment: 20, message: 'Connecting to services...' });
      
      // Connect to Neo4j
      await neo4jService!.connect();
      
      progress.report({ increment: 30, message: 'Processing markdown files...' });
      
      // Embed markdown files
      const result = await nomicEmbedService!.embedMarkdownFiles();
      
      progress.report({ increment: 50, message: 'Complete!' });
      
      vscode.window.showInformationMessage(
        `✅ Embedded ${result.files.length} markdown files (${result.totalEmbeddings} embeddings) in ${result.processingTime}ms`
      );
    });

  } catch (error) {
    vscode.window.showErrorMessage(`Failed to embed markdown files: ${error instanceof Error ? error.message : String(error)}`);
    console.error('Embedding error:', error);
  }
}

/**
 * Search evidence command
 */
async function searchEvidenceCommand(): Promise<void> {
  if (!neo4jService || !nomicEmbedService) {
    vscode.window.showErrorMessage('Required services not initialized');
    return;
  }

  try {
    // Get search query from user
    const query = await vscode.window.showInputBox({
      placeHolder: 'Enter search query for evidence...',
      prompt: 'Search for evidence using semantic similarity'
    });

    if (!query) return;

    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'Searching evidence...',
      cancellable: false
    }, async (progress) => {
      
      progress.report({ increment: 30, message: 'Generating query embedding...' });
      
      // Generate embedding for query
      const queryEmbedding = await nomicEmbedService!.embedTexts([query]);
      
      progress.report({ increment: 40, message: 'Searching database...' });
      
      // Connect to Neo4j if not connected
      await neo4jService!.connect();
      
      // Search for similar evidence
      const searchResults = await neo4jService!.searchEvidence(
        query,
        queryEmbedding.embeddings[0],
        undefined,
        20
      );
      
      progress.report({ increment: 30, message: 'Formatting results...' });
      
      if (searchResults.length === 0) {
        vscode.window.showInformationMessage('No evidence found matching your query.');
        return;
      }

      vscode.window.showInformationMessage(
        `🔍 Found ${searchResults.length} evidence items matching "${query}"`
      );
    });

  } catch (error) {
    vscode.window.showErrorMessage(`Evidence search failed: ${error instanceof Error ? error.message : String(error)}`);
    console.error('Search error:', error);
  }
}

/**
 * Parse evidence SIMD JSON command
 */
async function parseEvidenceSimdJsonCommand(): Promise<void> {
  if (!simdJsonParser) {
    vscode.window.showErrorMessage('SIMD JSON Parser not initialized');
    return;
  }

  try {
    // Get JSON data from user
    const method = await vscode.window.showQuickPick([
      { label: '📁 Select JSON file', value: 'file' },
      { label: '📝 Paste JSON data', value: 'input' },
      { label: '📋 Use clipboard', value: 'clipboard' }
    ], {
      placeHolder: 'Choose JSON data source'
    });

    if (!method) return;

    let jsonData: string;

    switch (method.value) {
      case 'file':
        const fileUri = await vscode.window.showOpenDialog({
          canSelectFiles: true,
          canSelectMany: false,
          filters: { 'JSON Files': ['json'] }
        });
        
        if (!fileUri || fileUri.length === 0) return;
        
        const document = await vscode.workspace.openTextDocument(fileUri[0]);
        jsonData = document.getText();
        break;

      case 'input':
        const inputData = await vscode.window.showInputBox({
          placeHolder: 'Paste JSON data here...',
          prompt: 'Enter JSON evidence data for SIMD parsing'
        });
        
        if (!inputData) return;
        jsonData = inputData;
        break;

      case 'clipboard':
        jsonData = await vscode.env.clipboard.readText();
        if (!jsonData.trim()) {
          vscode.window.showErrorMessage('Clipboard is empty or contains no text');
          return;
        }
        break;

      default:
        return;
    }

    await vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'Parsing evidence with SIMD optimization...',
      cancellable: false
    }, async (progress) => {
      
      progress.report({ increment: 30, message: 'Initializing parser...' });
      
      // Initialize parser
      await simdJsonParser!.initialize();
      
      progress.report({ increment: 40, message: 'Parsing JSON data...' });
      
      // Parse evidence
      const extractionResult = await simdJsonParser!.extractEvidenceFromJson(jsonData);
      
      progress.report({ increment: 30, message: 'Complete!' });
      
      const successMsg = extractionResult.parseResult.success 
        ? `✅ Parsed ${extractionResult.evidence.length} evidence items in ${extractionResult.parseResult.processingTime}ms using ${extractionResult.parseResult.method}`
        : `❌ Parsing failed: ${extractionResult.parseResult.error}`;
      
      vscode.window.showInformationMessage(successMsg);
    });

  } catch (error) {
    vscode.window.showErrorMessage(`SIMD JSON parsing failed: ${error instanceof Error ? error.message : String(error)}`);
    console.error('SIMD parsing error:', error);
  }
}

export function deactivate() {
  // Cleanup services
  if (nomicEmbedService) {
    nomicEmbedService.dispose();
    nomicEmbedService = null;
  }

  if (neo4jService) {
    neo4jService.dispose();
    neo4jService = null;
  }

  if (simdJsonParser) {
    simdJsonParser.dispose();
    simdJsonParser = null;
  }
}