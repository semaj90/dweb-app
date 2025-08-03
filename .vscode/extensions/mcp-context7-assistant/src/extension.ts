import * as vscode from "vscode";
import { EmbeddingManager } from "./embeddingManager";
import { Neo4jService, SimdJsonService } from "./integrations";
import { MarkdownIndexer } from "./indexer";
import * as path from "path";

let embeddingManager: EmbeddingManager;
let neo4jService: Neo4jService;
let markdownIndexer: MarkdownIndexer;
let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
  console.log("🚀 Context7 MCP Assistant extension is now active!");

  // Create output channel for long-form results
  outputChannel = vscode.window.createOutputChannel("MCP Context7 Assistant");

  // Initialize services
  initializeServices();

  // Register new commands
  registerNewCommands(context);

  // Show activation message
  vscode.window.showInformationMessage(
    "Context7 MCP Assistant with Nomic Embed integration is ready!"
  );
}

export function deactivate() {
  console.log("🛑 Context7 MCP Assistant extension is deactivating...");

  if (neo4jService) {
    neo4jService.close().catch(console.error);
  }

  if (outputChannel) {
    outputChannel.dispose();
  }
}

function initializeServices() {
  const config = vscode.workspace.getConfiguration("mcpContext7");

  // Initialize Embedding Manager with Nomic Embed
  const nomicEmbedUrl = config.get("nomicEmbedUrl", "http://localhost:5000");
  embeddingManager = new EmbeddingManager(nomicEmbedUrl);

  // Initialize Neo4j Service
  const neo4jConfig = {
    url: config.get("neo4jUrl", "bolt://localhost:7687"),
    username: config.get("neo4jUsername", "neo4j"),
    password: config.get("neo4jPassword", "password")
  };
  neo4jService = new Neo4jService(neo4jConfig);

  // Initialize Markdown Indexer
  markdownIndexer = new MarkdownIndexer(embeddingManager, neo4jService);

  // Test connections
  testServiceConnections();
}

async function testServiceConnections() {
  try {
    // Test Nomic Embed connection
    const nomicHealthy = await embeddingManager.testConnection();
    if (nomicHealthy) {
      outputChannel.appendLine("✅ Nomic Embed server is healthy");
    } else {
      outputChannel.appendLine("❌ Nomic Embed server is not available");
    }

    // Test Neo4j connection
    await neo4jService.connect();
    outputChannel.appendLine("✅ Neo4j database connected successfully");
  } catch (error) {
    outputChannel.appendLine(`⚠️ Service connection error: ${error}`);
  }
}

function registerNewCommands(context: vscode.ExtensionContext) {
  // Embed VSCode Markdown Files command
  const embedVscodeMarkdownCommand = vscode.commands.registerCommand(
    "mcp.embedVscodeMarkdown",
    async () => {
      try {
        vscode.window.showInformationMessage("Starting batch embedding of VSCode markdown files...");

        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
          vscode.window.showErrorMessage("No workspace folder found");
          return;
        }

        const vscodeFolder = path.join(workspaceFolder.uri.fsPath, ".vscode");
        
        // Check if .vscode folder exists
        const vscodeUri = vscode.Uri.file(vscodeFolder);
        try {
          await vscode.workspace.fs.stat(vscodeUri);
        } catch {
          vscode.window.showErrorMessage(".vscode folder not found in workspace");
          return;
        }

        const results = await markdownIndexer.batchEmbedMarkdownFiles(vscodeFolder);
        
        const successCount = results.filter(r => r.success).length;
        const failureCount = results.filter(r => !r.success).length;

        const report = `# VSCode Markdown Embedding Results\n\n` +
          `- **Total files processed:** ${results.length}\n` +
          `- **Successfully embedded:** ${successCount}\n` +
          `- **Failed:** ${failureCount}\n\n` +
          `## Detailed Results\n\n` +
          results.map(r => 
            `- ${r.success ? '✅' : '❌'} ${r.file}${r.error ? ` - ${r.error}` : ''}\n`
          ).join('');

        showResultInOutputChannel(report);
        vscode.window.showInformationMessage(
          `Embedding complete: ${successCount} success, ${failureCount} failed`
        );

      } catch (error) {
        vscode.window.showErrorMessage(`Embedding failed: ${error}`);
        outputChannel.appendLine(`❌ Embedding error: ${error}`);
      }
    }
  );

  // Search Evidence command
  const searchEvidenceCommand = vscode.commands.registerCommand(
    "mcp.searchEvidence",
    async () => {
      try {
        const query = await vscode.window.showInputBox({
          prompt: "Enter search query for evidence",
          placeHolder: "e.g., legal precedents, contract clauses, documentation"
        });

        if (!query) {
          return;
        }

        vscode.window.showInformationMessage("Searching evidence...");

        const results = await markdownIndexer.searchEvidence(query, 0.7, 10);

        if (results.length === 0) {
          vscode.window.showInformationMessage("No evidence found matching your query");
          return;
        }

        const report = `# Evidence Search Results\n\n` +
          `**Query:** ${query}\n\n` +
          `**Found ${results.length} matches:**\n\n` +
          results.map((result, index) => 
            `## ${index + 1}. Similarity: ${(result.similarity * 100).toFixed(1)}%\n\n` +
            `**ID:** ${result.id}\n\n` +
            `**Text Preview:**\n${result.text.substring(0, 200)}...\n\n`
          ).join('---\n\n');

        showResultInOutputChannel(report);
        vscode.window.showInformationMessage(`Found ${results.length} evidence matches`);

      } catch (error) {
        vscode.window.showErrorMessage(`Search failed: ${error}`);
        outputChannel.appendLine(`❌ Search error: ${error}`);
      }
    }
  );

  // Parse Evidence SIMD JSON command
  const parseEvidenceSimdJsonCommand = vscode.commands.registerCommand(
    "mcp.parseEvidenceSimdJson",
    async () => {
      try {
        const fileUri = await vscode.window.showOpenDialog({
          canSelectFiles: true,
          canSelectFolders: false,
          canSelectMany: false,
          filters: {
            'JSON files': ['json']
          }
        });

        if (!fileUri || fileUri.length === 0) {
          return;
        }

        const filePath = fileUri[0].fsPath;
        vscode.window.showInformationMessage("Parsing JSON with SIMD...");

        const startTime = Date.now();
        const parsed = await markdownIndexer.parseEvidenceSimdjson(filePath);
        const endTime = Date.now();

        const report = `# SIMD JSON Parsing Results\n\n` +
          `**File:** ${filePath}\n` +
          `**Parse time:** ${endTime - startTime}ms\n` +
          `**Object keys:** ${Object.keys(parsed).length}\n\n` +
          `**Parsed content:**\n\`\`\`json\n${JSON.stringify(parsed, null, 2)}\n\`\`\``;

        showResultInOutputChannel(report);
        vscode.window.showInformationMessage(`JSON parsed successfully in ${endTime - startTime}ms`);

      } catch (error) {
        vscode.window.showErrorMessage(`JSON parsing failed: ${error}`);
        outputChannel.appendLine(`❌ JSON parsing error: ${error}`);
      }
    }
  );

  // Register all commands
  context.subscriptions.push(
    embedVscodeMarkdownCommand,
    searchEvidenceCommand,
    parseEvidenceSimdJsonCommand,
    outputChannel
  );
}

function showResultInOutputChannel(result: string) {
  outputChannel.clear();
  outputChannel.appendLine(result);
  outputChannel.show();
}