// MCP Server Integration - Gemma3 Legal Context
// File: mcp-legal-server.mjs

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import fetch from 'node-fetch';

class LegalMCPServer {
  constructor() {
    this.server = new Server({
      name: "legal-ai-server",
      version: "1.0.0"
    }, {
      capabilities: {
        tools: {},
        resources: {}
      }
    });

    this.setupTools();
    this.setupResources();
  }

  setupTools() {
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: "synthesize_evidence",
          description: "Synthesize multiple evidence items using Gemma3 Legal",
          inputSchema: {
            type: "object",
            properties: {
              evidenceIds: { type: "array", items: { type: "string" }},
              synthesisType: { type: "string", enum: ["correlation", "timeline", "compare", "merge"] },
              prompt: { type: "string" }
            }
          }
        },
        {
          name: "legal_rag_query",
          description: "Query legal knowledge base with enhanced RAG",
          inputSchema: {
            type: "object",
            properties: {
              query: { type: "string" },
              documentTypes: { type: "array", items: { type: "string" }},
              maxResults: { type: "number" }
            }
          }
        },
        {
          name: "get_case_summary",
          description: "Generate AI case summary using local legal LLM",
          inputSchema: {
            type: "object",
            properties: {
              caseId: { type: "string" },
              includeEvidence: { type: "boolean" }
            }
          }
        }
      ]
    }));

    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case "synthesize_evidence":
          return await this.synthesizeEvidence(args);
        case "legal_rag_query":
          return await this.legalRAGQuery(args);
        case "get_case_summary":
          return await this.getCaseSummary(args);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  setupResources() {
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        {
          uri: "legal://synthesis/api",
          name: "Evidence Synthesis API",
          description: "Direct access to evidence synthesis pipeline"
        },
        {
          uri: "legal://rag/studio",
          name: "RAG Studio Interface",
          description: "Enhanced RAG query interface"
        }
      ]
    }));

    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      return { contents: [{ uri, mimeType: "application/json", text: await this.getResource(uri) }] };
    });
  }

  async synthesizeEvidence(args) {
    const response = await fetch('http://localhost:5173/api/evidence/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        evidenceIds: args.evidenceIds,
        synthesisType: args.synthesisType || 'correlation',
        prompt: args.prompt,
        caseId: 'mcp-case',
        title: 'MCP Evidence Synthesis'
      })
    });

    const result = await response.json();
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  }

  async legalRAGQuery(args) {
    const response = await fetch('http://localhost:5173/api/enhanced-rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args)
    });

    const result = await response.json();
    return { content: [{ type: "text", text: result.analysis || JSON.stringify(result) }] };
  }

  async getCaseSummary(args) {
    const response = await fetch(`http://localhost:5173/api/cases/${args.caseId}/summary`, {
      method: 'GET'
    });

    const result = await response.json();
    return { content: [{ type: "text", text: result.summary || "Case summary unavailable" }] };
  }

  async getResource(uri) {
    switch (uri) {
      case "legal://synthesis/api":
        return JSON.stringify({ status: "active", endpoint: "/api/evidence/synthesize" });
      case "legal://rag/studio":
        return JSON.stringify({ status: "active", endpoint: "/api/enhanced-rag/query" });
      default:
        return "{}";
    }
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log("Legal MCP Server running");
  }
}

new LegalMCPServer().start().catch(console.error);
