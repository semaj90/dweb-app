import type { RequestHandler } from "@sveltejs/kit";

// MCP Context7.2 Get Library Docs utility
// Enhanced for Svelte 5 + Bits UI v2 integration
// Ensures #mcp_context72_get-library-docs is available in the codebase


export interface LibraryDocsRequest {
  context7CompatibleLibraryID: string;
  topic?: string;
  tokens?: number;
  format?: 'markdown' | 'json' | 'typescript';
}

export interface LibraryDocsResponse {
  content: string;
  metadata: {
    library: string;
    version?: string;
    topic?: string;
    tokenCount: number;
  };
  snippets?: {
    title: string;
    code: string;
    description: string;
  }[];
}

export async function mcpContext72GetLibraryDocs(
  libraryId: string,
  topic?: string,
  options: Partial<LibraryDocsRequest> = {},
  fetchFn: typeof fetch = fetch
): Promise<LibraryDocsResponse> {
  // Enhanced MCP context7.2 get-library-docs endpoint
  const response = await fetchFn("/api/mcp/context72/get-library-docs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      context7CompatibleLibraryID: libraryId, 
      topic,
      tokens: options.tokens || 10000,
      format: options.format || 'markdown',
      ...options
    }),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`Failed to get library docs from Context7.2: ${error.message || response.statusText}`);
  }
  
  return response.json();
}

// Specialized helpers for common libraries
export async function getSvelte5Docs(topic?: string, fetchFn?: typeof fetch): Promise<LibraryDocsResponse> {
  return mcpContext72GetLibraryDocs('/svelte/svelte', topic, { 
    format: 'typescript',
    tokens: 15000 
  }, fetchFn);
}

export async function getBitsUIv2Docs(topic?: string, fetchFn?: typeof fetch): Promise<LibraryDocsResponse> {
  return mcpContext72GetLibraryDocs('/bits-ui/bits-ui', topic, { 
    format: 'typescript',
    tokens: 12000 
  }, fetchFn);
}

export async function getMeltUIDocs(topic?: string, fetchFn?: typeof fetch): Promise<LibraryDocsResponse> {
  return mcpContext72GetLibraryDocs('/melt-ui/melt-ui', topic, { 
    format: 'typescript',
    tokens: 10000 
  }, fetchFn);
}

export async function getXStateDocs(topic?: string, fetchFn?: typeof fetch): Promise<LibraryDocsResponse> {
  return mcpContext72GetLibraryDocs('/xstate/xstate', topic, { 
    format: 'typescript',
    tokens: 8000 
  }, fetchFn);
}

// Tag: #mcp_context72_get-library-docs
// Keywords: #context7 #get-library-docs #svelte5 #bitsui #meltui #xstate
