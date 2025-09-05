// @ts-nocheck
// MCP Context7.2 Get Library Docs utility
// Ensures #mcp_context72_get-library-docs is available in the codebase

import type { RequestHandler } from "@sveltejs/kit";

export async function mcpContext72GetLibraryDocs(
  libraryId: string,
  topic?: string,
): Promise<any> {
  // Example: Call MCP context7.2 get-library-docs endpoint
  const response = await fetch("/api/mcp/context72/get-library-docs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context7CompatibleLibraryID: libraryId, topic }),
  });
  if (!response.ok)
    throw new Error("Failed to get library docs from Context7.2");
  return response.json();
}

// Tag: #mcp_context72_get-library-docs
