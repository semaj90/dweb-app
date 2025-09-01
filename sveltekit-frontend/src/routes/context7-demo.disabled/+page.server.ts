import path from 'path';
import type { PageServerLoad } from './$types.js';
import { copilotSelfPrompt } from '$lib/utils/copilot-self-prompt';

import { resolveLibraryId, getLibraryDocs } from '$lib/services/context7-lib-resolver';
import fs from 'fs/promises';
import { URL } from "url";

// Load Copilot architecture context for enhanced prompting
async function loadCopilotContext(): Promise<Record<string, string | null>> {
  const contextFiles = [
    "markdown_files/copilot-architecture-summary.md",
    "markdown_files/copilot-context.md",
    "copilot.md",
  ];

  const context: Record<string, string | null> = {};
  for (const file of contextFiles) {
    try {
      const fullPath = path.join(process.cwd(), "..", file);
      const content = await fs.readFile(fullPath, "utf-8");
      const key = path.basename(file, ".md");
      context[key] = content;
    } catch (err: any) {
      console.warn(`Could not load ${file}:`, (err as Error).message);
      context[path.basename(file, ".md")] = null;
    }
  }
  return context;
}

export const load: PageServerLoad = async ({ url }) => {
  const library = url.searchParams.get("lib") || "sveltekit";
  const topic = url.searchParams.get("topic") || "overview";
  const prompt =
    url.searchParams.get("prompt") ||
    "Analyze legal AI workflow with SvelteKit";

  let libraryData = null;
  let copilotContext = null;
  let orchestrationResult = null;

  try {
    // Load Copilot architecture context
    copilotContext = await loadCopilotContext();

    // Pre-resolve library on server for SSR
    const libId = await resolveLibraryId(library);
    const docs = await getLibraryDocs(libId, topic);

    libraryData = {
      library,
      libId,
      topic,
      docs: docs.substring(0, 1000), // Truncate for SSR
      timestamp: new Date().toISOString(),
    };

    // Enhanced: Run Copilot self-prompt orchestration on server if requested
    if (url.searchParams.get("orchestrate") === "true") {
      orchestrationResult = await copilotSelfPrompt(prompt, {
        useSemanticSearch: true,
        useMemory: true,
        useMultiAgent: false, // Disable heavy multi-agent for SSR performance
        context: {
          projectPath: process.cwd(),
          platform: "webapp" as const,
          urgency: "medium" as const,
          includeTests: false,
        },
      });

      // Store architecture context for client hydration
      if (orchestrationResult) {
        (orchestrationResult as any).copilotArchitecture =
          copilotContext["copilot-architecture-summary"];
        (orchestrationResult as any).legalContext =
          copilotContext["copilot-context"];
      }

      // Truncate large result for SSR transfer
      if (orchestrationResult) {
        orchestrationResult.synthesizedOutput =
          orchestrationResult.synthesizedOutput?.substring(0, 500) + "...";
        orchestrationResult.contextResults =
          orchestrationResult.contextResults?.slice(0, 3);
      }
    }
  } catch (error: any) {
    console.error("SSR Context7 error:", error);
    libraryData = {
      library,
      libId: null,
      topic,
      docs: null,
      error: (error as Error).message,
      timestamp: new Date().toISOString(),
    };
  }

  return {
    libraryData,
    copilotContext: {
      architecture: copilotContext?.["copilot-architecture-summary"]?.substring(
        0,
        500,
      ),
      legalContext: copilotContext?.["copilot-context"]?.substring(0, 500),
      reference: copilotContext?.["copilot"],
    },
    orchestrationResult,
    seo: {
      title: `Context7 Demo - ${library} ${topic}`,
      description:
        "Production Context7 MCP integration with Copilot orchestration for legal AI workflows",
    },
  };
};
