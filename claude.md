r

---

## 🚀 Context7 MCP Self-Prompting & Agent Orchestration Guide

### What is Self-Prompting?

Self-prompting is the process of automatically generating the next best action or analysis step by synthesizing results from:

- Semantic search (Context7 MCP)
- Memory graph (knowledge graph, user/system history)
- Multi-agent orchestration (AutoGen, CrewAI, Copilot, Claude, VLLM, etc)
- Codebase and error analysis

### How to Use Self-Prompting in This Project

1. **Orchestration Entry Point:**

   - Use `copilotOrchestrator(prompt, options)` from `src/lib/utils/mcp-helpers.ts` to run a full self-prompting workflow.
   - Options include toggling memory, semantic search, codebase, multi-agent, error log, and synthesis steps.

2. **Prompt Utilities:**

   - See `generateMCPPrompt` in `mcp-helpers.ts` for building natural language prompts for Context7 tools.
   - Use `copilot-self-prompt.ts` for advanced synthesis, next-action generation, and fallback summaries.

3. **Types & Integration:**

   - Types for agent triggers, audit logs, and orchestration are in `src/lib/ai/types.ts`.
   - See TODOs in `types.ts` for wiring up real Context7 semantic search, logging, and agent triggers using MCP tools.

4. **Keyword Shortcuts:**

   - Use #context7, #semantic_search, #mcp_memory2_create_relations, #get-library-docs, #memory, #mcp_context72_resolve-library-id, #mcp_microsoft-doc_microsoft_docs_search, #get_vscode_api in Copilot/Claude prompts to trigger automation.
   - See `context7-keywords.md` for a full list of supported keywords.

5. **Example Self-Prompt Flow:**

   - Compose a prompt (e.g. "Analyze evidence upload errors and suggest fixes")
   - Call `copilotOrchestrator(prompt, { useSemanticSearch: true, useMemory: true, useMultiAgent: true, synthesizeOutputs: true })`
   - Review the synthesized result and actionable next steps.

6. **Want to automate more?**
   - Add new agent integrations or prompt flows in `mcp-helpers.ts` and `copilot-self-prompt.ts`.
   - Use the #want keyword in your prompt to request new automations or flows.

---

For more details, see `src/lib/utils/mcp-helpers.ts`, `src/lib/utils/copilot-self-prompt.ts`, and `src/lib/ai/types.ts`.

---

## Context7 MCP Assistant Quickstart & Troubleshooting Guide

### Quickstart

1. **Install & Build Extension**

   - Ensure `.vscode/extensions/mcp-context7-assistant/` exists.
   - Run `npm install` and `npm run compile` in the extension directory.
   - Reload VS Code to activate the extension.

2. **Custom Port & Logging**

   - The MCP server is configured to run on port **40000** (not 3000).
   - Debug-level logging is enabled for troubleshooting.
   - These are set in `.vscode/settings.json`:
     ```jsonc
     "mcpContext7.serverPort": 40000,
     "mcpContext7.logLevel": "debug"
     ```

3. **Basic Usage**

   - Use the Command Palette (Ctrl+Shift+P) and search for `MCP` to access:
     - Analyze Current Context
     - Suggest Best Practices
     - Get Context-Aware Documentation
     - Start/Stop MCP Server
   - The extension auto-detects SvelteKit/Legal AI context and provides tailored suggestions.

4. **Context7/Memory/Keyword Usage**

   - Use keywords in prompts and code:
     - `#context7`, `#get-library-docs`, `#resolve-library-id`, `#mcp_memory2_create_relations`, `#mcp_context72_get-library-docs`, `#mcp_context72_resolve-library-id`, `#mcp_microsoft-doc_microsoft_docs_search`, `#mcp_memory2_read_graph`, `#memory`, `#codebase`, `#get_vscode_api`
   - See `src/lib/tracking/context7-utils.js` for prompt and utility examples.

5. **Integrating with SvelteKit/Legal AI**
   - Use helpers in `src/lib/ai/mcp-helpers.ts` and `src/lib/services/context7Service.ts` for programmatic access.
   - Types for semantic search, logging, and agent triggers are in `src/lib/ai/types.ts`.

### Troubleshooting

- **Server Not Starting or Wrong Port:**

  - Confirm `mcpContext7.serverPort` is set to `40000` in `.vscode/settings.json`.
  - Reload VS Code after changing settings.
  - Check for port conflicts (ensure nothing else is using 40000).

- **No Logs or Insufficient Debug Info:**

  - Ensure `mcpContext7.logLevel` is set to `debug`.
  - Check the VS Code Output panel for `Context7 MCP Assistant` logs.

- **Extension Not Responding:**

  - Rebuild the extension (`npm run compile` in the extension folder).
  - Reload VS Code.
  - Check for errors in the Output panel or Developer Tools (Help > Toggle Developer Tools).

- **API/Integration Issues:**
  - Review helper usage in `src/lib/ai/mcp-helpers.ts` and `src/lib/services/context7Service.ts`.
  - Ensure the MCP server endpoint in `context7Service.ts` matches the configured port (should be `http://localhost:40000/mcp`).

---

## 🧠 Automating Evidence Analysis, Memory Graph, and Multi-Agent Orchestration

You can automate advanced flows (evidence analysis, memory graph updates, multi-agent orchestration) using:

### 1. VS Code Extensions (Context7 MCP, Copilot, Custom)

- Use the **Context7 MCP Assistant** and **Copilot** extensions for:
  - Self-prompting (see `copilot-self-prompt.ts`)
  - Semantic search, memory, and agent orchestration
  - Triggering flows via Command Palette or custom commands
- Example: Run `MCP: Analyze Current Context` or use a custom command to trigger a multi-agent workflow.

### 2. Sub-Agents & Orchestration in Codebase

- Use `copilotOrchestrator` and `copilotSelfPrompt` (see `src/lib/utils/mcp-helpers.ts` and `copilot-self-prompt.ts`):
  - Compose prompts like: `"Analyze evidence upload errors and suggest fixes"`
  - Options: `{ useSemanticSearch: true, useMemory: true, useMultiAgent: true, synthesizeOutputs: true }`
  - Sub-agents (AutoGen, CrewAI, Claude, VLLM) are orchestrated automatically.
- Example:
  ```ts
  const result = await copilotSelfPrompt("Analyze new evidence for case 123", {
    useSemanticSearch: true,
    useMemory: true,
    useMultiAgent: true,
    synthesizeOutputs: true,
  });
  // result contains synthesis, next steps, and recommendations
  ```

### 3. Memory Graph Updates

- Use the MCP memory API (see `accessMemoryMCP` in `copilot-self-prompt.ts`) to query/update the knowledge graph.
- Example:
  ```ts
  const memories = await accessMemoryMCP("Show all evidence for case 123", {
    caseId: 123,
  });
  ```

### 4. Evidence Analysis API

- Use the `/api/multi-agent/analyze` endpoint (see `src/routes/api/multi-agent/analyze/+server.ts`) to trigger full multi-agent evidence analysis from the frontend or extension.
- Example:
  ```ts
  const response = await fetch("/api/multi-agent/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      caseId: "123",
      evidenceContent: "...",
      evidenceTitle: "...",
      evidenceType: "document",
    }),
  });
  const { evidenceAnalysis, caseSynthesis } = await response.json();
  ```

### 5. Automate More Flows (#want)

- Use the `#want` keyword in prompts or issues to request new automations (e.g., new sub-agent flows, custom evidence analyzers, or memory graph integrations).
- See `copilot-self-prompt.ts` and `mcp-helpers.ts` for extensible orchestration patterns.

---

**See also:**

- `copilot-context.md` for Copilot/Context7 integration
- `src/lib/utils/copilot-self-prompt.ts` for orchestration logic
- `src/routes/api/multi-agent/analyze/+server.ts` for backend agent flows
- `frontendshadunocss.md` for UI/SSR best practices

---

## Reference: Shadcn-Svelte + UnoCSS Styling Guide

See `frontendshadunocss.md` for up-to-date best practices on integrating shadcn-svelte and UnoCSS, shared UI patterns, and SSR/hydration with SvelteKit 2, XState, Superforms, Drizzle ORM, and pgvector.

## CRITICAL: Current TypeScript Check Issue (2025-07-28)

**STATUS**: ⚠️ TypeScript check (`npm run check`) hangs/times out during "Getting Svelte diagnostics..." phase

**CONTEXT**: UnoCSS theme preprocessing errors have been resolved, but full TypeScript project checking fails to complete. See `phase-error-analysis-20250728-184109.md` for complete analysis.

**URGENT ACTIONS NEEDED**:

1. Resolve TypeScript check hanging issue
2. Investigate memory/performance bottlenecks
3. Implement incremental checking strategy
4. Fix remaining syntax errors in non-critical files

**WORKING ELEMENTS**: Lint checks, UnoCSS preprocessing, individual component compilation
**BLOCKING**: Full TypeScript validation, build process verification

## Advanced Self-Prompting, Memory, and Agent Orchestration (Context7/MCP)

- **Self-Prompting:**

  - Use the `copilot-self-prompt.ts` utility (`src/lib/utils/copilot-self-prompt.ts`) for orchestrating semantic search, memory graph, and multi-agent workflows.
  - Supports options for semantic search, memory, multi-agent, and autonomous engineering. See the `CopilotSelfPromptOptions` and `CopilotSelfPromptResult` types in `src/lib/ai/types.ts`.
  - Example usage:
    ```typescript
    import { copilotSelfPrompt } from "$lib/utils/copilot-self-prompt";
    const result = await copilotSelfPrompt(
      "How do I integrate Bits UI dialog?",
      { useSemanticSearch: true, useMemory: true }
    );
    ```
  - The result includes context, memory, agent results, recommendations, and a synthesized self-prompt for next actions.

- **Prompt Utilities:**

  - See `src/lib/tracking/context7-utils.js` for:
    - Context7/Bits UI/SvelteKit doc queries
    - Memory/keyword prompt templates
    - Example: `Context7Helper.getBitsUIDoc("dialog")` or `mcpUtils.createEntity("LegalCase", "entity", ["A legal case entity"])`

- **Agent Orchestration:**

  - Use `copilotOrchestrator` in `src/lib/utils/mcp-helpers.ts` for advanced workflows:
    - Dynamically select agents (autogen, crewai, copilot, claude, vllm)
    - Compose results from semantic/codebase/memory/agent analysis
    - Example:
      ```typescript
      import { copilotOrchestrator } from "$lib/utils/mcp-helpers";
      const results = await copilotOrchestrator(
        "Summarize precedent search best practices",
        { useSemanticSearch: true, useMemory: true, useMultiAgent: true }
      );
      ```
    - See the `OrchestrationOptions` and `AgentResult` types for extensibility.

- **MCP/Context7 Keywords:**

  - Use these in prompts, code, or documentation to trigger advanced features:
    - `#context7`, `#get-library-docs`, `#resolve-library-id`, `#mcp_memory2_create_relations`, `#semantic_search`, `#mcp_context72_get-library-docs`, `#memory`, `#mcp_microsoft-doc_microsoft_docs_search`, `#mcp_memory2_read_graph`, `#get_vscode_api`

- **Best Practices:**

  - Always validate MCP tool requests using `validateMCPRequest` (see `mcp-helpers.ts`).
  - Use prompt generators for consistent queries (see `generateMCPPrompt`).
  - For new features, add agent stubs to the registry in `mcp-helpers.ts`.

- **See Also:**
  - TODOs and type definitions in `src/lib/ai/types.ts` for extending orchestration and prompt logic.
  - Example prompt patterns and memory/graph usage in `context7-utils.js`.

## Memory: Debugging Workflow

- `npm run check` entire app, attempt to read log file of errors and then prioritize critical to easiest to fix and solve iteratively using:
  - Systematic log file parsing
  - Error triage and categorization
  - Incremental troubleshooting approach
  - Focus on resolving critical path issues first