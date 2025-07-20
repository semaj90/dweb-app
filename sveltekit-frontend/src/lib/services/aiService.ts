import { writable } from "svelte/store";

interface AIState {
  isLoading: boolean;
  summary: string | null;
  error: string | null;
  lastSummarizedContent: string | null;
  model: string;
}
interface SummarizeRequest {
  content: string;
  type?: "report" | "evidence" | "poi" | "general";
  caseId?: string;
  sourceId?: string;
}
interface SummarizeResponse {
  summary: string;
  model: string;
  processingTime: number;
  confidence?: number;
}
function createAIService() {
  const { subscribe, set, update } = writable<AIState>({
    isLoading: false,
    summary: null,
    error: null,
    lastSummarizedContent: null,
    model: "gemma3-legal",
  });

  return {
    subscribe,

    /**
     * Summarize any content using the local LLM
     */
    summarize: async (request: SummarizeRequest): Promise<string | null> => {
      if (!request.content || request.content.trim().length === 0) {
        update((state) => ({
          ...state,
          error: "No content provided to summarize",
        }));
        return null;
      }
      // Set loading state
      update((state) => ({
        ...state,
        isLoading: true,
        summary: null,
        error: null,
        lastSummarizedContent: request.content,
      }));

      try {
        // Call the SvelteKit API endpoint that wraps Ollama
        const response = await fetch("/api/ai/summarize", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            content: request.content,
            type: request.type || "general",
            caseId: request.caseId,
            sourceId: request.sourceId,
            model: "gemma3-legal", // Use our custom legal model
          }),
        });

        if (!response.ok) {
          const errorData = await response
            .json()
            .catch(() => ({ error: "Unknown error" }));
          throw new Error(
            errorData.error ||
              `HTTP ${response.status}: ${response.statusText}`,
          );
        }
        const data: SummarizeResponse = await response.json();

        // Update state with successful result
        update((state) => ({
          ...state,
          isLoading: false,
          summary: data.summary,
          error: null,
          model: data.model,
        }));

        return data.summary;
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to generate summary";

        update((state) => ({
          ...state,
          isLoading: false,
          summary: null,
          error: errorMessage,
        }));

        console.error("AI summarization error:", err);
        return null;
      }
    },

    /**
     * Summarize a report's content
     */
    summarizeReport: async (
      reportContent: any,
      reportId?: string,
      caseId?: string,
    ): Promise<string | null> => {
      // Extract plain text from Slate.js content structure
      const plainText = extractTextFromSlateContent(reportContent);

      return await aiService.summarize({
        content: plainText,
        type: "report",
        sourceId: reportId,
        caseId: caseId,
      });
    },

    /**
     * Summarize evidence
     */
    summarizeEvidence: async (
      evidence: { title: string; description?: string; aiAnalysis?: any },
      evidenceId?: string,
      caseId?: string,
    ): Promise<string | null> => {
      let content = evidence.title;
      if (evidence.description) {
        content += "\n\nDescription: " + evidence.description;
      }
      if (evidence.aiAnalysis && typeof evidence.aiAnalysis === "object") {
        content +=
          "\n\nAI Analysis: " + JSON.stringify(evidence.aiAnalysis, null, 2);
      }
      return await aiService.summarize({
        content: content,
        type: "evidence",
        sourceId: evidenceId,
        caseId: caseId,
      });
    },

    /**
     * Summarize a Person of Interest profile
     */
    summarizePOI: async (
      poiData: { name: string; profileData: any },
      poiId?: string,
      caseId?: string,
    ): Promise<string | null> => {
      const profileData = poiData.profileData || {};
      let content = `Person of Interest: ${poiData.name}\n\n`;

      if (profileData.who) content += `Who: ${profileData.who}\n\n`;
      if (profileData.what) content += `What: ${profileData.what}\n\n`;
      if (profileData.why) content += `Why: ${profileData.why}\n\n`;
      if (profileData.how) content += `How: ${profileData.how}\n\n`;

      return await aiService.summarize({
        content: content,
        type: "poi",
        sourceId: poiId,
        caseId: caseId,
      });
    },

    /**
     * Reset the AI state
     */
    reset: () => {
      set({
        isLoading: false,
        summary: null,
        error: null,
        lastSummarizedContent: null,
        model: "gemma3-legal",
      });
    },

    /**
     * Clear just the summary and error
     */
    clearResults: () => {
      update((state) => ({
        ...state,
        summary: null,
        error: null,
      }));
    },
  };
}
/**
 * Extract plain text from Slate.js content structure
 */
function extractTextFromSlateContent(nodes: any[]): string {
  if (!Array.isArray(nodes)) return "";

  const extractFromNode = (node: any): string => {
    if (typeof node === "string") return node;

    if (node.text !== undefined) {
      return node.text;
    }
    if (node.children && Array.isArray(node.children)) {
      return node.children.map(extractFromNode).join("");
    }
    return "";
  };

  return nodes.map(extractFromNode).join("\n");
}
// Export the singleton instance
export const aiService = createAIService();

// Export types for use in components
export type { SummarizeRequest, SummarizeResponse, AIState };
