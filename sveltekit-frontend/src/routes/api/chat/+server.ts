import { chatEmbeddings } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { desc, eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";
import VectorService from "$lib/server/services/vector-service";
import { ollamaService } from "$lib/services/ollama-service";
import type { RequestHandler } from "./$types";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  contextUsed?: any;
  suggestions?: string[];
  actions?: Array<{ type: string; text: string; data?: any }>;
}
interface ChatRequest {
  message: string;
  conversationId: string;
  userId: string;
  caseId?: string;
  mode?: "professional" | "investigative" | "evidence" | "strategic";
  useContext?: boolean;
  maxTokens?: number;
}
interface ChatResponse {
  success: boolean;
  message?: ChatMessage;
  conversation?: ChatMessage[];
  suggestions?: string[];
  actions?: Array<{ type: string; text: string; data?: any }>;
  contextUsed?: any;
  error?: string;
}
export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    // Get user session
    const sessionId = cookies.get("session_id");
    if (!sessionId) {
      return json(
        { success: false, error: "Authentication required" },
        { status: 401 },
      );
    }
    const body: ChatRequest = await request.json();
    const {
      message,
      conversationId,
      userId,
      caseId,
      mode = "professional",
      useContext = true,
      maxTokens = 1000,
    } = body;

    // Validate input
    if (!message || !conversationId || !userId) {
      return json(
        {
          success: false,
          error: "Missing required fields: message, conversationId, userId",
        },
        { status: 400 },
      );
    }
    // Generate embedding for the user message
    const messageEmbedding = await VectorService.generateEmbedding(message, {
      model: "ollama",
      userId,
      caseId,
      conversationId,
    });

    // Store user message embedding
    await VectorService.storeChatEmbedding({
      conversationId,
      userId,
      role: "user",
      content: message,
      embedding: messageEmbedding,
      metadata: { mode, caseId },
    });

    let contextData = null;
    let systemPrompt = getSystemPromptForMode(mode);

    // Retrieve relevant context if enabled
    if (useContext) {
      const similarContext = await VectorService.findSimilar(messageEmbedding, {
        limit: 3,
        threshold: 0.75,
        userId,
        caseId,
      });

      if (similarContext.length > 0) {
        contextData = {
          similar_cases: similarContext.filter(
            (c) => c.metadata?.type === "case",
          ),
          similar_evidence: similarContext.filter(
            (c) => c.metadata?.type === "evidence",
          ),
          previous_chats: similarContext.filter(
            (c) => c.metadata?.type === "chat",
          ),
        };

        // Add context to system prompt
        const contextText = similarContext
          .map(
            (c) =>
              `Context: ${c.content} (similarity: ${c.similarity.toFixed(2)})`,
          )
          .join("\n");

        systemPrompt += `\n\nRELEVANT CONTEXT:\n${contextText}`;
      }
    }
    // Get conversation history
    const conversationHistory = await db
      .select()
      .from(chatEmbeddings)
      .where(eq(chatEmbeddings.conversationId, conversationId))
      .orderBy(desc(chatEmbeddings.createdAt))
      .limit(10);

    // Build conversation context for LLM
    const conversationContext = conversationHistory
      .reverse()
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join("\n");

    // Generate response using Ollama
    const fullPrompt = `${systemPrompt}\n\nConversation History:\n${conversationContext}\n\nUser: ${message}\n\nAssistant:`;

    const response = await ollamaService.generateResponse(fullPrompt, {
      model: "gemma3-legal",
      maxTokens,
      temperature: 0.7,
      stream: false,
    });

    if (!response.success) {
      throw new Error(`LLM generation failed: ${response.error}`);
    }
    // Generate response embedding and store
    const responseEmbedding = await VectorService.generateEmbedding(
      response.content,
      {
        model: "ollama",
        userId,
        caseId,
        conversationId,
      },
    );

    await VectorService.storeChatEmbedding({
      conversationId,
      userId,
      role: "assistant",
      content: response.content,
      embedding: responseEmbedding,
      metadata: { mode, caseId, contextUsed: contextData },
    });

    // Generate suggestions and actions
    const suggestions = generateSuggestions(response.content, mode);
    const actions = generateActions(response.content, mode, caseId);

    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: response.content,
      timestamp: new Date(),
      contextUsed: contextData,
      suggestions,
      actions,
    };

    return json({
      success: true,
      message: assistantMessage,
      suggestions,
      actions,
      contextUsed: contextData,
    } as ChatResponse);
  } catch (error) {
    console.error("Chat API error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      } as ChatResponse,
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async ({ url, cookies }) => {
  try {
    // Get user session
    const sessionId = cookies.get("session_id");
    if (!sessionId) {
      return json(
        { success: false, error: "Authentication required" },
        { status: 401 },
      );
    }
    const conversationId = url.searchParams.get("conversationId");
    const userId = url.searchParams.get("userId");
    const limit = parseInt(url.searchParams.get("limit") || "50");

    if (!conversationId || !userId) {
      return json(
        {
          success: false,
          error: "conversationId and userId required",
        },
        { status: 400 },
      );
    }
    // Get conversation history
    const messages = await db
      .select()
      .from(chatEmbeddings)
      .where(eq(chatEmbeddings.conversationId, conversationId))
      .orderBy(desc(chatEmbeddings.createdAt))
      .limit(limit);

    const conversation: ChatMessage[] = messages.reverse().map((msg) => ({
      id: msg.id,
      role: msg.role as "user" | "assistant",
      content: msg.content,
      timestamp: msg.createdAt,
      contextUsed: msg.metadata?.contextUsed,
      suggestions: msg.metadata?.suggestions,
      actions: msg.metadata?.actions,
    }));

    return json({
      success: true,
      conversation,
    } as ChatResponse);
  } catch (error) {
    console.error("Chat history API error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      } as ChatResponse,
      { status: 500 },
    );
  }
};

function getSystemPromptForMode(mode: string): string {
  const basePrompt =
    "You are an expert legal AI assistant specializing in prosecutor case management.";

  switch (mode) {
    case "professional":
      return `${basePrompt} Provide formal, precise legal analysis with proper citations and structured responses.`;

    case "investigative":
      return `${basePrompt} Focus on deep case analysis, evidence relationships, and investigative strategies. Ask probing questions and identify gaps.`;

    case "evidence":
      return `${basePrompt} Concentrate on evidence analysis, chain of custody, admissibility, and evidentiary standards. Provide detailed evidence-focused guidance.`;

    case "strategic":
      return `${basePrompt} Emphasize case strategy, trial preparation, and tactical considerations. Provide actionable strategic recommendations.`;

    default:
      return basePrompt;
  }
}
function generateSuggestions(content: string, mode: string): string[] {
  const suggestions: string[] = [];

  // Basic suggestions based on mode
  switch (mode) {
    case "professional":
      suggestions.push(
        "Request additional legal precedents",
        "Review applicable statutes",
        "Analyze case citations",
      );
      break;
    case "investigative":
      suggestions.push(
        "Examine witness statements",
        "Analyze timeline discrepancies",
        "Identify evidence gaps",
      );
      break;
    case "evidence":
      suggestions.push(
        "Review chain of custody",
        "Assess admissibility",
        "Identify corroborating evidence",
      );
      break;
    case "strategic":
      suggestions.push(
        "Develop trial strategy",
        "Prepare opening arguments",
        "Plan witness examination",
      );
      break;
  }
  // Add content-specific suggestions
  if (content.toLowerCase().includes("evidence")) {
    suggestions.push("Analyze evidence relationships");
  }
  if (content.toLowerCase().includes("witness")) {
    suggestions.push("Review witness credibility");
  }
  if (content.toLowerCase().includes("statute")) {
    suggestions.push("Research related case law");
  }
  return suggestions.slice(0, 3); // Limit to 3 suggestions
}
function generateActions(
  content: string,
  mode: string,
  caseId?: string,
): Array<{ type: string; text: string; data?: any }> {
  const actions: Array<{ type: string; text: string; data?: any }> = [];

  // Mode-specific actions
  switch (mode) {
    case "evidence":
      actions.push({
        type: "analyze_evidence",
        text: "Analyze Evidence",
        data: { caseId },
      });
      break;
    case "investigative":
      actions.push({
        type: "investigate_leads",
        text: "Investigate Leads",
        data: { caseId },
      });
      break;
    case "strategic":
      actions.push({
        type: "create_strategy",
        text: "Create Strategy Document",
        data: { caseId },
      });
      break;
  }
  // Universal actions
  actions.push({
    type: "save_summary",
    text: "Save Summary",
    data: { caseId },
  });

  if (caseId) {
    actions.push({
      type: "add_to_case",
      text: "Add to Case File",
      data: { caseId },
    });
  }
  return actions;
}
