import { json } from "@sveltejs/kit";
import VectorService from "$lib/server/services/vector-service";
import type { RequestHandler } from "./$types";

const vectorService = new VectorService();

interface EmbedRequest {
  text: string;
  type: "user_context" | "chat_message" | "evidence" | "case_summary";
  metadata?: {
    userId?: string;
    caseId?: string;
    conversationId?: string;
    evidenceId?: string;
    category?: string;
    [key: string]: any;
  };
  model?: "openai" | "ollama";
}
interface EmbedResponse {
  success: boolean;
  id?: string;
  vector?: number[];
  similarity_results?: Array<{
    id: string;
    content: string;
    similarity: number;
    metadata: any;
    createdAt: Date;
  }>;
  error?: string;
}
export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    // Get user session (implement your auth logic here)
    const sessionId = cookies.get("session_id");
    if (!sessionId) {
      return json(
        { success: false, error: "Authentication required" },
        { status: 401 },
      );
    }
    const body: EmbedRequest = await request.json();
    const { text, type, metadata = {}, model = "ollama" } = body;

    // Validate input
    if (!text || !type) {
      return json(
        {
          success: false,
          error: "Missing required fields: text and type",
        },
        { status: 400 },
      );
    }
    // Generate embedding
    const vector = await VectorService.generateEmbedding(text, {
      model,
      userId: metadata.userId,
      caseId: metadata.caseId,
      conversationId: metadata.conversationId,
    });

    let insertId: string;

    // Store embedding based on type
    switch (type) {
      case "user_context":
        if (!metadata.userId) {
          return json(
            {
              success: false,
              error: "userId required for user_context embeddings",
            },
            { status: 400 },
          );
        }
        const userResult = await VectorService.storeUserEmbedding(
          metadata.userId,
          text,
          vector,
          {
            metadata: metadata,
          } as any
        );
        insertId = userResult;
        break;

      case "chat_message":
        if (!metadata.conversationId) {
          return json(
            {
              success: false,
              error: "conversationId required for chat_message embeddings",
            },
            { status: 400 },
          );
        }
        await VectorService.storeChatEmbedding({
          conversationId: metadata.conversationId,
          userId: metadata.userId || "anonymous",
          role: metadata.role || "user",
          content: text,
          embedding: vector,
          metadata: metadata,
        });
        insertId = `chat_${Date.now()}`;
        break;

      case "evidence":
        if (!metadata.evidenceId) {
          return json(
            {
              success: false,
              error: "evidenceId required for evidence embeddings",
            },
            { status: 400 },
          );
        }
        await VectorService.storeEvidenceVector({
          evidenceId: metadata.evidenceId,
          caseId: metadata.caseId,
          content: text,
          embedding: vector,
          vectorType: metadata.category || "text",
          metadata: metadata,
        });
        insertId = `evidence_${Date.now()}`;
        break;

      case "case_summary":
        if (!metadata.caseId) {
          return json(
            {
              success: false,
              error: "caseId required for case_summary embeddings",
            },
            { status: 400 },
          );
        }
        await VectorService.storeCaseEmbedding({
          caseId: metadata.caseId,
          content: text,
          embedding: vector,
          summary_type: metadata.category || "general",
          metadata: metadata,
        });
        insertId = `case_${Date.now()}`;
        break;

      default:
        return json(
          {
            success: false,
            error: `Unsupported embedding type: ${type}`,
          },
          { status: 400 },
        );
    }
    // Find similar content for context
    const similarityResults = await VectorService.findSimilar(vector, {
      limit: 5,
      threshold: 0.7,
      userId: metadata.userId,
      caseId: metadata.caseId,
    });

    return json({
      success: true,
      id: insertId,
      vector: vector,
      similarity_results: similarityResults,
    } as EmbedResponse);
  } catch (error) {
    console.error("Embedding API error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      } as EmbedResponse,
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
    const queryVector = url.searchParams.get("vector");
    const userId = url.searchParams.get("userId");
    const caseId = url.searchParams.get("caseId");
    const limit = parseInt(url.searchParams.get("limit") || "10");
    const threshold = parseFloat(url.searchParams.get("threshold") || "0.7");

    if (!queryVector) {
      return json(
        {
          success: false,
          error: "Query vector required",
        },
        { status: 400 },
      );
    }
    const vector = JSON.parse(queryVector) as number[];

    const similarityResults = await VectorService.findSimilar(vector, {
      limit,
      threshold,
      userId,
      caseId,
    });

    return json({
      success: true,
      similarity_results: similarityResults,
    } as EmbedResponse);
  } catch (error) {
    console.error("Similarity search error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      } as EmbedResponse,
      { status: 500 },
    );
  }
};
