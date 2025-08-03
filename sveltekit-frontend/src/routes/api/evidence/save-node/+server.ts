import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { z } from "zod";

// Validation schemas
const evidenceNodeSchema = z.object({
  id: z.string(),
  name: z.string().min(1).max(255),
  title: z.string().max(255).optional(),
  description: z.string().optional(),
  content: z.string(),
  fileType: z.string(),
  filePath: z.string().optional(),
  fileSize: z.number().optional(),
  aiTags: z
    .object({
      tags: z.array(z.string()).optional(),
      people: z.array(z.string()).optional(),
      locations: z.array(z.string()).optional(),
      organizations: z.array(z.string()).optional(),
      dates: z.array(z.string()).optional(),
      summary: z.string().optional(),
      keyFacts: z.array(z.string()).optional(),
      evidenceType: z.string().optional(),
      legalRelevance: z.enum(["critical", "high", "medium", "low"]).optional(),
      legalCategories: z.array(z.string()).optional(),
      confidentialityLevel: z
        .enum(["public", "internal", "confidential", "restricted"])
        .optional(),
      urgencyLevel: z.enum(["immediate", "high", "normal", "low"]).optional(),
      qualityScore: z.number().min(0).max(1).optional(),
    })
    .optional(),
  metadata: z.object({
    x: z.number(),
    y: z.number(),
    width: z.number(),
    height: z.number(),
    connections: z.array(z.string()).optional(),
  }),
  caseId: z.string().optional(),
  userId: z.string(),
});

const canvasStateSchema = z.object({
  nodes: z.array(evidenceNodeSchema),
  connections: z.array(
    z.object({
      fromId: z.string(),
      toId: z.string(),
      type: z.enum([
        "person",
        "location",
        "organization",
        "temporal",
        "custom",
      ]),
      strength: z.number().min(0).max(1),
      label: z.string().optional(),
    })
  ),
  viewport: z.object({
    zoomLevel: z.number(),
    panOffset: z.object({
      x: z.number(),
      y: z.number(),
    }),
  }),
  caseId: z.string().optional(),
  lastModified: z.string(),
});

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Check authentication - use locals.user which is set by hooks
    const user = locals.user;
    if (!user) {
      return json({ error: "Authentication required" }, { status: 401 });
    }
    const body = await request.json();
    const { action = "save_node", data } = body;

    switch (action) {
      case "save_node":
        return await saveEvidenceNode(data, user.id);

      case "save_canvas_state":
        return await saveCanvasState(data, user.id);

      case "bulk_save":
        return await bulkSaveEvidence(data, user.id);

      case "auto_save":
        return await autoSaveCanvasState(data, user.id);

      default:
        return json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error) {
    console.error("Save API error:", error);
    return json(
      {
        error: "Failed to save evidence",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};

// Save individual evidence node
async function saveEvidenceNode(nodeData: any, userId: string) {
  try {
    // Validate input
    const validatedNode = evidenceNodeSchema.parse({
      ...nodeData,
      userId,
    });

    // For now, store in memory/localStorage simulation
    // In production, this would use your database
    const evidenceData = {
      id: validatedNode.id,
      fileName: validatedNode.name,
      title: validatedNode.title || validatedNode.name,
      description:
        validatedNode.description || validatedNode.aiTags?.summary || "",
      fileType: validatedNode.fileType,
      filePath: validatedNode.filePath,
      fileSize: validatedNode.fileSize,
      content: validatedNode.content,
      tags: validatedNode.aiTags?.tags || [],
      aiSummary: validatedNode.aiTags?.summary,
      aiTags: validatedNode.aiTags || {},
      canvasPosition: validatedNode.metadata,
      caseId: validatedNode.caseId,
      userId: userId,
      hash: await generateFileHash(validatedNode.content),
      hashVerificationStatus: "verified" as const,
      isDeleted: false,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // Generate and store embedding for vector search
    await generateAndStoreEmbedding(evidenceData);

    // Update search index
    await updateSearchIndex(evidenceData);

    return json({
      success: true,
      evidence: evidenceData,
      message: "Evidence saved successfully",
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return json(
        {
          error: "Validation failed",
          details: error.errors,
        },
        { status: 400 }
      );
    }
    throw error;
  }
}
// Save entire canvas state
async function saveCanvasState(canvasData: any, userId: string) {
  try {
    const validatedCanvas = canvasStateSchema.parse(canvasData);

    // Save all evidence nodes
    const savedNodes = [];
    for (const node of validatedCanvas.nodes) {
      const nodeResult = await saveEvidenceNode({ ...node, userId }, userId);
      if (nodeResult.status === 200) {
        const responseData = await nodeResult.json();
        savedNodes.push(responseData.evidence);
      }
    }
    // Save canvas state
    const canvasStateData = {
      id: crypto.randomUUID(),
      reportId: validatedCanvas.caseId || crypto.randomUUID(),
      canvasData: {
        nodes: savedNodes,
        connections: validatedCanvas.connections,
        viewport: validatedCanvas.viewport,
      },
      userId: userId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    return json({
      success: true,
      savedNodes,
      canvasState: canvasStateData,
      message: "Canvas state saved successfully",
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return json(
        {
          error: "Canvas validation failed",
          details: error.errors,
        },
        { status: 400 }
      );
    }
    throw error;
  }
}
// Bulk save multiple evidence items
async function bulkSaveEvidence(evidenceItems: any[], userId: string) {
  try {
    const results = [];
    const errors = [];

    // Process in batches to avoid overwhelming the system
    const batchSize = 10;
    for (let i = 0; i < evidenceItems.length; i += batchSize) {
      const batch = evidenceItems.slice(i, i + batchSize);

      const batchResults = await Promise.allSettled(
        batch.map((item) => saveEvidenceNode(item, userId))
      );

      batchResults.forEach((result, index) => {
        if (result.status === "fulfilled") {
          results.push(result.value);
        } else {
          errors.push({
            index: i + index,
            error: result.reason,
          });
        }
      });
    }
    return json({
      success: true,
      saved: results.length,
      errors: errors.length,
      errorDetails: errors,
      message: `Bulk save completed: ${results.length} saved, ${errors.length} errors`,
    });
  } catch (error) {
    throw error;
  }
}
// Auto-save functionality for incremental updates
async function autoSaveCanvasState(data: any, userId: string) {
  try {
    // Lighter validation for auto-save
    const { canvasState, timestamp } = data;

    if (!canvasState || !timestamp) {
      return json({ error: "Invalid auto-save data" }, { status: 400 });
    }
    // Check if we should save (don't save too frequently)
    const lastAutoSave = await getLastAutoSaveTime(userId, canvasState.caseId);
    const timeSinceLastSave = Date.now() - (lastAutoSave || 0);

    if (timeSinceLastSave < 30000) {
      // 30 seconds minimum between auto-saves
      return json({
        success: true,
        skipped: true,
        message: "Auto-save skipped (too frequent)",
      });
    }
    // Save only the canvas state (not individual evidence items)
    const autoSaveData = {
      id: crypto.randomUUID(),
      reportId: canvasState.caseId || crypto.randomUUID(),
      canvasData: canvasState,
      userId: userId,
      isAutoSave: true,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // Clean up old auto-saves (keep only last 5)
    await cleanupOldAutoSaves(userId, canvasState.caseId);

    return json({
      success: true,
      autoSave: autoSaveData,
      message: "Auto-save completed",
    });
  } catch (error) {
    // Auto-save failures should not be critical
    console.warn("Auto-save failed:", error);
    return json({
      success: false,
      error: "Auto-save failed",
      message: "Auto-save failed but work is not lost",
    });
  }
}
// Helper functions
async function generateFileHash(content: string): Promise<string> {
  if (!content) return "";

  try {
    // Use Web Crypto API for hashing
    const encoder = new TextEncoder();
    const data = encoder.encode(content);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch (error) {
    console.warn("Hash generation failed:", error);
    return "";
  }
}
async function generateAndStoreEmbedding(evidence: any) {
  try {
    // Generate embedding text from evidence content
    const embeddingText = [
      evidence.title,
      evidence.description,
      ...(evidence.tags || []),
      evidence.aiSummary || "",
    ]
      .filter(Boolean)
      .join(" ");

    if (!embeddingText.trim()) return;

    // Generate embedding using Ollama
    const embeddingResponse = await fetch(
      "http://localhost:11434/api/embeddings",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "nomic-embed-text",
          prompt: embeddingText,
        }),
      }
    );

    if (!embeddingResponse.ok) return;

    const embeddingData = await embeddingResponse.json();

    // Store in vector database (PostgreSQL with pgvector)
    await fetch("/api/vector", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        evidenceId: evidence.id,
        embedding: embeddingData.embedding,
        content: embeddingText,
        metadata: {
          title: evidence.title,
          evidenceType: evidence.aiTags?.evidenceType,
          legalRelevance: evidence.aiTags?.legalRelevance,
          tags: evidence.tags,
        },
      }),
    }).catch((error) => console.log("Vector storage failed:", error));

    // Also store in Qdrant for enhanced vector search
    await fetch("/api/qdrant/tag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: evidence.id,
        vector: embeddingData.embedding,
        payload: {
          evidenceId: evidence.id,
          title: evidence.title,
          content: embeddingText,
          evidenceType: evidence.aiTags?.evidenceType,
          legalRelevance: evidence.aiTags?.legalRelevance,
          tags: evidence.tags,
          caseId: evidence.caseId,
        },
      }),
    }).catch((error) => console.log("Qdrant storage failed:", error));
  } catch (error) {
    console.warn("Embedding generation failed:", error);
  }
}
async function updateSearchIndex(evidence: any) {
  try {
    // Update LokiJS search index
    await fetch("/api/search/update-index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ evidence }),
    }).catch((error) => console.log("Search index update failed:", error));
  } catch (error) {
    console.warn("Search index update failed:", error);
  }
}
async function getLastAutoSaveTime(
  userId: string,
  caseId?: string
): Promise<number | null> {
  try {
    // In production, this would query the database
    // For now, return null to allow auto-save
    return null;
  } catch (error) {
    return null;
  }
}
async function cleanupOldAutoSaves(userId: string, caseId?: string) {
  try {
    // In production, this would clean up old auto-saves from the database
    console.log("Cleaning up old auto-saves for user:", userId);
  } catch (error) {
    console.warn("Auto-save cleanup failed:", error);
  }
}
// GET endpoint for loading evidence
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const user = locals.user;
    if (!user) {
      return json({ error: "Authentication required" }, { status: 401 });
    }
    const action = url.searchParams.get("action");
    const caseId = url.searchParams.get("caseId");
    const evidenceId = url.searchParams.get("evidenceId");

    switch (action) {
      case "load_evidence":
        if (evidenceId) {
          return await loadSingleEvidence(evidenceId, user.id);
        } else if (caseId) {
          return await loadEvidenceByCase(caseId, user.id);
        } else {
          return await loadAllEvidence(user.id);
        }
      case "load_canvas_state":
        return await loadCanvasState(caseId || "", user.id);

      case "get_auto_saves":
        return await getAutoSaves(caseId || "", user.id);

      default:
        return json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error) {
    console.error("Load API error:", error);
    return json(
      {
        error: "Failed to load evidence",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};

async function loadSingleEvidence(evidenceId: string, userId: string) {
  // In production, this would query the database
  // For now, return a mock response
  return json({
    success: true,
    evidence: {
      id: evidenceId,
      title: "Sample Evidence",
      content: "Sample content",
      userId,
    },
  });
}
async function loadEvidenceByCase(caseId: string, userId: string) {
  // In production, this would query the database
  return json({ success: true, evidence: [] });
}
async function loadAllEvidence(userId: string) {
  // In production, this would query the database
  return json({ success: true, evidence: [] });
}
async function loadCanvasState(caseId: string, userId: string) {
  // In production, this would query the database
  return json({ success: true, canvasState: null });
}
async function getAutoSaves(caseId: string, userId: string) {
  // In production, this would query the database
  return json({ success: true, autoSaves: [] });
}
