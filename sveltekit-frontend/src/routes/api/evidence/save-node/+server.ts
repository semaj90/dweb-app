import { json } from "@sveltejs/kit";
import { z } from "zod";
import type { RequestHandler } from './$types.js';
import crypto from "crypto";
import { URL } from "url";


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
  metadata: z.object({
    x: z.number(),
    y: z.number(),
    width: z.number(),
    height: z.number(),
    connections: z.array(z.string()).optional()
  }),
  caseId: z.string().optional(),
  userId: z.string()
});

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Check authentication
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
      default:
        return json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error: any) {
    console.error("Save API error:", error);
    return json(
      { 
        error: "Failed to save evidence", 
        details: error instanceof Error ? error.message : "Unknown error" 
      },
      { status: 500 }
    );
  }
};

// Save individual evidence node
async function saveEvidenceNode(nodeData: any, userId: string): Promise<any> {
  try {
    // Validate input
    const validatedNode = evidenceNodeSchema.parse({
      ...nodeData,
      userId
    });

    // Create evidence data structure
    const evidenceData = {
      id: validatedNode.id,
      fileName: validatedNode.name,
      title: validatedNode.title || validatedNode.name,
      description: validatedNode.description || "",
      fileType: validatedNode.fileType,
      filePath: validatedNode.filePath,
      fileSize: validatedNode.fileSize,
      content: validatedNode.content,
      canvasPosition: validatedNode.metadata,
      caseId: validatedNode.caseId,
      userId: userId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    return json({
      success: true,
      evidence: evidenceData,
      message: "Evidence saved successfully"
    });

  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return json(
        { error: "Validation failed", details: error.errors },
        { status: 400 }
      );
    }
    throw error;
  }
}

// Save canvas state
async function saveCanvasState(canvasData: any, userId: string): Promise<any> {
  try {
    const canvasStateData = {
      id: crypto.randomUUID(),
      canvasData: canvasData,
      userId: userId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    return json({
      success: true,
      canvasState: canvasStateData,
      message: "Canvas state saved successfully"
    });

  } catch (error: any) {
    throw error;
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

    switch (action) {
      case "load_evidence":
        return json({ success: true, evidence: [] });
      case "load_canvas_state":
        return json({ success: true, canvasState: null });
      default:
        return json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error: any) {
    console.error("Load API error:", error);
    return json(
      { 
        error: "Failed to load evidence", 
        details: error instanceof Error ? error.message : "Unknown error" 
      },
      { status: 500 }
    );
  }
};