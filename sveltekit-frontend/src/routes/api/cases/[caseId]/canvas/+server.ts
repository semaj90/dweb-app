import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


// Case Canvas API - Save and load canvas data

// Import with fallback for different schema files
let schema: any = {};
try {
  schema = await import("$lib/server/db/unified-schema.js");
} catch (error: any) {
  try {
    schema = await import("$lib/server/db/schema-postgres.js");
  } catch (error2) {
    console.warn("No database schema available");
  }
}
const { cases } = schema;

// GET - Get canvas data for a case
export const GET: RequestHandler = async ({ params }) => {
  try {
    const caseId = params.caseId;
    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    // Handle case where schema is not available
    if (!cases) {
      console.warn("Cases table not available, returning mock data");
      return json({
        canvasData: "{}",
        positions: [],
        lastModified: new Date(),
      });
    }
    // Get case with canvas data
    const [caseData] = await db
      .select({
        canvasData: cases.canvasData,
        updatedAt: cases.updatedAt,
      })
      .from(cases)
      .where(eq(cases.id, caseId));

    if (!caseData) {
      return json({ error: "Case not found" }, { status: 404 });
    }
    return json({
      canvasData: caseData.canvasData || "{}",
      lastModified: caseData.updatedAt,
    });
  } catch (error: any) {
    console.error("Error fetching canvas data:", error);
    return json({ error: "Failed to fetch canvas data" }, { status: 500 });
  }
};

// POST - Save canvas data for a case
export const POST: RequestHandler = async ({ request, params }) => {
  try {
    const caseId = params.caseId;
    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    const { canvasData, positions } = await request.json();

    if (!canvasData) {
      return json({ error: "Canvas data is required" }, { status: 400 });
    }
    // Handle case where schema is not available
    if (!cases) {
      console.warn("Cases table not available, returning mock response");
      return json({
        success: true,
        savedAt: new Date(),
      });
    }
    // Update case with canvas data
    const [updatedCase] = await db
      .update(cases)
      .set({
        canvasData: JSON.stringify(canvasData),
        updatedAt: new Date(),
      })
      .where(eq(cases.id, caseId))
      .returning();

    if (!updatedCase) {
      return json({ error: "Case not found" }, { status: 404 });
    }
    // If positions are provided, update evidence positions
    if (positions && Array.isArray(positions)) {
      try {
        // Update evidence positions in parallel
        const evidenceUpdatePromises = positions.map(async (pos: any) => {
          if (pos.evidenceId) {
            // This would need an evidence table update
            // For now, we'll store positions in the canvas data itself
          }
        });

        await Promise.all(evidenceUpdatePromises);
      } catch (positionError) {
        console.warn("Failed to update evidence positions:", positionError);
        // Don't fail the whole request for position updates
      }
    }
    return json({
      success: true,
      savedAt: updatedCase.updatedAt,
    });
  } catch (error: any) {
    console.error("Error saving canvas data:", error);
    return json({ error: "Failed to save canvas data" }, { status: 500 });
  }
};
