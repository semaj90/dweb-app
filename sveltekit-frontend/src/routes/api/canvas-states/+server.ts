import { canvasStates } from "$lib/server/db/schema-postgres";
import type { RequestEvent } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { and, desc, eq, like, sql } from "drizzle-orm";
import { db } from "$lib/server/db/index";

export async function GET({ url, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const caseId = url.searchParams.get("caseId");
    const canvasId = url.searchParams.get("id");
    const search = url.searchParams.get("search") || "";
    const isTemplate = url.searchParams.get("isTemplate");
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const sortBy = url.searchParams.get("sortBy") || "updatedAt";
    const sortOrder = url.searchParams.get("sortOrder") || "desc";

    if (canvasId) {
      // Get specific canvas state
      const [canvasState] = await db
        .select()
        .from(canvasStates)
        .where(eq(canvasStates.id, canvasId))
        .limit(1);

      if (!canvasState) {
        return json({ error: "Canvas state not found" }, { status: 404 });
      }
      return json(canvasState);
    } else {
      // Build base query
      let queryBuilder = db.select().from(canvasStates);
      const filters: any[] = [];

      // Add case filter
      if (caseId) {
        filters.push(eq(canvasStates.caseId, caseId));
      }
      // Add search filter
      if (search) {
        filters.push(like(canvasStates.name, `%${search}%`));
      }
      // Add template filter
      if (isTemplate !== null) {
        filters.push(eq(canvasStates.isDefault, isTemplate === "true"));
      }

      // Apply filters to the query builder
      if (filters.length > 0) {
        queryBuilder = queryBuilder.where(and(...filters));
      }

      // Determine the column for sorting
      const orderColumn =
        sortBy === "name"
          ? canvasStates.name
          : sortBy === "version"
            ? canvasStates.version
            : canvasStates.createdAt; // Default to createdAt

      // Apply sorting to the query builder
      queryBuilder = queryBuilder.orderBy(
        sortOrder === "asc" ? orderColumn : desc(orderColumn),
      );

      // Apply pagination to the query builder
      queryBuilder = queryBuilder.limit(limit).offset(offset);

      const canvasStateList = await queryBuilder;

      // Get total count for pagination (using the same filters)
      let countQuery = db
        .select({ count: sql<number>`count(*)` })
        .from(canvasStates);
      if (filters.length > 0) {
        countQuery = countQuery.where(and(...filters));
      }
      const totalCountResult = await countQuery;
      const totalCount = totalCountResult[0]?.count || 0;

      return json({
        canvasStates: canvasStateList,
        totalCount,
        hasMore: offset + limit < totalCount,
        pagination: {
          limit,
          offset,
          total: totalCount,
        },
      });
    }
  } catch (error) {
    console.error("Error fetching canvas states:", error);
    return json({ error: "Failed to fetch canvas states" }, { status: 500 });
  }
}
export async function POST({ request, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    // Validate required fields
    if (!data.name || !data.canvasData) {
      return json(
        { error: "Name and canvas data are required" },
        { status: 400 },
      );
    }
    const canvasStateData = {
      caseId: data.caseId || null,
      name: data.name.trim(),
      canvasData: data.canvasData,
      version: data.version || 1,
      isDefault: data.isDefault || false,
      createdBy: locals.user.id,
    };

    const [newCanvasState] = await db
      .insert(canvasStates)
      .values(canvasStateData)
      .returning();

    return json(newCanvasState, { status: 201 });
  } catch (error) {
    console.error("Error creating canvas state:", error);
    return json({ error: "Failed to create canvas state" }, { status: 500 });
  }
}
export async function PUT({ request, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    if (!data.id) {
      return json({ error: "Canvas state ID is required" }, { status: 400 });
    }
    // Check if canvas state exists
    const existingCanvasState = await db
      .select()
      .from(canvasStates)
      .where(eq(canvasStates.id, data.id))
      .limit(1);

    if (!existingCanvasState.length) {
      return json({ error: "Canvas state not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Only update provided fields
    if (data.name !== undefined) updateData.name = data.name.trim();
    if (data.canvasData !== undefined) updateData.canvasData = data.canvasData;
    if (data.version !== undefined) updateData.version = data.version;
    if (data.isDefault !== undefined) updateData.isDefault = data.isDefault;

    const [updatedCanvasState] = await db
      .update(canvasStates)
      .set(updateData)
      .where(eq(canvasStates.id, data.id))
      .returning();

    return json(updatedCanvasState);
  } catch (error) {
    console.error("Error updating canvas state:", error);
    return json({ error: "Failed to update canvas state" }, { status: 500 });
  }
}
export async function DELETE({ url, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const canvasId = url.searchParams.get("id");
    if (!canvasId) {
      return json({ error: "Canvas state ID is required" }, { status: 400 });
    }
    // Check if canvas state exists
    const existingCanvasState = await db
      .select()
      .from(canvasStates)
      .where(eq(canvasStates.id, canvasId))
      .limit(1);

    if (!existingCanvasState.length) {
      return json({ error: "Canvas state not found" }, { status: 404 });
    }
    // Delete the canvas state
    const [deletedCanvasState] = await db
      .delete(canvasStates)
      .where(eq(canvasStates.id, canvasId))
      .returning();

    return json({ success: true, deletedCanvasState });
  } catch (error) {
    console.error("Error deleting canvas state:", error);
    return json({ error: "Failed to delete canvas state" }, { status: 500 });
  }
}
// PATCH endpoint for partial updates
export async function PATCH({ request, url, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const canvasId = url.searchParams.get("id");
    if (!canvasId) {
      return json({ error: "Canvas state ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if canvas state exists
    const existingCanvasState = await db
      .select()
      .from(canvasStates)
      .where(eq(canvasStates.id, canvasId))
      .limit(1);

    if (!existingCanvasState.length) {
      return json({ error: "Canvas state not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Handle specific patch operations
    if (data.operation === "incrementVersion") {
      updateData.version = (existingCanvasState[0].version || 1) + 1;
    } else if (data.operation === "setAsDefault") {
      // First, unset all other default canvases for this case
      if (existingCanvasState[0].caseId) {
        await db
          .update(canvasStates)
          .set({ isDefault: false })
          .where(eq(canvasStates.caseId, existingCanvasState[0].caseId));
      }
      updateData.isDefault = true;
    } else if (data.operation === "updateData") {
      updateData.canvasData = data.canvasData;
      updateData.version = (existingCanvasState[0].version || 1) + 1;
    } else {
      // Regular field updates
      Object.keys(data).forEach((key) => {
        if (key !== "operation") {
          updateData[key] = data[key];
        }
      });
    }
    const [updatedCanvasState] = await db
      .update(canvasStates)
      .set(updateData)
      .where(eq(canvasStates.id, canvasId))
      .returning();

    return json(updatedCanvasState);
  } catch (error) {
    console.error("Error patching canvas state:", error);
    return json({ error: "Failed to update canvas state" }, { status: 500 });
  }
}
