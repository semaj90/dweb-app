// Qdrant Vector Database Management API
// Handles syncing between PostgreSQL and Qdrant vector database
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "@sveltejs/kit";
import { qdrant } from "$lib/server/vector/qdrant";

// Sync request interface
interface QdrantSyncRequest {
  collection?: string;
  batchSize?: number;
  limit?: number;
  forceRecreate?: boolean;
}
// Collection management request
interface CollectionRequest {
  name: string;
  vectorSize?: number;
  distance?: "Cosine" | "Euclid" | "Dot";
}
// Sync data from PostgreSQL to Qdrant
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Check admin permissions
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required",
        },
        { status: 403 },
      );
    }
    const body: QdrantSyncRequest = await request.json();
    const {
      collection = "default",
      batchSize = 100,
      limit = 1000,
      forceRecreate = false,
    } = body;

    const startTime = Date.now();

    // Sync embeddings to Qdrant
    const syncResult = await qdrant.syncFromPostgreSQL({
      collection,
      batchSize,
      limit,
      forceRecreate,
    });

    return json({
      success: true,
      data: {
        ...syncResult,
        executionTime: Date.now() - startTime,
        collection,
      },
    });
  } catch (error) {
    console.error("Qdrant sync error:", error);
    return json(
      {
        success: false,
        error: "Failed to sync with Qdrant",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Get Qdrant status and collection info
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const collection = url.searchParams.get("collection") || "default";

    // Get Qdrant health status
    const isHealthy = await qdrant.healthCheck();

    if (!isHealthy) {
      return json(
        {
          success: false,
          error: "Qdrant service is not available",
        },
        { status: 503 },
      );
    }
    // Get collection information
    const collections = await qdrant.listCollections();
    const collectionInfo = await qdrant.getCollectionInfo(collection);

    return json({
      success: true,
      data: {
        status: "healthy",
        collections,
        currentCollection: {
          name: collection,
          ...collectionInfo,
        },
        endpoints: {
          "POST /api/qdrant": "Sync PostgreSQL data to Qdrant",
          "PUT /api/qdrant": "Create or recreate collection",
          "DELETE /api/qdrant": "Delete collection",
          "GET /api/qdrant": "Get status and collection info",
        },
      },
    });
  } catch (error) {
    console.error("Qdrant status error:", error);
    return json(
      {
        success: false,
        error: "Failed to get Qdrant status",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Create or recreate Qdrant collection
export const PUT: RequestHandler = async ({ request, locals }) => {
  try {
    // Check admin permissions
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required",
        },
        { status: 403 },
      );
    }
    const body: CollectionRequest = await request.json();
    const {
      name,
      vectorSize = 384, // Default for sentence-transformers/all-MiniLM-L6-v2
      distance = "Cosine",
    } = body;

    if (!name || name.trim().length === 0) {
      return json(
        {
          success: false,
          error: "Collection name is required",
        },
        { status: 400 },
      );
    }
    // Create collection
    const result = await qdrant.createCollection(name, {
      vectorSize,
      distance,
    });

    return json({
      success: true,
      data: {
        message: `Collection '${name}' created successfully`,
        collection: name,
        config: {
          vectorSize,
          distance,
        },
        ...result,
      },
    });
  } catch (error) {
    console.error("Qdrant collection creation error:", error);
    return json(
      {
        success: false,
        error: "Failed to create collection",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Delete Qdrant collection
export const DELETE: RequestHandler = async ({ request, locals }) => {
  try {
    // Check admin permissions
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required",
        },
        { status: 403 },
      );
    }
    const { collection } = await request.json();

    if (!collection || collection.trim().length === 0) {
      return json(
        {
          success: false,
          error: "Collection name is required",
        },
        { status: 400 },
      );
    }
    // Prevent deletion of default collection without explicit confirmation
    if (collection === "default") {
      return json(
        {
          success: false,
          error:
            "Cannot delete default collection. Use forceDelete=true to override.",
        },
        { status: 400 },
      );
    }
    const result = await qdrant.deleteCollection(collection);

    return json({
      success: true,
      data: {
        message: `Collection '${collection}' deleted successfully`,
        collection,
        ...result,
      },
    });
  } catch (error) {
    console.error("Qdrant collection deletion error:", error);
    return json(
      {
        success: false,
        error: "Failed to delete collection",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};
