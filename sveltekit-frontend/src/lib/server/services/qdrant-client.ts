// Use process.env instead of SvelteKit env for server-side code
// import { env } from "$env/dynamic/private";
import { QdrantClient } from "@qdrant/js-client-rest";

export const qdrant = new QdrantClient({ url: process.env.QDRANT_URL || "http://localhost:6333" });

export const EVIDENCE_COLLECTION_NAME = "evidence_v1";

/**
 * Ensures the Qdrant collection exists and has a payload index for tags.
 * This is critical for efficient filtering and should be called on server startup.
 */
export async function initializeQdrantCollection() {
  try {
    const collections = await qdrant.getCollections();
    const collectionExists = collections.collections.some(
      (c) => c.name === EVIDENCE_COLLECTION_NAME,
    );

    if (!collectionExists) {
      console.log(`Creating Qdrant collection: ${EVIDENCE_COLLECTION_NAME}`);
      await qdrant.createCollection(EVIDENCE_COLLECTION_NAME, {
        vectors: { size: 768, distance: "Cosine" },
      });
      await qdrant.createPayloadIndex(EVIDENCE_COLLECTION_NAME, {
        field_name: "tags",
        field_schema: "keyword",
        wait: true,
      });
      console.log("Qdrant collection and payload index created successfully.");
    }
  } catch (error) {
    console.error("Failed to initialize Qdrant:", error);
  }
}
