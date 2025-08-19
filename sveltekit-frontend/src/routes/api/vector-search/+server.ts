import { json } from "@sveltejs/kit";
// Orphaned content: import type { RequestHandler

const GO_MICROSERVICE_URL = "http://localhost:8080";

export const POST: RequestHandler = async ({ request }) => {
  let searchRequest: { query?: string } | null = null;
  try {
    searchRequest = await request.json();

    const response = await fetch(`${GO_MICROSERVICE_URL}/api/vector-search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(searchRequest),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return json(result);
  } catch (error: unknown) {
    console.error("Vector search failed:", error);
    return json(
      {
        results: [],
        total: 0,
        query: searchRequest?.query || "",
        took: "0ms",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};
