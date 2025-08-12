import { json } from "@sveltejs/kit";
import type { RequestHandler } from "@sveltejs/kit";

const GO_MICROSERVICE_URL = "http://localhost:8080";

export const GET: RequestHandler = async () => {
  try {
    const response = await fetch(`${GO_MICROSERVICE_URL}/api/health`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const health = await response.json();
    return json(health);
  } catch (error) {
    console.error("Health check failed:", error);
    return json(
      {
        status: "unhealthy",
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};
