/**
 * Server-side health check endpoints for system monitoring
 */
import { json } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { db, sql } from "$lib/server/db/drizzle.js";

/**
 * @type {import('./$types').RequestHandler}
 */
export async function GET() {
  try {
    const ollamaUrl = env.OLLAMA_URL || "http://ollama:11434";

    // Check Ollama service
    let ollamaStatus;
    try {
      const response = await fetch(`${ollamaUrl}/api/version`, {
        method: "GET",
        headers: { Accept: "application/json" },
        timeout: 3000,
      });

      if (response.ok) {
        const data = await response.json();
        ollamaStatus = {
          status: "connected",
          version: data.version || "unknown",
        };
      } else {
        ollamaStatus = {
          status: "error",
          message: `HTTP ${response.status}`,
        };
      }
    } catch (error) {
      console.error("Ollama check error:", error);
      ollamaStatus = {
        status: "disconnected",
        error: error.message,
      };
    }

    // Check PostgreSQL/Drizzle connection
    let dbStatus;
    try {
      // Simple query to verify connection works
      await db.execute(sql`SELECT 1 as connected`);
      dbStatus = { status: "connected" };
    } catch (error) {
      console.error("Database check error:", error);
      dbStatus = {
        status: "error",
        message: error.message,
      };
    }

    // Return combined status
    return json({
      timestamp: new Date().toISOString(),
      services: {
        ollama: ollamaStatus,
        database: dbStatus,
      },
      environment: {
        ollamaUrl: ollamaUrl.replace(/:[^:]+@/, ":***@"), // Hide password if present
      },
    });
  } catch (error) {
    console.error("System check error:", error);
    return json({ error: "System check failed" }, { status: 500 });
  }
}
