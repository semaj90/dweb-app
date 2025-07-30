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
    console.log(`[System Check] Starting health check with Ollama URL: ${ollamaUrl}`);

    // Check Ollama service
    let ollamaStatus;
    try {
      console.log(`[System Check] Checking Ollama at ${ollamaUrl}/api/version`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch(`${ollamaUrl}/api/version`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

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
      console.log(`[System Check] Checking database connection`);
      // Simple query to verify connection works
      const result = await db.execute(sql`SELECT 1 as connected`);
      console.log(`[System Check] Database query result:`, result);
      dbStatus = { status: "connected" };
    } catch (error) {
      console.error("[System Check] Database check error:", error);
      dbStatus = {
        status: "error",
        error: error.message,
      };
    }

    // Return combined status
    const result = {
      timestamp: new Date().toISOString(),
      services: {
        ollama: ollamaStatus,
        database: dbStatus,
      },
      environment: {
        ollamaUrl: ollamaUrl.replace(/:[^:]+@/, ":***@"), // Hide password if present
      },
    };
    
    console.log(`[System Check] Returning result:`, JSON.stringify(result, null, 2));
    return json(result);
  } catch (error) {
    console.error("[System Check] Top-level error:", error);
    return json({ 
      error: "System check failed", 
      details: error.message,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
}
