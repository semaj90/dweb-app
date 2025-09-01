import { json } from '@sveltejs/kit';
import { librarySyncService } from "$lib/services/library-sync-service";
import type { RequestHandler } from './$types.js';
import crypto from "crypto";
import { URL } from "url";


// GET /api/agent-logs - Get recent agent logs
export const GET: RequestHandler = async ({ url }) => {
  try {
    const agentType = url.searchParams.get("agentType") || undefined;
    const limit = parseInt(url.searchParams.get("limit") || "50");
    
    const logs = await librarySyncService.getRecentAgentLogs(
      agentType as any,
      limit
    );
    
    return json({
      success: true,
      logs,
      count: logs.length
    });
  } catch (error: any) {
    console.error("Failed to get agent logs:", error);
    return json(
      { success: false, error: "Failed to get agent logs" },
      { status: 500 }
    );
  }
};

// POST /api/agent-logs - Log a new agent call
export const POST: RequestHandler = async ({ request }) => {
  try {
    const logData = await request.json();
    const agentLog = {
      id: crypto.randomUUID(),
      timestamp: new Date(),
      ...logData
    };
    
    await librarySyncService.logAgentCall(agentLog);
    
    return json({
      success: true,
      message: "Agent call logged successfully",
      logId: agentLog.id
    });
  } catch (error: any) {
    console.error("Failed to log agent call:", error);
    return json(
      { success: false, error: "Failed to log agent call" },
      { status: 500 }
    );
  }
};