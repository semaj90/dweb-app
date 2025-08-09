// @ts-nocheck
// Simple analytics logger for API endpoints
// In production, replace with a real analytics/event system
import { writeFile } from "fs/promises";
import path from "path";

const LOG_PATH = path.resolve(process.cwd(), "analytics-log.jsonl");

export async function json(event: any) {
  try {
    const line = JSON.stringify(event) + "\n";
    await writeFile(LOG_PATH, line, { flag: "a" });
  } catch (err) {
    // Fallback: log to console
    console.error("Analytics log error:", err);
  }
}
