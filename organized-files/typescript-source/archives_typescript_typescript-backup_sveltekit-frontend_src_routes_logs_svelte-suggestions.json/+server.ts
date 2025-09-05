import type { RequestHandler } from '@sveltejs/kit';
import { readFile } from 'fs/promises';
import path from "path";

export const GET: RequestHandler = async () => {
  try {
    const appCwd = process.cwd();
    const logsPath = path.resolve(
      appCwd,
      "..",
      "logs",
      "svelte-suggestions.json"
    );
    const raw = await readFile(logsPath, "utf8");
    return new Response(raw, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: any) {
    const detail = err instanceof Error ? err.message : "missing";
    return new Response(JSON.stringify({ error: "not_found", detail }), {
      status: 404,
    });
  }
};
