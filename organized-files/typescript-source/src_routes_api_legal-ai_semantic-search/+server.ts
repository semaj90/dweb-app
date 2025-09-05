// @ts-nocheck
import type { RequestHandler } from "@sveltejs/kit";

export const POST: RequestHandler = async ({ request }) => {
  const { query } = await request.json();
  // TODO: Integrate with backend semantic search
  // Return mock results for now
  return new Response(JSON.stringify({ results: [] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};
