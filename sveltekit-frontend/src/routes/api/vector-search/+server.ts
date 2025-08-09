// @ts-nocheck
import { json } from "@sveltejs/kit";
import { db, cases } from "$lib/server/db/index";
import { qdrant } from "$lib/server/vector/qdrant";

export async function POST({ request }) {
  const { query, embedding } = await request.json();

  // 1. Search with pgvector (fast, for most cases)
  const pgvectorResults = await db.query.cases.findMany({
    orderBy: (cases, { sql }) => sql`embedding <-> ${embedding}`,
    limit: 10,
  });

  // 2. Search with Qdrant (advanced, for more complex queries)
  const qdrantResults = await qdrant.searchCases(query, {
    limit: 10,
  });

  return json({ pgvectorResults, qdrantResults });
}
