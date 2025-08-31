
import { json } from "@sveltejs/kit";
import { db, legalDocuments } from "$lib/server/db";
import { eq } from "drizzle-orm";
import { getVectorStore } from "$lib/ai/langchain-rag";
import type { RequestHandler } from './$types';


export async function POST({ request }): Promise<any> {
  try {
    const body = await request.json();
    const query = body.query;
    const k = body.k || 5;
    if (!query) return json({ error: 'query required' }, { status: 400 });

    // Retrieve from vector store
    const store = getVectorStore();
    const results = await store.similaritySearch(query, k);

    // Hydrate documents from DB for richer metadata
    const hydrated = [];
    for (const r of results) {
      const id = r.metadata?.id;
      if (id) {
        const docs = await db
          .select()
          .from(legalDocuments)
          .where(eq(legalDocuments.id, id))
          .limit(1);
        if (docs[0]) hydrated.push({ chunk: r.pageContent, score: r.score, doc: docs[0] });
      } else {
        hydrated.push({ chunk: r.pageContent, score: r.score });
      }
    }

    return json({ results: hydrated, count: hydrated.length });
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

