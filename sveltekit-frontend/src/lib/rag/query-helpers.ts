import type { ChatMessage, RAGContext } from "$lib/types/ai-chat";
import { normalize, SOMGrid, type IntentPoint } from "./som-intent";

export interface RankedChunk {
  id: string;
  text: string;
  score: number;
  documentId?: string;
  sectionId?: string;
}

export interface RAGInputs {
  context: RAGContext | undefined | null;
  history: ChatMessage[];
  embeddings: (text: string) => Promise<number[]>; // embedding fn (e.g., Ollama endpoint)
  search: (queryVec: number[], limit: number) => Promise<RankedChunk[]>; // pgvector/Qdrant
}

export async function buildIntentAwareRetrieval(input: RAGInputs) {
  const { context, history, embeddings, search } = input;
  const recent = history.slice(-20); // light window
  const lastUser = [...recent].reverse().find((m) => m.role === "user");
  const queryText =
    lastUser?.content ?? recent.map((m) => m.content).join("\n");
  const qVec = normalize(await embeddings(queryText));

  // SOM grid for short-lived clustering of last N messages
  const points: IntentPoint[] = await Promise.all(
    recent.map(async (m) => ({
      id: m.id,
      vector: normalize(await embeddings(m.content)),
      ts: m.timestamp,
    }))
  );
  const som = new SOMGrid(qVec.length, { width: 6, height: 4 });
  som.trainBatch(
    points.map((p) => p.vector),
    3
  );

  const proj = som.project(qVec);
  const affinity = (p: IntentPoint) => {
    const node = som.project(p.vector);
    const dx = node.x - proj.x;
    const dy = node.y - proj.y;
    const dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
    const rec = Math.exp(-(Date.now() - p.ts) / (1000 * 60 * 60)); // 1h decay
    return rec / dist;
  };
  const topIntent = points
    .map((p) => ({ p, a: affinity(p) }))
    .sort((a, b) => b.a - a.a)
    .slice(0, 3);

  // Build enriched query (intent + context metadata)
  const enrichedQuery = [
    queryText,
    context?.caseId ? `Case:${context.caseId}` : "",
    (context?.documents ?? [])
      .slice(0, 10)
      .map((d) => `Doc:${d}`)
      .join(" "),
    "Intent anchor messages:\n" +
      topIntent.map(({ p }) => `- ${p.id}`).join("\n"),
  ]
    .filter(Boolean)
    .join("\n");

  const enrichedVec = normalize(await embeddings(enrichedQuery));
  const results = await search(enrichedVec, 10);
  return { enrichedQuery, results };
}
