/// <reference types="vite/client" />
export async function queryQdrant(vec: number[], limit = 200, temperature = 0.3): Promise<any> {
  const minScore = 0.65;
  const maxScore = 0.3;
  const t = Math.max(0, Math.min(1, temperature));
  const score_threshold = minScore - (minScore - maxScore) * t;

  const body = { vector: vec, limit, with_payload: true, with_vector: true, score_threshold };
  const r = await fetch(`${import.meta.env.QDRANT}/collections/chunks/points/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Qdrant search failed: ${r.status} ${await r.text()}`);
  const j = await r.json();
  return {
    ann: (j.result || []).map((p: any) => ({
      id: String(p.id),
      text: p.payload?.text ?? '',
      metadata: p.payload?.metadata ?? {},
      embedding: p.vector ?? null,
      score: p.score ?? null,
    })),
  };
}
