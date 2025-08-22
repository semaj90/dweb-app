export async function queryQdrant(vec: number[], limit = 200, temperature = 0.3) {
  // Map temperature to score_threshold (Qdrant returns higher = closer for Cosine/Score)
  // We want tight threshold at low temp and lower threshold (more inclusive) at high temp.
  const minScore = 0.65; // strict
  const maxScore = 0.30; // permissive (lower means accept weaker matches)
  const t = Math.max(0, Math.min(1, temperature));
  const score_threshold = minScore - (minScore - maxScore) * t;

  const body = {
    vector: vec,
    limit,
    with_payload: true,
    with_vector: true,
    // some Qdrant builds support `score_threshold` in search body; otherwise filter client-side
    score_threshold
  };

  const r = await fetch(`${process.env.QDRANT}/collections/chunks/points/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!r.ok) {
    const text = await r.text();
    throw new Error(`Qdrant search failed: ${r.status} ${text}`);
  }
  const j = await r.json();

  // Normalize result object
  const results = (j.result || []).map((p: unknown) => ({
    id: String(p.id),
    text: p.payload?.text ?? '',
    metadata: p.payload?.metadata ?? {},
    embedding: p.vector ?? null,
    score: p.score ?? null
  }));

  return { ann: results };
}

// Legacy export for backward compatibility
export { queryQdrant as default };
