
import { json } from "@sveltejs/kit";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { text } = await request.json();
    if (!text || text.trim() === "") {
      return json({ error: "Text is required" }, { status: 400 });
    }
    // Call Ollama embedding API (nomic-embed-text)
    const ollamaRes = await fetch("http://localhost:11434/api/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "nomic-embed-text", input: text }),
    });
    if (!ollamaRes.ok) {
      const errorText = await ollamaRes.text();
      return json(
        { error: "Ollama error", details: errorText },
        { status: 500 }
      );
    }
    const data = await ollamaRes.json();
    let embedding: number[] = [];
    if (Array.isArray(data?.embedding)) embedding = data.embedding;
    else if (Array.isArray(data?.embeddings)) {
      embedding = Array.isArray(data.embeddings[0])
        ? data.embeddings[0]
        : data.embeddings;
    } else if (
      Array.isArray(data?.data) &&
      Array.isArray(data.data[0]?.embedding)
    ) {
      embedding = data.data[0].embedding;
    }
    return json({ embedding });
  } catch (error: any) {
    return json({ error: "Failed to get embedding" }, { status: 500 });
  }
};
