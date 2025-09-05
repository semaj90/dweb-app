// Root copy: Cross-Encoder Reranker mock implementation

export interface LegalDocument {
  id: string;
  title: string;
  content: string;
  type?: string;
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  score: number;
  document: LegalDocument;
}

export async function rerankResults(
  results: SearchResult[],
  query: string
): Promise<SearchResult[]> {
  // Simple mock: boost score if query terms appear in title
  const terms = query.toLowerCase().split(/\s+/);
  return results
    .map((r) => {
      const matchBoost = terms.some((t) => r.title.toLowerCase().includes(t))
        ? 0.1
        : 0;
      return { ...r, score: r.score + matchBoost };
    })
    .sort((a, b) => b.score - a.score);
}

// Provide some mock data if needed by tests
export function getMockResults(): SearchResult[] {
  return [
    {
      id: "doc1",
      title: "Contract Overview",
      content: "Contract content",
      score: 0.6,
      document: {
        id: "doc1",
        title: "Contract Overview",
        content: "Contract content",
      },
    },
    {
      id: "doc2",
      title: "Liability Clause",
      content: "Liability details",
      score: 0.8,
      document: {
        id: "doc2",
        title: "Liability Clause",
        content: "Liability details",
      },
    },
    {
      id: "doc3",
      title: "Termination Terms",
      content: "Termination content",
      score: 0.4,
      document: {
        id: "doc3",
        title: "Termination Terms",
        content: "Termination content",
      },
    },
  ];
}
