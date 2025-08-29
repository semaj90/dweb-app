import { URL } from "url";

import type { PageServerLoad } from './$types';
// Mock semantic search function
const semanticSearch = async (query: string): Promise<any> => {
  return [{ text: `Mock result for: ${query}` }];
};

export const load: PageServerLoad = async ({ url }) => {
  const initialQuery = url.searchParams.get("q") || "";
  let initialResults = [];
  if (initialQuery) {
    try {
      initialResults = await semanticSearch(initialQuery);
    } catch (err: any) {
      initialResults = [{ text: "Error fetching initial results." }];
    }
  }
  return {
    initialQuery,
    initialResults,
  };
};
