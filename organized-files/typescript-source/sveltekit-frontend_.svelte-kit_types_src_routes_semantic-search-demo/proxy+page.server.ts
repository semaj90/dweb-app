// @ts-nocheck
// @ts-nocheck
import type { PageServerLoad } from "./$types";
import { semanticSearch } from "$lib/ai/mcp-helpers";

export const load = async ({ url }: Parameters<PageServerLoad>[0]) => {
  const initialQuery = url.searchParams.get("q") || "";
  let initialResults = [];
  if (initialQuery) {
    try {
      initialResults = await semanticSearch(initialQuery);
    } catch (err) {
      initialResults = [{ text: "Error fetching initial results." }];
    }
  }
  return {
    initialQuery,
    initialResults,
  };
};
