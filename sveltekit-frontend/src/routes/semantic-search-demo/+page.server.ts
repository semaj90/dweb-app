import { URL } from "url";

import type { PageServerLoad } from "./$types.js";
// Orphaned content: import {

export const load: PageServerLoad = async ({ url }) => {
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
