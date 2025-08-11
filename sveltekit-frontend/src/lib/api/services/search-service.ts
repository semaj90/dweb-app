import { apiFetch } from "../clients/api-client";

export interface SemanticQuery {
  query: string;
  limit?: number;
}
export async function semanticSearch(q: SemanticQuery) {
  return apiFetch("http://localhost:8080/api/v1/search/semantic", "POST", {
    body: q,
  });
}
