import { apiFetch } from "../clients/api-client";

export async function getMetricsText(): Promise<string> {
  return apiFetch("http://localhost:8080/metrics");
}
