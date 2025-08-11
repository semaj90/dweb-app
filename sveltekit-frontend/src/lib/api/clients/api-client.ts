// Minimal API client stub to satisfy barrel exports; expand with real logic later.
export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";

export interface RequestOptions {
  headers?: Record<string, string>;
  query?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
}

export async function apiFetch<T = unknown>(
  url: string,
  method: HttpMethod = "GET",
  opts: RequestOptions = {}
): Promise<T> {
  const { headers, query, body } = opts;
  const qs = query
    ? `?${new URLSearchParams(Object.entries(query).filter(([, v]) => v !== undefined) as any)}`
    : "";
  const res = await fetch(`${url}${qs}`, {
    method,
    headers: { "Content-Type": "application/json", ...(headers || {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  } as RequestInit);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const ct = res.headers.get("content-type") || "";
  return (
    ct.includes("application/json")
      ? await res.json()
      : ((await res.text()) as any)
  ) as T;
}

export const ApiClient = { fetch: apiFetch };
