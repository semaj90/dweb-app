import type { RequestHandler } from "@sveltejs/kit";

type ServiceMetrics = {
  total: number;
  success: number;
  errors: number;
  avgLatencyMs: number;
  lastLatencyMs: number;
  uptimeSec: number;
  startTime: number;
};

type ServiceStatus = {
  ok: boolean;
  metrics?: ServiceMetrics;
  error?: string;
};

const CLUSTER_METRICS_URL = "http://localhost:8090/metrics";
const SUMMARIZER_METRICS_URL = "http://localhost:8091/metrics";

async function getJSON<T>(url: string): Promise<T> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), 3000);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(id);
  }
}

export const GET: RequestHandler = async () => {
  const [cluster, summarizer] = await Promise.allSettled([
    getJSON<ServiceMetrics>(CLUSTER_METRICS_URL),
    getJSON<ServiceMetrics>(SUMMARIZER_METRICS_URL),
  ]);

  const clusterStatus: ServiceStatus =
    cluster.status === "fulfilled"
      ? { ok: true, metrics: cluster.value }
      : {
          ok: false,
          error: cluster.reason?.message ?? "Cluster metrics unavailable",
        };

  const summarizerStatus: ServiceStatus =
    summarizer.status === "fulfilled"
      ? { ok: true, metrics: summarizer.value }
      : {
          ok: false,
          error: summarizer.reason?.message ?? "Summarizer metrics unavailable",
        };

  const ok = clusterStatus.ok && summarizerStatus.ok;

  const body = {
    status: ok ? "ok" : "degraded",
    cluster: clusterStatus,
    summarizer: summarizerStatus,
  };

  return new Response(JSON.stringify(body), {
    status: ok ? 200 : 503,
    headers: { "content-type": "application/json" },
  });
};
