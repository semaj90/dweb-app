import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";
import net from "node:net";

type HttpCheck = {
  url: string;
  ok: boolean;
  status?: number;
  error?: string;
};

function tcpCheck(
  host: string,
  port: number,
  timeoutMs = 1000
): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let done = false;
    const finish = (result: boolean) => {
      if (!done) {
        done = true;
        try {
          socket.destroy();
        } catch {
          /* ignore */
        }
        resolve(result);
      }
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    try {
      socket.connect(port, host);
    } catch {
      finish(false);
    }
  });
}

async function httpCheck(url: string): Promise<HttpCheck> {
  try {
    const r = await fetch(url, { method: "GET" });
    return { url, ok: r.ok, status: r.status };
  } catch (e: any) {
    return {
      url,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

export const GET: RequestHandler = async () => {
  const [
    clusterHealth,
    summarizerHealth,
    qdrantHealth,
    ollamaVersion,
    pgOpen,
    redisOpen,
  ] = await Promise.all([
    httpCheck("http://localhost:8090/health"),
    httpCheck("http://localhost:8091/health"),
    httpCheck("http://localhost:6333/collections"),
    httpCheck("http://localhost:11434/api/version"),
    tcpCheck("127.0.0.1", 5432),
    tcpCheck("127.0.0.1", 6379),
  ]);

  const status = {
    cluster: clusterHealth,
    summarizer: summarizerHealth,
    qdrant: qdrantHealth,
    ollama: ollamaVersion,
    postgres: { host: "127.0.0.1", port: 5432, open: pgOpen },
    redis: { host: "127.0.0.1", port: 6379, open: redisOpen },
    caching: {
      l1: { type: "memory", ok: true },
      l2: { type: "redis", host: "127.0.0.1", port: 6379, open: redisOpen },
      l3: { type: "postgres", host: "127.0.0.1", port: 5432, open: pgOpen },
      l4: {
        type: "qdrant",
        url: "http://localhost:6333/collections",
        ok: qdrantHealth.ok || false,
      },
    },
    ok:
      (clusterHealth.ok || false) &&
      (summarizerHealth.ok || false) &&
      (qdrantHealth.ok || false) &&
      (ollamaVersion.ok || false) &&
      pgOpen &&
      redisOpen,
    timestamp: new Date().toISOString(),
  } as const;

  const ok =
    (clusterHealth.ok || false) &&
    (summarizerHealth.ok || false) &&
    (qdrantHealth.ok || false) &&
    (ollamaVersion.ok || false) &&
    pgOpen &&
    redisOpen;

  return json(status, { status: ok ? 200 : 503 });
};
