import { browser } from "$app/environment";
import { realtimeComm, , type TelemetryPayload = {,   session_id: string;,   user_id?: string;,   is_typing?: boolean;,   visible?: boolean;,   long_tasks?: number;,   hints?: string[]; } from

let typingTimer: ReturnType<typeof setTimeout> | null = null;
const TYPING_IDLE_MS = 800;

export function initTypingDetector(getSession: () => string, getUser?: () => string) {
  if (!browser) return;
  const send = (data: Partial<TelemetryPayload>) => {
    const payload: TelemetryPayload = {
      session_id: getSession(),
      user_id: getUser?.(),
      ...data,
    };
    realtimeComm.sendMessage('user_activity', { telemetry: payload }, 'low').catch(() => {});
  };

  const onInput = () => {
    if (typingTimer) clearTimeout(typingTimer);
    send({ is_typing: true });
    typingTimer = setTimeout(() => send({ is_typing: false }), TYPING_IDLE_MS);
  };

  window.addEventListener('keydown', onInput, { passive: true });
  window.addEventListener('input', onInput as any, { passive: true } as any);

  document.addEventListener('visibilitychange', () =>
    send({ visible: document.visibilityState === 'visible' })
  );

  try {
    // Minimal long task observer
    const po = new PerformanceObserver((list) => {
      send({ long_tasks: list.getEntries().length });
    });
    po.observe({ entryTypes: ['longtask'] as any });
  } catch {}
}

export function bridgePrefetchToSW(urls: string[]) {
  if (!browser) return;
  if (navigator.serviceWorker?.controller) {
    navigator.serviceWorker.controller.postMessage({ type: 'PREFETCH_URLS', data: { urls } });
  }
}

// Optional: wire realtime PrefetchPlan -> Service Worker prefetch (offline-safe)
export function enablePrefetchFromRealtime() {
  if (!browser) return;
  try {
    realtimeComm.onMessage('prefetch_plan' as any, (msg: any) => {
      const urls: string[] = msg?.data?.urls ?? [];
      if (Array.isArray(urls) && urls.length) bridgePrefetchToSW(urls);
    });
  } catch {
    // No-op if realtime layer not initialized yet
  }
}

// Tiny helper to request GET_CACHE_STATUS from Service Worker
export async function getServiceWorkerCacheStatus(): Promise<{ name: string; count: number }[]> {
  if (!browser) return [];
  if (!navigator.serviceWorker?.controller) return [];

  return new Promise((resolve) => {
    const channel = new MessageChannel();
    const timeout = setTimeout(() => {
      try {
        channel.port1.close();
      } catch {}
      resolve([]);
    }, 3000);

    channel.port1.onmessage = (evt) => {
      clearTimeout(timeout);
      try {
        channel.port1.close();
      } catch {}
      const data = evt.data;
      if (data?.type === 'CACHE_STATUS' && Array.isArray(data.data)) {
        resolve(data.data as { name: string; count: number }[]);
      } else {
        resolve([]);
      }
    };

    navigator.serviceWorker.controller.postMessage({ type: 'GET_CACHE_STATUS' }, [channel.port2]);
  });
}

