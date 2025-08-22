// Chat and user history store using Svelte 5 runes with time/context awareness
// Importable anywhere; values are reactive via Svelte runes
//
// Note: This file uses Svelte 5 runes in a .svelte.ts module.
// The Svelte compiler will transform $state/$derived appropriately.

import type {
  ChatMessage,
  ChatSession,
  ConnectionStatus,
  Recommendation,
  UserActivity,
  RAGContext,
} from "$lib/types/ai-chat";

// Internal shapes

// Map of sessionId -> messages
export type SessionMap = Map<string, ChatMessage[]>;

// Core state
export let sessions = $state<ChatSession[]>([]);
export const sessionMessages = $state<SessionMap>(new Map());
export let currentSessionId = $state<string | null>(null);
export let connectionStatus = $state<ConnectionStatus>("disconnected");
export const isTyping = $state(false);
export let userActivity = $state<UserActivity[]>([]);
export let recommendations = $state<Recommendation[]>([]);

// Deriveds
export const currentSession = $derived(
  sessions.find((s) => s.id === currentSessionId) ?? null
);

export const currentMessages = $derived(
  currentSessionId ? (sessionMessages.get(currentSessionId) ?? []) : []
);

// Session helpers
export function createSession(input: {
  id: string;
  title?: string;
  context?: RAGContext;
  created?: number;
}): ChatSession {
  const now = Date.now();
  const session: ChatSession = {
    id: input.id,
    title: input.title ?? "New Chat",
    created: input.created ?? now,
    updated: now,
    messageCount: 0,
    status: "active",
    context: input.context,
  };
  sessions = [session, ...sessions.filter((s) => s.id !== session.id)];
  if (!sessionMessages.has(session.id)) sessionMessages.set(session.id, []);
  currentSessionId = session.id;
  return session;
}

export function switchSession(id: string) {
  if (sessions.some((s) => s.id === id)) currentSessionId = id;
}

export function addMessage(msg: ChatMessage) {
  const list = sessionMessages.get(msg.sessionId) ?? [];
  list.push(msg);
  sessionMessages.set(msg.sessionId, list);
  const idx = sessions.findIndex((s) => s.id === msg.sessionId);
  if (idx !== -1) {
    const updated = { ...sessions[idx] };
    updated.messageCount = list.length;
    updated.updated = Date.now();
    sessions = [updated, ...sessions.filter((s) => s.id !== updated.id)];
  }
}

// Presence tracking
export function setUserActivity(activity: UserActivity) {
  const i = userActivity.findIndex(
    (a) => a.userId === activity.userId && a.sessionId === activity.sessionId
  );
  if (i === -1) userActivity.push(activity);
  else userActivity[i] = activity;
}

export function clearStaleActivity(staleMs = 60_000) {
  const cutoff = Date.now() - staleMs;
  userActivity = userActivity.filter((a) => a.lastSeen >= cutoff);
}

// Time-aware context window selection (recency + role weighting)
export function getContextWindow(opts: {
  sessionId: string;
  maxTokens?: number; // soft budget
  maxMessages?: number;
  halfLifeMinutes?: number; // recency decay half-life
}) {
  const {
    sessionId,
    maxTokens = 3000,
    maxMessages = 30,
    halfLifeMinutes = 30,
  } = opts;
  const messages = sessionMessages.get(sessionId) ?? [];
  const now = Date.now();
  const decay = (t: number) => {
    const dtMin = (now - t) / 60000;
    return Math.pow(0.5, dtMin / halfLifeMinutes);
  };
  const roleWeight = (role: ChatMessage["role"]) =>
    role === "assistant" ? 1.0 : role === "user" ? 0.9 : 0.5;

  const scored = messages.map((m) => ({
    msg: m,
    score: decay(m.timestamp) * roleWeight(m.role),
    estTokens: Math.ceil(m.content.length / 4),
  }));

  // Sort by weighted recency, then take until budgets hit
  scored.sort((a, b) => b.score - a.score);
  const out: ChatMessage[] = [];
  let tokenBudget = 0;
  for (const s of scored) {
    if (out.length >= maxMessages) break;
    if (tokenBudget + s.estTokens > maxTokens) continue;
    out.push(s.msg);
    tokenBudget += s.estTokens;
  }
  // Preserve chronological order for the final window
  out.sort((a, b) => a.timestamp - b.timestamp);
  return out;
}

// Realtime (WebSocket + optional SSE)
let ws: WebSocket | null = null;
let heartbeat: number | null = null;
let es: EventSource | null = null;

export function connectRealtimeWS(
  url = typeof location !== "undefined"
    ? (() => {
        const env =
          (import.meta as unknown as { env?: Record<string, string> }).env ??
          {};
        const explicit = env["VITE_WS_URL"];
        return explicit || `${location.origin.replace(/^http/, "ws")}/api/ws`;
      })()
    : ""
) {
  if (!url) return;
  try {
    connectionStatus = "connecting";
    ws = new WebSocket(url);
    ws.onopen = () => {
      connectionStatus = "connected";
      if (heartbeat) clearInterval(heartbeat);
      heartbeat = setInterval(
        () =>
          ws?.readyState === WebSocket.OPEN &&
          ws.send(JSON.stringify({ type: "ping" })),
        25_000
      ) as unknown as number;
    };
    ws.onclose = () => {
      connectionStatus = "disconnected";
      if (heartbeat) clearInterval(heartbeat);
    };
    ws.onerror = () => {
      connectionStatus = "error";
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data as string);
        if (data.type === "chat.message")
          addMessage(data.payload as ChatMessage);
        if (data.type === "presence.update")
          setUserActivity(data.payload as UserActivity);
        if (data.type === "recommendations")
          recommendations = data.payload as Recommendation[];
      } catch {
        // ignore
      }
    };
  } catch {
    connectionStatus = "error";
  }
}

export function connectRealtimeSSE(
  url = typeof location !== "undefined" ? `${location.origin}/api/realtime` : ""
) {
  if (!url) return;
  try {
    es = new EventSource(url);
    es.onopen = () => (connectionStatus = "connected");
    es.onerror = () => (connectionStatus = "error");
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "chat.message")
          addMessage(data.payload as ChatMessage);
        if (data.type === "presence.update")
          setUserActivity(data.payload as UserActivity);
        if (data.type === "recommendations")
          recommendations = data.payload as Recommendation[];
      } catch {
        // ignore
      }
    };
  } catch {
    connectionStatus = "error";
  }
}

export function sendRealtime(payload: any) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  // For SSE, send via fetch POST to /api/realtime
  if (!ws && typeof fetch !== "undefined") {
    fetch("/api/realtime", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }
}

export function disconnectRealtime() {
  if (heartbeat) clearInterval(heartbeat);
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  if (es) es.close();
  ws = null;
  es = null;
  connectionStatus = "disconnected";
}
