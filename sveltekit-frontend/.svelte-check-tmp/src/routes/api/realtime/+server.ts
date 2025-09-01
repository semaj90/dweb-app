import stream from "stream";
import type { RequestHandler } from './$types';


// Minimal SSE endpoint: emits keepalive and relays posted messages to connected clients
const clients = new Set<WritableStreamDefaultWriter<string>>();

export const GET: RequestHandler = async () => {
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();
  clients.add(writer);

  // Send welcome and keepalive
  const send = (obj: any) =>
    writer.write(`data: ${JSON.stringify(obj)}\n\n`);
  send({ type: "welcome", ts: Date.now() });

  const keep = setInterval(
    () => send({ type: "keepalive", ts: Date.now() }),
    30000
  );
  const abort = new AbortController();
  const onAbort = () => {
    clearInterval(keep);
    clients.delete(writer);
    writer.close().catch(() => {});
  };
  abort.signal.addEventListener("abort", onAbort);

  return new Response(stream.readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const data = await request.json();
    const payload = `data: ${JSON.stringify(data)}\n\n`;
    await Promise.all(
      Array.from(clients).map((w) =>
        w.write(payload).catch(() => clients.delete(w))
      )
    );
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  } catch (e: any) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), {
      status: 400,
    });
  }
};
