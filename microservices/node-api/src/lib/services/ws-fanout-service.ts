// WebSocket fan-out service bridging NATS AI pipeline updates to browser clients.
import { WebSocketServer } from 'ws';
import { subscribe, NATS_SUBJECTS } from './nats-messaging-service';

const PORT = parseInt(process.env.WS_FANOUT_PORT || '8080', 10);
const clients = new Set<any>();

const wss = new WebSocketServer({ port: PORT });
console.log(`[ws-fanout] WebSocket server listening on :${PORT}`);

wss.on('connection', (ws) => {
  clients.add(ws);
  console.log('[ws-fanout] client connected, total=', clients.size);
  ws.on('close', () => clients.delete(ws));
});

function broadcast(obj: any){
  const payload = JSON.stringify(obj);
  for (const ws of clients){
    try { ws.send(payload); } catch {}
  }
}

// Subscribe to AI responses and evidence uploads for streaming progress.
(async () => {
  await subscribe(NATS_SUBJECTS.EVIDENCE_UPLOAD, (msg) => broadcast({ type: 'evidence.upload', msg }));
  // Single subject carries all stage + final markers; we fan-out by inspecting stage/final flags
  await subscribe(NATS_SUBJECTS.AI_RESPONSE, (msg) => {
    if (msg.final) return broadcast({ type: 'ai.final', msg });
    if (msg.stage) return broadcast({ type: `ai.stage.${msg.stage}`, msg });
    broadcast({ type: 'ai.response', msg });
  });
})();
