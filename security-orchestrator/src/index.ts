import Fastify from 'fastify';
import cors from '@fastify/cors';
import websocket from '@fastify/websocket';
import { securityValidationRequestSchema, securityValidationResponseSchema } from './schemas.js';
import { computeRisk } from './riskScoring.js';
import { randomUUID } from 'node:crypto';

const PORT = Number(process.env.SECURITY_ORCH_PORT || 8600);

const fastify = Fastify({ logger: true });
await fastify.register(cors, { origin: true });
await fastify.register(websocket);

fastify.get('/health', async () => ({ status: 'ok', service: 'mcpGPUOrchestrator', port: PORT, timestamp: new Date().toISOString() }));

fastify.post('/validate/security', async (request, reply) => {
  const json = request.body;
  const parsed = securityValidationRequestSchema.safeParse(json);
  if (!parsed.success) {
    return reply.status(400).send({ error: 'Invalid payload', issues: parsed.error.issues });
  }
  const start = performance.now();
  const risk = computeRisk(parsed.data);

  // Basic structural verifications (placeholders for future external API calls)
  const verification = {
    emailFormatValid: /@/.test(parsed.data.user.email),
    usernameValid: /^[a-zA-Z0-9_\-]{3,32}$/.test(parsed.data.user.username),
    roleAllowed: true,
    referralValid: parsed.data.user.referralCode ? /[A-Z0-9]{6,}/.test(parsed.data.user.referralCode) : true
  };

  const response = {
    requestId: risk.requestId,
    riskScore: risk.riskScore,
    securityScore: risk.securityScore,
    verification,
    signals: risk.signals,
    status: risk.status,
    modelVersion: risk.modelVersion,
    durationMs: risk.durationMs + Math.round(performance.now() - start),
    timestamp: new Date().toISOString()
  };

  const validated = securityValidationResponseSchema.parse(response);
  return validated;
});

// Realtime stream (optional): send incremental scoring updates (mocked)
fastify.get('/ws/security', { websocket: true }, (connection /*, req */) => {
  const id = randomUUID();
  connection.socket.send(JSON.stringify({ type: 'welcome', id }));
  let step = 0;
  const interval = setInterval(() => {
    step++;
    connection.socket.send(JSON.stringify({ type: 'progress', step, pct: Math.min(100, step * 20) }));
    if (step >= 5) {
      connection.socket.send(JSON.stringify({ type: 'complete', id }));
      clearInterval(interval);
    }
  }, 300);
  connection.socket.on('close', () => clearInterval(interval));
});

try {
  await fastify.listen({ port: PORT, host: '0.0.0.0' });
  console.log(`Security Orchestrator listening on ${PORT}`);
} catch (err) {
  fastify.log.error(err);
  process.exit(1);
}
