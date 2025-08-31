// Architecture validation script (CommonJS)
// Checks a few critical service endpoints/ports and reports a short summary.
const http = require('http');
const net = require('net');

function checkPort(host, port, timeout = 2000) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let done = false;
    socket.setTimeout(timeout);
    socket.on('connect', () => { done = true; socket.destroy(); resolve({ host, port, ok: true }); });
    socket.on('timeout', () => { if (!done) { done = true; socket.destroy(); resolve({ host, port, ok: false, reason: 'timeout' }); }});
    socket.on('error', () => { if (!done) { done = true; resolve({ host, port, ok: false }); }});
    socket.connect(port, host);
  });
}

(async () => {
  const services = [
    { name: 'SvelteKit (dev)', host: '127.0.0.1', port: 5173 },
    { name: 'Ollama', host: '127.0.0.1', port: 11434 },
    { name: 'Postgres', host: '127.0.0.1', port: 5432 },
    { name: 'Redis', host: '127.0.0.1', port: 6379 }
  ];

  console.log('🔍 Verifying service endpoints...');
  const results = await Promise.all(services.map(s => checkPort(s.host, s.port).then(r => ({ ...s, ...r }))));
  results.forEach(r => {
    console.log(`${r.ok ? '✅' : '❌'} ${r.name} (${r.host}:${r.port})` + (r.reason ? ` - ${r.reason}` : ''));
  });
  const allOk = results.every(r => r.ok);
  process.exit(allOk ? 0 : 2);
})();
