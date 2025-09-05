#!/usr/bin/env node
// Simple validation script for load balancer status & prometheus endpoints

const fetchJson = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
};

(async () => {
  let failures = 0;
  try {
    const status = await fetchJson('http://localhost:8099/status');
    console.log('STATUS:', status);
  } catch (e) {
    failures++; console.error('Status check failed:', e.message);
  }
  try {
    const prom = await fetch('http://localhost:8099/prometheus');
    console.log('PROMETHEUS HEAD:', prom.ok ? 'OK' : prom.status);
    const text = await prom.text();
    console.log(text.split('\n').slice(0,8).join('\n'));
  } catch (e) {
    failures++; console.error('Prometheus check failed:', e.message);
  }
  if (failures) {
    console.error(`❌ ${failures} check(s) failed`);
    process.exit(1);
  }
  console.log('✅ Load balancer tests passed');
})();
