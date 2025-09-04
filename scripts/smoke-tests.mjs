#!/usr/bin/env node
// Enhanced runtime smoke tests with dynamic port detection & robust JSON handling

import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';

const candidatePorts = (process.env.PORT_CANDIDATES?.split(',') || Array.from({ length: 8 }, (_, i) => 5173 + i)).map(p => Number(p));
let BASE_URL = process.env.BASE_URL; // Allow explicit override

async function tryFetch(url, opts = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), opts.timeoutMs || 4000);
    try {
        const res = await fetch(url, { signal: controller.signal, ...opts });
        return res;
    } catch (e) {
        return null;
    } finally {
        clearTimeout(timeout);
    }
}

async function detectBaseUrl() {
    if (BASE_URL) return BASE_URL;
    for (const port of candidatePorts) {
        const res = await tryFetch(`http://localhost:${port}/api/ocr/langextract?action=health`, { timeoutMs: 1500 });
        if (res && res.ok) {
            BASE_URL = `http://localhost:${port}`;
            return BASE_URL;
        }
    }
    // Fallback: try simple root fetch to detect vite HTML (will retry later)
    for (const port of candidatePorts) {
        const res = await tryFetch(`http://localhost:${port}/`, { timeoutMs: 800 });
        if (res && res.ok) {
            BASE_URL = `http://localhost:${port}`;
            return BASE_URL;
        }
    }
    throw new Error('Dev server not detected on candidate ports: ' + candidatePorts.join(','));
}

async function waitForEndpoint(path, { attempts = 15, intervalMs = 500 } = {}) {
    const url = `${BASE_URL}${path}`;
    for (let i = 0; i < attempts; i++) {
        const res = await tryFetch(url, { timeoutMs: 1500 });
        if (res && res.ok) return true;
        await delay(intervalMs);
    }
    throw new Error(`Timeout waiting for endpoint ${path}`);
}

async function safeJson(res) {
    if (!res) throw new Error('No response object');
    const text = await res.text();
    try {
        return JSON.parse(text);
    } catch (e) {
        throw new Error(`Invalid JSON (status ${res.status}): ${text.slice(0, 120)}`);
    }
}

async function postJSON(path, body) {
    const res = await fetch(`${BASE_URL}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return { ok: res.ok, status: res.status, json: await safeJson(res) };
}

async function testAnalyze() {
    const { ok, status, json } = await postJSON('/api/ai/analyze', {
        title: 'Test Contract',
        content: 'This Agreement is made between Party A and Party B regarding services.'
    });
    assert.ok(ok, `analyze failed ${status}`);
    assert.ok(json.documentId, 'missing documentId');
    return 'analyze';
}

async function testEmbeddings() {
    const { ok, status, json } = await postJSON('/api/ai/embeddings', {
        text: 'Sample embedding text'
    });
    assert.ok(ok, `embeddings failed ${status}`);
    assert.ok(Array.isArray(json.embedding), 'embedding not array');
    return 'embeddings';
}

async function testOCRHealth() {
    const res = await fetch(`${BASE_URL}/api/ocr/langextract?action=health`);
    assert.ok(res.ok, `ocr health failed ${res.status}`);
    const j = await safeJson(res);
    assert.ok(j.status, 'missing health status');
    return 'ocr';
}

(async () => {
    const results = [];
    try {
        await detectBaseUrl();
        await waitForEndpoint('/api/ocr/langextract?action=health', { attempts: 25, intervalMs: 400 });
        results.push(await testAnalyze());
        await delay(250);
        results.push(await testEmbeddings());
        await delay(250);
        results.push(await testOCRHealth());
        console.log('✅ Smoke tests passed:', results.join(', '), 'on', BASE_URL);
        process.exit(0);
    } catch (e) {
        console.error('❌ Smoke test failure:', e.message);
        console.error('   HINT: Ensure dev server is running (npm run dev) and OCR route available.');
        process.exit(1);
    }
})();
