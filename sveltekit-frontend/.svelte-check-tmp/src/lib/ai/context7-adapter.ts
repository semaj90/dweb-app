// Lightweight adapter to expose safe named helpers backed by the context7-wasm selector shim
// This file provides performContext7Search, context7AgentOrchestrator, and context7SemanticAuditor
// as small wrappers that work with the real WASM binding or the mock implementation.

import context7 from '$lib/integrations/context7-wasm';

export async function ensureContext7Ready(opts?: any) {
  try {
    const impl = await context7;
    if (impl && typeof impl.initialize === 'function') {
      await impl.initialize(opts);
    }
    return impl;
  } catch (e) {
    // swallow - adapter will still return a shim proxy
    return null;
  }
}

export async function performContext7Search(options: { query: string; maxResults?: number; confidenceThreshold?: number; includeCode?: boolean; includeDocs?: boolean; } ) {
  const impl = await ensureContext7Ready();
  if (!impl || typeof (impl as any).performSearch !== 'function') {
    // mock fallback: do a trivial in-memory search stub
    const results = [] as any[];
    for (let i = 0; i < (options.maxResults || 5); i++) {
      results.push({ id: `mock-${i}`, content: `Mock search result for '${options.query}' (#${i})`, score: 0.5 });
    }
    return results;
  }
  return (impl as any).performSearch(options);
}

// Minimal agent orchestrator wrapper. The real implementation exposes methods like
// triggerAgent, logAuditEntry, getAuditLog. The mock will be a small in-memory shim.
export const context7AgentOrchestrator = {
  async triggerAgent(trigger: any) {
    const impl = await ensureContext7Ready();
    if (!impl || typeof (impl as any).triggerAgent !== 'function') {
      // mock behavior: echo back a completed trigger
      return { ...trigger, result: `Mocked trigger for ${trigger.todoId || 'unknown'}`, status: 'completed' };
    }
    return (impl as any).triggerAgent(trigger);
  },
  logAuditEntry(entry: any) {
    // best-effort: call real implementation or noop
    (async () => {
      const impl = await ensureContext7Ready();
      if (impl && typeof (impl as any).logAuditEntry === 'function') {
        try { (impl as any).logAuditEntry(entry); } catch {}
      }
    })();
  },
  getAuditLog() {
    // real impl or empty
    return [] as any[];
  }
};

// Semantic auditor wrapper
export const context7SemanticAuditor = {
  async performSemanticAudit(component: string) {
    const impl = await ensureContext7Ready();
    if (!impl || typeof (impl as any).performSemanticAudit !== 'function') {
      // return a small mocked audit result set
      return [
        { id: 'mock-1', step: 'init-scan', status: 'ok', message: 'Mocked audit step', suggestedFix: null },
      ];
    }
    return (impl as any).performSemanticAudit(component);
  }
};

export default {
  ensureContext7Ready,
  performContext7Search,
  context7AgentOrchestrator,
  context7SemanticAuditor
};
