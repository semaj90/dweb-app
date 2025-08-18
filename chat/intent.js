let transformers = null;
async function load(){ if (!transformers){ try { transformers = await import('@xenova/transformers'); } catch { /* optional */ } } }
const FALLBACK_PATTERNS = [
  { intent:'ask_definition', re:/(what\s+is|define)\s+/i },
  { intent:'compare', re:/(difference between|versus|vs\.?)/i },
  { intent:'obligation_extraction', re:/obligation|duty|responsibility/i },
  { intent:'risk_analysis', re:/risk|liability|exposure/i },
  { intent:'summarize', re:/summarize|summary|brief/i }
];
export async function detectIntent(text){
  await load();
  for (const p of FALLBACK_PATTERNS){ if (p.re.test(text)) return { intent:p.intent, confidence:0.6, method:'regex' }; }
  if (transformers){
    try { const pipeline = await transformers.pipeline('zero-shot-classification'); const labels = FALLBACK_PATTERNS.map(p=>p.intent); const out = await pipeline(text, labels); if (out?.labels?.length){ return { intent: out.labels[0], confidence: out.scores[0], method:'zero-shot' }; } } catch {}
  }
  return { intent:'general_query', confidence:0.3, method:'fallback' };
}
