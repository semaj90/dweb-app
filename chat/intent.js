let transformers = null;

async function load() {
  if (!transformers) {
    try {
      transformers = await import('@xenova/transformers');
    } catch (error) {
      // Transformers library is optional - system will use regex fallback
      console.warn('Transformers library not available, using regex fallback for intent detection');
    }
  }
}

const FALLBACK_PATTERNS = [
  { intent: 'ask_definition', re: /(what\s+is|define)\s+/i },
  { intent: 'compare', re: /(difference between|versus|vs\.?)/i },
  { intent: 'obligation_extraction', re: /obligation|duty|responsibility/i },
  { intent: 'risk_analysis', re: /risk|liability|exposure/i },
  { intent: 'summarize', re: /summarize|summary|brief/i },
  { intent: 'contract_analysis', re: /contract|agreement|clause/i },
  { intent: 'legal_research', re: /precedent|case law|statute/i },
  { intent: 'compliance_check', re: /compliance|regulation|requirement/i }
];

export async function detectIntent(text) {
  if (!text || typeof text !== 'string') {
    return { intent: 'general_query', confidence: 0.1, method: 'fallback', error: 'Invalid input' };
  }

  await load();

  // First try regex patterns for fast, reliable detection
  for (const pattern of FALLBACK_PATTERNS) {
    if (pattern.re.test(text)) {
      return {
        intent: pattern.intent,
        confidence: 0.6,
        method: 'regex'
      };
    }
  }

  // Try transformer-based zero-shot classification if available
  if (transformers) {
    try {
      const pipeline = await transformers.pipeline('zero-shot-classification');
      const labels = FALLBACK_PATTERNS.map(p => p.intent);
      const result = await pipeline(text, labels);
      
      if (result?.labels?.length && result.scores?.length) {
        return {
          intent: result.labels[0],
          confidence: result.scores[0],
          method: 'zero-shot',
          alternatives: result.labels.slice(1, 3).map((label, i) => ({
            intent: label,
            confidence: result.scores[i + 1]
          }))
        };
      }
    } catch (error) {
      console.warn('Zero-shot classification failed:', error.message);
    }
  }

  // Fallback to general query
  return {
    intent: 'general_query',
    confidence: 0.3,
    method: 'fallback'
  };
}

// Export additional utility functions
export function getAvailableIntents() {
  return FALLBACK_PATTERNS.map(p => p.intent);
}

export function isTransformersAvailable() {
  return transformers !== null;
}
