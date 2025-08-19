/**
 * Elemental Awareness API - YOLO-style hover analysis
 * Provides legal context for any UI element when hovered
 */
import { json } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { elementType, content, context } = await request.json();

    if (!content || content.length < 3) {
      return json({ relevance: 'No content to analyze' });
    }

    // Quick legal relevance analysis
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt: `Analyze this UI element for legal relevance:

Element: ${elementType}
Content: "${content}"
Context: ${context}

Provide a brief 1-sentence legal relevance assessment and classification.
Format as JSON: {"relevance": "...", "legalContext": "evidence|case|statute|procedure|other", "actionable": true/false}`,
        stream: false
      })
    });

    const result = await response.json();
    
    try {
      const analysis = JSON.parse(result.response);
      return json(analysis);
    } catch (parseError) {
      return json({ 
        relevance: 'Content may have legal significance',
        legalContext: 'general',
        actionable: false
      });
    }
  } catch (error) {
    console.error('Element analysis failed:', error);
    return json({ relevance: 'Analysis unavailable' });
  }
};