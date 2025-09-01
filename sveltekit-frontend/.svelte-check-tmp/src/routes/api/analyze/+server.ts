import type { RequestHandler } from './$types';

// Clean, self-contained analysis endpoint that avoids referencing missing DB schema symbols.
// - Validates input
// - Calls local Ollama at http://localhost:11434/api/generate (if available)
// - Returns structured JSON analysis
// Note: Re-enable DB interactions once your drizzle schema and table symbols are available.

import { json } from '@sveltejs/kit';
import { createHash } from 'crypto';

type AnalysisRequest = {
  text?: string;
  documentId?: string;
  evidenceId?: string;
  caseId?: string;
  documentType?: 'evidence' | 'case_file' | 'legal_document' | 'ocr_scan';
  analysisType?: 'classification' | 'extraction' | 'reasoning' | 'compliance' | 'chain_of_custody';
  useThinkingStyle?: boolean;
  contextDocuments?: string[];
  userId?: string;
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = (await request.json()) as AnalysisRequest;
    const startTime = Date.now();

    // Basic validation
    if (!body.text && !body.evidenceId && !body.caseId) {
      return json({ error: 'Missing required field: text, evidenceId, or caseId' }, { status: 400 });
    }

    // For safety this implementation does not depend on DB lookups; prefer provided text
    const documentText = body.text ?? '';
    const documentMetadata: any = {}; // placeholder for callers that provide metadata
    const contextualInfo = ''; // no case context resolved in this simplified handler

    const modelName = body.useThinkingStyle ? 'legal-gemma3-thinking' : 'gemma3:7b';
    const prompt = buildEnhancedAnalysisPrompt(
      documentText,
      body.analysisType ?? 'classification',
      body.documentType ?? 'legal_document',
      !!body.useThinkingStyle,
      contextualInfo,
      documentMetadata
    );

    let aiContent = '';
    try {
      const resp = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal:latest',
          prompt,
          stream: false,
          options: {
            temperature: 0.2,
            top_p: 0.9,
            num_ctx: 4096,
          }
        })
      });

      if (!resp.ok) {
        console.warn(`Ollama API returned ${resp.status}: ${resp.statusText}`);
        aiContent = 'AI analysis unavailable (service error)';
      } else {
        const data = await resp.json();
        aiContent = data.response || 'No analysis returned';
      }
    } catch (fetchError) {
      console.warn('Could not connect to Ollama:', fetchError);
      aiContent = 'AI analysis unavailable (connection error)';
    }

    // Generate a request ID for logging
    const requestId = createHash('sha256').update(`${Date.now()}-${JSON.stringify(body)}`).digest('hex').slice(0, 8);

    const result = {
      requestId,
      analysis: aiContent,
      metadata: {
        documentType: body.documentType ?? 'legal_document',
        analysisType: body.analysisType ?? 'classification',
        useThinkingStyle: !!body.useThinkingStyle,
        textLength: documentText.length,
        processingTimeMs: Date.now() - startTime,
        timestamp: new Date().toISOString(),
        model: modelName
      }
    };

    return json(result);
  } catch (error: any) {
    console.error('Analysis endpoint error:', error);
    return json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
};

function buildEnhancedAnalysisPrompt(
  documentText: string,
  analysisType: string,
  documentType: string,
  useThinkingStyle: boolean,
  contextualInfo: string,
  documentMetadata: any
): string {
  const basePrompt = `Analyze this ${documentType} document for ${analysisType}.

Document Text:
${documentText}

${contextualInfo ? `Context: ${contextualInfo}` : ''}

Provide a structured analysis focusing on:
1. Key findings
2. Legal relevance
3. Compliance issues (if any)
4. Recommendations

Format: JSON with clear sections for each point.`;

  if (useThinkingStyle) {
    return `<thinking>
Let me analyze this ${documentType} for ${analysisType} purposes.

First, I'll examine the content structure and identify key elements...
Then I'll assess legal implications and compliance requirements...
Finally, I'll provide actionable recommendations...
</thinking>

${basePrompt}`;
  }

  return basePrompt;
}