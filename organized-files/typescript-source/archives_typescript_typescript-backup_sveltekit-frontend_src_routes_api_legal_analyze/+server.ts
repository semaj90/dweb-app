import type { RequestHandler } from '@sveltejs/kit';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const { documentId, analysisType = 'full' } = body;
    if (!documentId) {
      return new Response(JSON.stringify({ error: 'documentId required' }), { status: 400 });
    }
    // Placeholder analysis output
    return new Response(JSON.stringify({
      documentId,
      analysisType,
      summary: 'Placeholder summary. Connect to server ollamaService.analyzeLegalDocument()',
      keyPoints: [],
      entities: {},
      model: 'gemma3-legal:latest'
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: e.message || 'Analysis failed' }), { status: 500 });
  }
};
