import type { RequestHandler } from './$types';

async function generateSummary(content: string): Promise<string> {
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gemma2',
      prompt: `Summarize this legal document:\n\n${content}`,
      stream: false
    })
  });
  const data = await response.json();
  return data.response;
}

export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) return new Response('Unauthorized', { status: 401 });
  
  const { content } = await request.json();
  const summary = await generateSummary(content);
  
  return new Response(JSON.stringify({ summary }));
};
