import type { RequestHandler } from './$types';

// Simple GPU test endpoint without auth dependencies

export const GET: RequestHandler = async () => {
  return json({
    status: 'GPU Error System Ready',
    models: {
      llm: 'gemma3-legal:latest',
      embedding: 'nomic-embed-text:latest'
    },
    timestamp: new Date().toISOString()
  });
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const mockErrors = [
      'src/test1.ts(10,1): error TS1434: Unexpected keyword or identifier.',
      'src/test2.ts(15,5): error TS2304: Cannot find name "React".',
      'src/test3.ts(20,1): error TS2307: Cannot find module "./missing".',
      'src/test4.ts(25,1): error TS2457: Type alias name cannot be "type".',
      'src/test5.ts(30,1): error TS1005: ";" expected.'
    ];

    const processedErrors = mockErrors.map((line, index) => {
      const match = line.match(/^(.+?)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)$/);
      if (match) {
        const [, file, lineNum, col, code, message] = match;
        return {
          id: `error_${index}`,
          file: file.trim(),
          line: parseInt(lineNum),
          code,
          message: message.trim(),
          fixable: ['TS1434', 'TS2304', 'TS2307', 'TS2457', 'TS1005'].includes(code),
          confidence: 0.8 + Math.random() * 0.2,
          gpuProcessed: true,
          model: 'gemma3-legal:latest'
        };
      }
      return null;
    }).filter(Boolean);

    return json({
      success: true,
      stats: {
        totalErrors: processedErrors.length,
        processedErrors: processedErrors.length,
        fixableErrors: processedErrors.filter(e => e.fixable).length,
        gpuAccelerated: true,
        model: 'gemma3-legal:latest',
        embeddingModel: 'nomic-embed-text:latest'
      },
      errors: processedErrors,
      message: `GPU processed ${processedErrors.length} errors successfully`
    });

  } catch (error: any) {
    return json({ error: 'Processing failed' }, { status: 500 });
  }
};