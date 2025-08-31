import { readFile } from 'fs/promises';
import { existsSync } from 'fs';
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { file } = await request.json();
    
    if (!file) {
      return json({ error: 'File path is required' }, { status: 400 });
    }

    // Security check - ensure file is within project bounds
    const allowedPaths = [
      'src/',
      '.svelte-kit/',
      'static/',
      'tests/'
    ];

    const isAllowed = allowedPaths.some(path => file.startsWith(path));
    if (!isAllowed) {
      return json({ error: 'Access to file path not allowed' }, { status: 403 });
    }

    // Check if file exists
    if (!existsSync(file)) {
      return json({ error: 'File not found' }, { status: 404 });
    }

    // Read file content
    const content = await readFile(file, 'utf-8');
    
    return json({ 
      file,
      content,
      size: content.length,
      lines: content.split('\n').length
    });

  } catch (error: any) {
    console.error('File read error:', error);
    return json(
      { error: 'Failed to read file', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
};