import { writeFile, mkdir } from 'fs/promises';
import { dirname, extname } from 'path';
import { existsSync } from 'fs';
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { file, content, backup } = await request.json();
    
    if (!file || content === undefined) {
      return json({ error: 'File path and content are required' }, { status: 400 });
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

    // Only allow certain file types
    const allowedExtensions = ['.ts', '.js', '.svelte', '.json', '.md', '.css'];
    const fileExt = extname(file);
    if (!allowedExtensions.includes(fileExt)) {
      return json({ error: `File type ${fileExt} not allowed` }, { status: 403 });
    }

    // Create backup if requested
    if (backup && existsSync(file)) {
      const backupPath = `${file}.backup.${Date.now()}`;
      const originalContent = await import('fs/promises').then(fs => fs.readFile(file, 'utf-8'));
      await writeFile(backupPath, originalContent);
    }

    // Ensure directory exists
    const dir = dirname(file);
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true });
    }

    // Write file content
    await writeFile(file, content, 'utf-8');
    
    return json({ 
      success: true,
      file,
      size: content.length,
      lines: content.split('\n').length,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('File write error:', error);
    return json(
      { error: 'Failed to write file', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
};