import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: 'Unauthorized' }, { status: 401 });
    }

    const formData = await request.formData();
    const file = formData.get('file') as File;
    const userId = formData.get('userId') as string;

    if (!file) {
      return json({ error: 'No file provided' }, { status: 400 });
    }

    // Simple file handling - in production, use proper file storage
    const buffer = await file.arrayBuffer();
    const content = new TextDecoder().decode(buffer);

    const evidence = {
      id: Date.now().toString(),
      filename: file.name,
      content: content.substring(0, 1000), // Truncate for demo
      metadata: {
        size: file.size,
        type: file.type,
        uploadedAt: new Date().toISOString()
      },
      uploadedAt: new Date().toISOString(),
      userId
    };

    // In production, save to database here
    console.log('Evidence uploaded:', evidence.filename);

    return json(evidence);
  } catch (error) {
    console.error('Upload error:', error);
    return json({ error: 'Upload failed' }, { status: 500 });
  }
};
