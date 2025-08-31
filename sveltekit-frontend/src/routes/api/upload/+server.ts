import { json } from "@sveltejs/kit";
import { randomUUID } from "crypto";
import { existsSync, createReadStream } from "fs";
import { mkdir, writeFile, readFile } from "fs/promises";
import * as path from "path";
import { db } from '../../../lib/server/db/index';
import { evidence } from '../../../lib/server/db/schema-postgres';
import type { RequestHandler } from './$types';


// Ensure upload directory exists
const UPLOAD_DIR = "./uploads";

// Basic upload handler
export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return json({ error: 'No file provided' }, { status: 400 });
    }

    const fileId = randomUUID();
    const fileName = `${fileId}_${file.name}`;
    const filePath = path.join(UPLOAD_DIR, fileName);

    // Ensure upload directory exists
    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true });
    }

    // Save file
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filePath, buffer);

    return json({
      success: true,
      fileId,
      fileName: file.name,
      filePath,
      size: file.size
    });

  } catch (error: any) {
    console.error('Upload error:', error);
    return json({ error: 'Upload failed' }, { status: 500 });
  }
};