
import { fail, redirect } from '@sveltejs/kit';
import { superValidate, message } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { fileUploadSchema } from '$lib/schemas/upload';
import { db } from '$lib/server/db';
import { evidence } from '$lib/server/db/index';
import { eq } from 'drizzle-orm';
import { ollamaService } from '$lib/services/ollamaService';
import crypto from 'crypto';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import type { PageServerLoad, Actions } from './$types';

const UPLOAD_DIR = 'uploads';

export const load: PageServerLoad = async (): Promise<any> => {
  const form = await superValidate(zod(fileUploadSchema));
  return { form };
};

export const actions: Actions = {
  default: async ({ request, locals }): Promise<any> => {
    const formData = await request.formData();
    const form = await superValidate(formData, zod(fileUploadSchema));

    if (!form.valid) {
      return fail(400, { form });
    }

    try {
      // Get the file from form data
      const file = formData.get('file') as File;
      if (!file) {
        return message(form, 'No file uploaded', { status: 400 });
      }

      // Generate file hash for integrity
      const buffer = await file.arrayBuffer();
      const hash = crypto.createHash('sha256').update(Buffer.from(buffer)).digest('hex');

      // Create upload directory if it doesn't exist
      const uploadPath = join(process.cwd(), UPLOAD_DIR, form.data.caseId);
      await mkdir(uploadPath, { recursive: true });

      // Save file to disk
      const fileName = `${Date.now()}-${file.name}`;
      const filePath = join(uploadPath, fileName);
      await writeFile(filePath, Buffer.from(buffer));

      // Extract text content based on file type
      let extractedText = '';
      if (form.data.type === 'document' && file.type.includes('text')) {
        extractedText = await file.text();
      }
      // For PDFs and other document types, you would use a library like pdf-parse
      // For now, we'll use the description as a placeholder

      // Create evidence record
      const [newEvidence] = await db.insert(evidence).values({
        case_id: form.data.caseId,
        title: form.data.title,
        description: form.data.description,
        evidence_type: form.data.type,
        created_by: locals.user?.id || null,
        metadata: {
          tags: form.data.tags || [],
          summary: extractedText || form.data.description || '',
          originalName: file.name,
          fileSize: file.size,
          mimeType: file.type,
          uploadPath: filePath,
          extractedText: extractedText,
          isPrivate: form.data.isPrivate
        }
      }).returning();

      // Run AI analysis if enabled
      if (form.data.aiAnalysis && extractedText) {
        try {
          // Generate embeddings for the content
          const embeddingResult: any = await ollamaService.embedDocument?.(extractedText /* options may be ignored by some implementations */);
          const chunks = embeddingResult?.chunks || [];

          // Generate AI summary
          const summary: any = await ollamaService.analyzeDocument?.(extractedText);

          // Update evidence with AI analysis in metadata
          await db.update(evidence)
            .set({
              metadata: {
                ...(newEvidence.metadata as any),
                aiSummary: summary,
                embeddingGenerated: true,
                chunksCount: chunks.length
              }
            })
            .where(eq(evidence.id, newEvidence.id));

        } catch (error: any) {
          console.error('AI analysis failed:', error);
          // Continue without AI analysis
        }
      }

      return message(form, 'Evidence uploaded successfully!');
    } catch (error: any) {
      console.error('Upload error:', error);
      return message(form, 'Failed to upload evidence. Please try again.', { status: 500 });
    }
  }
};