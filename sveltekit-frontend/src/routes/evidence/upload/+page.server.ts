// Repaired evidence upload page server logic (simplified)
import { fail } from '@sveltejs/kit';
import { superValidate, message } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';

import { db } from '$lib/server/db';
import { evidence, documentEmbeddings } from '$lib/server/db/schema-unified';

import { eq } from 'drizzle-orm';
import crypto from 'crypto';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';

import { ollamaService } from '$lib/server/ai/ollama-service';
import { cuid } from '$lib/utils/cuid';
import type { PageServerLoad, Actions } from './$types';

const fileUploadSchema = z.object({
  caseId: z.string().min(1),
  title: z.string().min(1),
  description: z.string().optional(),
  type: z.string().default('document'),
  tags: z.array(z.string()).optional(),
  aiAnalysis: z.boolean().optional(),
  isPrivate: z.boolean().optional()
});

const UPLOAD_DIR = 'uploads';

export const load: PageServerLoad = async () => {
  const form = await superValidate(zod(fileUploadSchema));
  return { form };
};

export const actions: Actions = {
  default: async ({ request, locals }) => {
    const formData = await request.formData();
    const form = await superValidate(formData, zod(fileUploadSchema));

    if (!form.valid) {
      return fail(400, { form });
    }

    try {
      // Get the file from form data
      const file = formData.get('file') as File | null;
      if (!file) {
        return message(form, 'No file uploaded', { status: 400 });
      }

      // Generate file hash for integrity
      const buffer = await file.arrayBuffer();
      const hash = crypto.createHash('sha256').update(Buffer.from(buffer)).digest('hex');

      // Create upload directory if it doesn't exist
      const caseId = String(form.data.caseId);
      const uploadPath = join(process.cwd(), UPLOAD_DIR, caseId);
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
        id: cuid(),
        caseId: form.data.caseId,
        case_id: form.data.caseId, // alias for database field
        title: form.data.title,
        description: form.data.description,
        evidenceType: form.data.type,
        evidence_type: form.data.type, // alias for database field
        type: form.data.type, // another alias
        content: extractedText,
        createdBy: locals.user?.id || null,
        tags: form.data.tags || [],
        metadata: {
          hash: hash,
          originalName: file.name,
          fileSize: file.size,
          mimeType: file.type,
          uploadPath: filePath,
          extractedText: extractedText,
          isPrivate: form.data.isPrivate
        },
        created_at: new Date(),
        updated_at: new Date()
      }).returning();

      // Run AI analysis if enabled
      if (form.data.aiAnalysis && extractedText) {
        try {
          // Generate embeddings for the content
          let chunks = [];
          try {
            if (ollamaService.embedDocument) {
              const embeddingResult = await ollamaService.embedDocument(extractedText);
              chunks = embeddingResult || [];
            }
          } catch (embedError) {
            console.error('Embedding generation failed:', embedError);
            chunks = [];
          }

          // Store document vectors
          for (const chunk of chunks) {
            await db.insert(documentEmbeddings).values({
              evidenceId: newEvidence.id, // Using evidence ID
              chunkIndex: chunk.metadata?.chunkIndex || 0,
              content: chunk.content,
              embedding: chunk.embedding as any, // Type assertion for vector
              metadata: chunk.metadata || {}
            });
          }

          // Generate AI summary
          let summary = '';
          try {
            if (ollamaService.analyzeDocument) {
              const result = await ollamaService.analyzeDocument(extractedText);
              summary = typeof result === 'string' ? result : (result as any)?.summary || '';
            }
          } catch (summaryError) {
            console.error('AI summary generation failed:', summaryError);
            summary = '';
          }

          // Update evidence with AI analysis
          await db.update(evidence)
            .set({
              aiSummary: summary,
              aiAnalysis: {
                ...newEvidence.aiAnalysis as any,
                embeddingGenerated: true,
                chunksCount: chunks.length
              }
            })
            .where(eq(evidence.id, newEvidence.id));

        } catch (error) {
          console.error('AI analysis failed:', error);
          // Continue without AI analysis
        }
      }

      return message(form, 'Evidence uploaded successfully!');
    } catch (error) {
      console.error('Upload error:', error);
      return message(form, 'Failed to upload evidence. Please try again.', { status: 500 });
    }
  }
};