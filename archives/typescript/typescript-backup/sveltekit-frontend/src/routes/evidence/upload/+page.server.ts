/**
 * Evidence Upload Server Actions
 * Integrates with Superforms + Zod + Rich Evidence Schema
 */

import { fail, redirect } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import crypto from 'crypto';
import { evidenceUploadSchema, getFileTypeFromMime, validateFileSize, validateFileType } from '$lib/schemas/evidence-upload.js';
import { db } from '$lib/server/db/connection.js';
import { evidence, cases } from '$lib/server/db/schema-unified-postgres.js';
import { eq } from 'drizzle-orm';
import type { PageServerLoad, Actions } from './$types';

export const load: PageServerLoad = async ({ locals }) => {
  // Initialize the form with default values
  const form = await superValidate(zod(evidenceUploadSchema));
  
  // Get available cases for the current user
  try {
    const userCases = await db
      .select({
        id: cases.id,
        title: cases.title,
        case_number: cases.case_number,
        status: cases.status
      })
      .from(cases)
      .where(eq(cases.status, 'active'))
      .orderBy(cases.created_at);

    return {
      form,
      cases: userCases
    };
  } catch (error: any) {
    console.error('Failed to load cases:', error);
    return {
      form,
      cases: []
    };
  }
};

export const actions: Actions = {
  upload: async ({ request, locals }) => {
    const formData = await request.formData();
    const form = await superValidate(formData, zod(evidenceUploadSchema));

    if (!form.valid) {
      console.error('Form validation failed:', form.errors);
      return fail(400, { form });
    }

    const file = formData.get('file') as File;
    
    if (!file || file.size === 0) {
      return fail(400, {
        form: {
          ...form,
          errors: { file: ['Please select a file to upload'] }
        }
      });
    }

    // Validate file size
    if (!validateFileSize(file)) {
      return fail(400, {
        form: {
          ...form,
          errors: { file: ['File size exceeds the maximum limit of 100MB'] }
        }
      });
    }

    // Determine evidence type from file if not specified
    let evidenceType = form.data.evidence_type;
    if (evidenceType === 'UNKNOWN' || !evidenceType) {
      evidenceType = getFileTypeFromMime(file.type);
    }

    // Validate file type matches evidence type
    if (!validateFileType(file, evidenceType)) {
      return fail(400, {
        form: {
          ...form,
          errors: { file: [`File type ${file.type} is not supported for ${evidenceType} evidence`] }
        }
      });
    }

    try {
      // Verify the case exists (if case_id is provided)
      if (form.data.case_id) {
        const caseRecord = await db
          .select()
          .from(cases)
          .where(eq(cases.id, form.data.case_id))
          .limit(1);

        if (caseRecord.length === 0) {
          return fail(400, {
            form: {
              ...form,
              errors: { case_id: ['Selected case not found'] }
            }
          });
        }
      }

      // Generate unique storage key
      const fileExtension = path.extname(file.name);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const randomSuffix = crypto.randomBytes(8).toString('hex');
      const storageKey = `evidence/${form.data.case_id}/${timestamp}-${randomSuffix}${fileExtension}`;
      
      // Create upload directory
      const uploadDir = path.join(process.cwd(), 'uploads', 'evidence', form.data.case_id);
      if (!existsSync(uploadDir)) {
        await mkdir(uploadDir, { recursive: true });
      }

      // Save file to disk
      const filePath = path.join(uploadDir, `${timestamp}-${randomSuffix}${fileExtension}`);
      const fileBuffer = Buffer.from(await file.arrayBuffer());
      await writeFile(filePath, fileBuffer);

      // Generate file hash for integrity
      const fileHash = crypto.createHash('sha256').update(fileBuffer).digest('hex');

      // Generate file URL (relative to static assets)
      const fileUrl = `/uploads/evidence/${form.data.case_id || 'default'}/${timestamp}-${randomSuffix}${fileExtension}`;

      // OCR Processing for supported file types
      let ocrResult: any = null;
      if (form.data.enableOcr && (evidenceType === 'PDF' || evidenceType === 'IMAGE')) {
        try {
          // Call OCR service
          const ocrFormData = new FormData();
          ocrFormData.append('file', new Blob([fileBuffer], { type: file.type }), file.name);
          
          const ocrResponse = await fetch('/api/ocr/extract', {
            method: 'POST',
            body: ocrFormData
          });

          if (ocrResponse.ok) {
            ocrResult = await ocrResponse.json();
            console.log('OCR processing completed:', {
              filename: ocrResult.filename,
              pages: ocrResult.pages,
              averageConfidence: ocrResult.averageConfidence,
              legalConceptsFound: ocrResult.legalConcepts?.length || 0,
              citationsFound: ocrResult.citations?.length || 0
            });
          } else {
            console.warn('OCR processing failed:', ocrResponse.statusText);
          }
        } catch (ocrError) {
          console.warn('OCR processing error (non-critical):', ocrError);
        }
      }

      // Generate rich metadata based on file type
      let metadata: any = { 
        kind: evidenceType,
        uploadedAt: new Date().toISOString(),
        fileSize: file.size,
        processingOptions: {
          enableAiAnalysis: form.data.enableAiAnalysis,
          enableOcr: form.data.enableOcr,
          enableEmbeddings: form.data.enableEmbeddings,
          enableSummarization: form.data.enableSummarization
        }
      };

      switch (evidenceType) {
        case 'PDF':
          metadata = {
            ...metadata,
            kind: 'PDF',
            pageCount: ocrResult?.pages || 1,
            isEncrypted: false,
            title: file.name,
            extractedText: ocrResult?.text,
            legalConcepts: ocrResult?.legalConcepts || [],
            citations: ocrResult?.citations || [],
            ocrConfidence: ocrResult?.averageConfidence
          };
          break;

        case 'IMAGE':
          metadata = {
            ...metadata,
            kind: 'IMAGE',
            resolution: { width: 0, height: 0 }, // Would be extracted with sharp
            format: file.type.split('/')[1] || 'unknown',
            hasAlphaChannel: file.type === 'image/png',
            extractedText: ocrResult?.text,
            ocrConfidence: ocrResult?.averageConfidence
          };
          break;

        case 'VIDEO':
          metadata = {
            kind: 'VIDEO',
            durationSeconds: 0, // Would be extracted with ffprobe
            resolution: { width: 0, height: 0 },
            codec: 'unknown',
            frameRate: 0,
            fileSize: file.size,
            uploadedAt: new Date().toISOString()
          };
          break;

        case 'AUDIO':
          metadata = {
            kind: 'AUDIO',
            durationSeconds: 0, // Would be extracted with ffprobe
            codec: 'unknown',
            sampleRate: 44100,
            channels: 2,
            fileSize: file.size,
            uploadedAt: new Date().toISOString()
          };
          break;

        case 'TEXT':
          // For text files, we can read the content
          const textContent = fileBuffer.toString('utf-8');
          metadata = {
            kind: 'TEXT',
            wordCount: textContent.split(/\s+/).filter(word => word.length > 0).length,
            characterCount: textContent.length,
            language: 'unknown', // Could detect with a language detection library
            fileSize: file.size,
            uploadedAt: new Date().toISOString()
          };
          break;

        default:
          metadata = {
            kind: 'UNKNOWN',
            fileSize: file.size,
            uploadedAt: new Date().toISOString()
          };
      }

      // Insert evidence record into database with unified schema
      const evidenceRecord = await db
        .insert(evidence)
        .values({
          case_id: form.data.case_id || null,
          uploader_id: locals.user?.id, // Assuming user session is available
          title: form.data.title,
          description: form.data.description || null,
          evidence_type: evidenceType as any,
          file_url: fileUrl,
          storage_key: storageKey,
          file_hash: `sha256:${fileHash}`,
          file_size: file.size.toString(),
          metadata: {
            ...metadata,
            // Include additional unified schema fields
            tags: form.data.tags || [],
            confidentialityLevel: form.data.confidentialityLevel || 'standard',
            isAdmissible: form.data.isAdmissible !== false,
            collectedAt: form.data.collectedAt || new Date().toISOString(),
            collectedBy: form.data.collectedBy || 'system',
            location: form.data.location,
            chainOfCustody: form.data.chainOfCustody || [],
            ocrResult: ocrResult ? {
              extractedText: ocrResult.text,
              confidence: ocrResult.averageConfidence,
              legalConcepts: ocrResult.legalConcepts,
              citations: ocrResult.citations,
              pageCount: ocrResult.pages
            } : null
          }
        })
        .returning();

      console.log('Evidence uploaded successfully:', {
        id: evidenceRecord[0].id,
        title: evidenceRecord[0].title,
        type: evidenceRecord[0].evidence_type,
        size: file.size,
        hash: fileHash.substring(0, 8) + '...'
      });

      // Trigger Go Upload Service for additional processing
      try {
        console.log('📤 Sending file to Go upload service for processing...');
        
        const uploadFormData = new FormData();
        uploadFormData.append('file', new Blob([fileBuffer], { type: file.type }), file.name);
        uploadFormData.append('evidenceId', evidenceRecord[0].id);
        uploadFormData.append('caseId', form.data.case_id || '');
        uploadFormData.append('title', form.data.title);
        uploadFormData.append('evidenceType', evidenceType);
        
        const goServiceResponse = await fetch('http://localhost:5173/api/upload/go-service', {
          method: 'POST',
          body: uploadFormData
        });

        if (goServiceResponse.ok) {
          const goResult = await goServiceResponse.json();
          console.log('✅ Go service processing completed:', goResult);
          
          // Update metadata with Go service results if available
          if (goResult.embeddings || goResult.analysis) {
            metadata = {
              ...metadata,
              goServiceProcessing: {
                embeddings: goResult.embeddings,
                analysis: goResult.analysis,
                processedAt: new Date().toISOString()
              }
            };
          }
        } else {
          console.warn('⚠️ Go upload service processing failed:', goServiceResponse.statusText);
          console.warn('Continuing with local processing only');
        }
      } catch (goServiceError) {
        console.warn('⚠️ Go upload service error (non-critical):', goServiceError);
        console.warn('Continuing with local processing only');
        // Don't fail the upload, just log the warning
      }

    } catch (error: any) {
      console.error('Evidence upload error:', error);
      return fail(500, {
        form: {
          ...form,
          errors: { file: ['Failed to upload file. Please try again.'] }
        }
      });
    }

    // Redirect to evidence list or case details
    throw redirect(302, `/cases/${form.data.case_id}/evidence`);
  }
};