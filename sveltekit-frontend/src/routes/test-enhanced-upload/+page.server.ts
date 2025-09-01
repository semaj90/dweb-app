import type { PageServerLoad } from './$types';
import { superValidate } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { fileUploadSchema } from '$lib/schemas/file-upload.js';
import type { Actions, PageServerLoad } from './$types';
import { fail } from '@sveltejs/kit';

export const load: PageServerLoad = async () => {
;
  // Initialize the form with default values for testing
  const form = await superValidate(zod(fileUploadSchema));

  // Set some test defaults
  form.data.caseId = 'TEST-CASE-' + Date.now();
  form.data.enableAiAnalysis = true;
  form.data.enableOcr = true;
  form.data.enableEmbeddings = true;
  form.data.evidenceType = 'documents';

  return {
    form
  };
};

export const actions: Actions = {
;
  upload: async ({ request, fetch }) => {
    const formData = await request.formData();

    // Validate the form data
    const form = await superValidate(formData, zod(fileUploadSchema));

    if (!form.valid) {
      return fail(400, {
        form,
        message: 'Validation failed'
      });
    }

    try {
      const file = formData.get('file') as File;
      const enhancedAnalysis = formData.get('enhancedAnalysis');

      if (!file) {
        return fail(400, {
          form,
          message: 'No file provided'
        });
      }

      console.log('📤 Processing enhanced upload:', {
        filename: file.name,
        size: file.size,
        type: file.type,
        caseId: form.data.caseId,
        hasEnhancedAnalysis: !!enhancedAnalysis
      });

      // Simulate successful upload processing
      const uploadResult = {
        success: true,
        documentId: 'DOC-' + Date.now(),
        filename: file.name,
        size: file.size,
        type: file.type,
        caseId: form.data.caseId,
        uploadedAt: new Date().toISOString(),
        enhancedProcessing: !!enhancedAnalysis,
        analysis: enhancedAnalysis ? JSON.parse(enhancedAnalysis as string) : null,
        webhookTriggered: !!enhancedAnalysis,
        metadata: {
          title: form.data.title,
          description: form.data.description,
          evidenceType: form.data.evidenceType,
          tags: form.data.tags?.split(',').map(tag => tag.trim()).filter(Boolean),
          flags: {
            enableAiAnalysis: form.data.enableAiAnalysis,
            enableOcr: form.data.enableOcr,
            enableEmbeddings: form.data.enableEmbeddings,
            isAdmissible: form.data.isAdmissible
          }
        }
      };

      return {
        form,
        uploadResult,
        message: `File "${file.name}" uploaded successfully with enhanced processing!`
      };

    } catch (error: any) {
      console.error('❌ Upload processing error:', error);

      return fail(500, {
        form,
        message: `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    }
  }
};