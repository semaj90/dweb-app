import { fail } from '@sveltejs/kit';
import { superValidate } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { fileUploadSchema } from '$lib/schemas/upload';
import type { Actions, PageServerLoad } from './$types';

const UPLOAD_SERVICE_URL = process.env.UPLOAD_SERVICE_URL || 'http://localhost:8093';

export const load: PageServerLoad = async () => {
  // Initialize the form
  const form = await superValidate(zod(fileUploadSchema));
  
  return {
    form
  };
};

export const actions: Actions = {
  upload: async ({ request, fetch }) => {
    const formData = await request.formData();
    const form = await superValidate(formData, zod(fileUploadSchema));

    if (!form.valid) {
      return fail(400, { form });
    }

    try {
      // Create FormData for upload service
      const uploadFormData = new FormData();
      
      // Add file
      const file = formData.get('file');
      if (!(file instanceof File)) {
        return fail(400, { 
          form, 
          message: 'Invalid file provided' 
        });
      }
      if (!file || file.size === 0) {
        return fail(400, { 
          form, 
          message: 'No file provided' 
        });
      }
      
      uploadFormData.append('file', file);
      uploadFormData.append('caseId', form.data.caseId);
      uploadFormData.append('documentType', form.data.type);
      
      if (form.data.description) {
        uploadFormData.append('description', form.data.description);
      }

      // Add tags if provided
      if (form.data.tags && Array.isArray(form.data.tags) && form.data.tags.length > 0) {
        uploadFormData.append('tags', JSON.stringify(
          (form.data.tags as string[]).reduce((acc, tag) => {
            acc[tag] = 'true';
            return acc;
          }, {} as Record<string, string>)
        ));
      }

      // Add metadata
      const metadata = {
        title: form.data.title,
        isPrivate: form.data.isPrivate.toString(),
        aiAnalysis: form.data.aiAnalysis.toString(),
        uploadedBy: 'user', // TODO: Get from session
        uploadedAt: new Date().toISOString()
      };
      uploadFormData.append('metadata', JSON.stringify(metadata));

      // Upload to MinIO service
      const uploadResponse = await fetch(`${UPLOAD_SERVICE_URL}/upload`, {
        method: 'POST',
        body: uploadFormData
      });

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        console.error('Upload service error:', errorText);
        return fail(uploadResponse.status, {
          form,
          message: `Upload failed: ${errorText}`
        });
      }

      const uploadResult = await uploadResponse.json();

      if (!uploadResult.success) {
        return fail(500, {
          form,
          message: uploadResult.message || 'Upload failed'
        });
      }

      // Return success with upload result
      return {
        form,
        uploadResult,
        message: 'Document uploaded successfully!'
      };

    } catch (error) {
      console.error('Upload error:', error);
      return fail(500, {
        form,
        message: 'Internal server error during upload'
      });
    }
  }
};