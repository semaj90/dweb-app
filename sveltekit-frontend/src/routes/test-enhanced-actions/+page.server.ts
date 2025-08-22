import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
  return {
    form: {
      data: {
        title: '',
        description: '',
        priority: 'medium' as const
      },
      errors: {},
      valid: true,
      posted: false
    }
  };
};

export const actions: Actions = {
  testAction: async ({ request }) => {
    const formData = await request.formData();
    const data = {
      title: formData.get('title')?.toString() || '',
      description: formData.get('description')?.toString() || '',
      priority: formData.get('priority')?.toString() || 'medium'
    };

    // Basic validation
    const errors: Record<string, string> = {};
    if (!data.title.trim()) {
      errors.title = 'Title is required';
    }

    if (Object.keys(errors).length > 0) {
      return fail(400, {
        form: {
          data,
          errors,
          valid: false,
          posted: true
        }
      });
    }

    // Simulate processing
    console.log('✅ Enhanced Actions Test - Form submitted:', data);

    return {
      form: {
        data,
        errors: {},
        valid: true,
        posted: true
      },
      success: true,
      message: 'Form submitted successfully!'
    };
  }
};