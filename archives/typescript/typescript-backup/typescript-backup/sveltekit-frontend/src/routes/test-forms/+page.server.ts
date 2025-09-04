import { fail } from '@sveltejs/kit';
import { superValidate, message } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';
import type { Actions, PageServerLoad } from './$types';

const testSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  email: z.string().email('Invalid email'),
  message: z.string().min(10, 'Message must be at least 10 characters')
});

export const load: PageServerLoad = async () => {
  // Initialize empty form
  const form = await superValidate(zod(testSchema));
  
  return {
    form
  };
};

export const actions: Actions = {
  submit: async ({ request }) => {
    console.log('Server action called');
    
    const form = await superValidate(request, zod(testSchema));
    
    if (!form.valid) {
      console.log('Form validation failed:', form.errors);
      return fail(400, { form });
    }

    console.log('Form data received:', form.data);

    // Simulate processing
    await new Promise(resolve => setTimeout(resolve, 1000));

    return message(form, {
      type: 'success',
      text: 'Form submitted successfully!',
      data: form.data
    });
  }
};