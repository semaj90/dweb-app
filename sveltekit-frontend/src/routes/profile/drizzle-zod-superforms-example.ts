// ================================
// DRIZZLE-ZOD + SUPERFORMS COMPATIBILITY GUIDE
// ================================

import { superValidate } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { 
  profileTableUpdateSchema, 
  profileUpdateZodSchema, 
  extractZodSchema 
} from '$lib/db/schema';

// ❌ PROBLEM: This doesn't work directly with SuperForms
// const form = await superValidate(data, zod(profileTableUpdateSchema));
// Error: BuildSchema is not assignable to ZodObjectType

// ✅ SOLUTION 1: Extract the underlying schema manually
const schema1 = profileTableUpdateSchema._def.schema;
const form1 = await superValidate(data, zod(schema1));

// ✅ SOLUTION 2: Use the helper function
const schema2 = extractZodSchema(profileTableUpdateSchema);
const form2 = await superValidate(data, zod(schema2));

// ✅ SOLUTION 3: Use pre-extracted schemas (cleanest approach)
const schema3 = profileUpdateZodSchema;
const form3 = await superValidate(data, zod(schema3));

// ================================
// USAGE PATTERN IN YOUR ROUTES
// ================================

export const load = (async ({ locals }) => {
  const userData = {
    id: 'user-123',
    firstName: 'John', 
    lastName: 'Doe'
  };

  // Option A: Extract on-the-fly
  const form = await superValidate(userData, zod(profileTableUpdateSchema._def.schema));

  // Option B: Use helper function
  const form = await superValidate(userData, zod(extractZodSchema(profileTableUpdateSchema)));

  // Option C: Use pre-extracted schema (recommended)
  const form = await superValidate(userData, zod(profileUpdateZodSchema));

  return { form };
};

export const actions = {
  update: async ({ request }) => {
    // Same pattern for form validation
    const form = await superValidate(request, zod(profileUpdateZodSchema));

    if (!form.valid) {
      return { form };
    }

    // Process form.data
    console.log('Updated data:', form.data);
    return { form };
  }
};

// ================================
// PARTIAL SCHEMAS
// ================================

import { z } from 'zod';

// Create partial schemas for specific use cases
const profilePartialSchema = profileUpdateZodSchema.pick({
  firstName: true,
  lastName: true
});

// Or create custom schemas that extend the base
const profileWithEmailSchema = profileUpdateZodSchema.extend({
  email: z.string().email()
});

export { 
  profilePartialSchema, 
  profileWithEmailSchema 
};