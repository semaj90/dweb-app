import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals }) => {
  // Check authentication - in production, verify session
  // For demo, we'll allow access
  return {};
};

export const actions: Actions = {
  default: async ({ request, locals }) => {
    const data = await request.formData();
    const title = data.get('title')?.toString();
    const description = data.get('description')?.toString();
    const caseType = data.get('caseType')?.toString();
    const priority = data.get('priority')?.toString();
    const jurisdiction = data.get('jurisdiction')?.toString();
    const assignedDetective = data.get('assignedDetective')?.toString();
    const assignedProsecutor = data.get('assignedProsecutor')?.toString();
    const estimatedDuration = data.get('estimatedDuration')?.toString();
    const budget = data.get('budget')?.toString();

    // Basic validation
    const fieldErrors: Record<string, string> = {};

    if (!title || title.trim().length < 5) {
      fieldErrors.title = 'Case title must be at least 5 characters long';
    }

    if (!description || description.trim().length < 20) {
      fieldErrors.description = 'Case description must be at least 20 characters long';
    }

    if (!caseType) {
      fieldErrors.caseType = 'Case type is required';
    }

    if (!priority) {
      fieldErrors.priority = 'Priority level is required';
    }

    if (!jurisdiction) {
      fieldErrors.jurisdiction = 'Jurisdiction is required';
    }

    if (Object.keys(fieldErrors).length > 0) {
      return fail(400, {
        fieldErrors,
        message: 'Please correct the errors below',
        type: 'error'
      });
    }

    try {
      // Generate case details
      const caseNumber = `${caseType?.substring(0, 3).toUpperCase()}-${new Date().getFullYear()}-${Math.floor(Math.random() * 10000).toString().padStart(4, '0')}`;
      const createdAt = new Date().toISOString();

      // Simulate database save
      const newCase = {
        id: `case_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        caseNumber,
        title: title?.trim(),
        description: description?.trim(),
        caseType,
        priority,
        jurisdiction,
        assignedDetective: assignedDetective?.trim() || null,
        assignedProsecutor: assignedProsecutor?.trim() || null,
        estimatedDuration: estimatedDuration ? parseInt(estimatedDuration) : null,
        budget: budget ? parseFloat(budget) : null,
        status: 'active',
        createdAt,
        createdBy: 'demo_user', // In production, get from locals.user
        updatedAt: createdAt
      };

      // Log case creation for demo
      console.log('Demo case created:', newCase);

      // Simulate database operation delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // In production, you would:
      // 1. Save to database
      // 2. Create initial case timeline entry
      // 3. Send notifications to assigned team members
      // 4. Initialize case folders and documents
      // 5. Log the creation activity

      return {
        message: `Case "${caseNumber}" created successfully!`,
        type: 'success',
        caseData: newCase
      };

    } catch (error) {
      console.error('Case creation error:', error);
      return fail(500, {
        message: 'An error occurred while creating the case. Please try again.',
        type: 'error'
      });
    }
  }
};