import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals, url }) => {
  // Simple form data for testing SuperForms + Enhanced Actions
  const form = {
    data: {
      title: '',
      description: '',
      priority: 'medium' as const
    },
    errors: {},
    valid: true,
    posted: false
  };
  
  // Pre-populate form if editing (check for case ID in URL) - temporarily disabled for testing
  const caseId = url.searchParams.get('edit');
  // Temporarily skip database operations for SuperForms testing
  // if (caseId) {
  //   // Fetch existing case data
  //   try {
  //     // Replace with your actual database call
  //     const existingCase = await locals.db.case.findUnique({
  //       where: { id: caseId }
  //     });
  //     
  //     if (existingCase) {
  //       // Pre-populate form with existing data
  //       form.data = {
  //         caseNumber: existingCase.caseNumber,
  //         title: existingCase.title,
  //         description: existingCase.description || '',
  //         priority: existingCase.priority,
  //         status: existingCase.status,
  //         assignedTo: existingCase.assignedTo || undefined,
  //         dueDate: existingCase.dueDate?.toISOString().slice(0, 16) || undefined,
  //         tags: existingCase.tags || [],
  //         isConfidential: existingCase.isConfidential || false,
  //         notifyAssignee: existingCase.notifyAssignee || true
  //       };
  //     }
  //   } catch (error) {
  //     console.error('Failed to load case for editing:', error);
  //   }
  // }

  return {
    form,
    editMode: !!caseId,
    caseId
  };
};

export const actions: Actions = {
  createCase: async ({ request, locals }) => {
    // Parse form data manually for testing Enhanced Actions
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

    const form = {
      data,
      errors,
      valid: Object.keys(errors).length === 0,
      posted: true
    };

    // Return form with errors if validation fails
    if (!form.valid) {
      return fail(400, { form });
    }

    try {
      // Simulate successful case creation for testing Enhanced Actions
      console.log('✅ Enhanced Actions Test - Form submitted:', data);

      // Process uploaded files
      const formData = await request.formData();
      const attachments = [];
      
      // Extract all uploaded files
      for (const [key, value] of formData.entries()) {
        if (key.startsWith('attachments[') && value instanceof File && value.size > 0) {
          attachments.push({
            file: value,
            originalName: value.name,
            size: value.size,
            type: value.type
          });
        }
      }

      // Validate file uploads
      const maxFileSize = 10 * 1024 * 1024; // 10MB
      const allowedTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'image/jpeg',
        'image/png'
      ];

      for (const attachment of attachments) {
        if (attachment.size > maxFileSize) {
          return fail(400, {
            form,
            message: `File ${attachment.originalName} exceeds 10MB limit`
          });
        }
        
        if (!allowedTypes.includes(attachment.type)) {
          return fail(400, {
            form,
            message: `File type ${attachment.type} is not allowed`
          });
        }
      }

      // Check for duplicate case number
      const existingCase = await locals.db.case.findUnique({
        where: { caseNumber }
      });

      if (existingCase) {
        return fail(409, {
          form,
          message: 'A case with this number already exists'
        });
      }

      // Process file uploads to storage
      const uploadedFiles = [];
      for (const attachment of attachments) {
        try {
          // Upload to your file storage (S3, MinIO, etc.)
          const fileUrl = await locals.storage.upload({
            file: attachment.file,
            bucket: 'case-documents',
            path: `cases/${caseNumber}/documents/`
          });

          uploadedFiles.push({
            originalName: attachment.originalName,
            fileName: fileUrl.split('/').pop(),
            url: fileUrl,
            size: attachment.size,
            mimeType: attachment.type,
            uploadedAt: new Date()
          });
        } catch (uploadError) {
          console.error('File upload failed:', uploadError);
          return fail(500, {
            form,
            message: `Failed to upload file: ${attachment.originalName}`
          });
        }
      }

      // Create case in database
      const newCase = await locals.db.case.create({
        data: {
          caseNumber,
          title,
          description: description || null,
          priority,
          status,
          assignedTo: assignedTo || null,
          dueDate: dueDate ? new Date(dueDate) : null,
          tags: tags || [],
          isConfidential: isConfidential || false,
          notifyAssignee: notifyAssignee ?? true,
          createdBy: locals.user.id,
          documents: {
            create: uploadedFiles.map(file => ({
              originalName: file.originalName,
              fileName: file.fileName,
              url: file.url,
              size: file.size,
              mimeType: file.mimeType,
              uploadedBy: locals.user.id,
              uploadedAt: file.uploadedAt
            }))
          }
        },
        include: {
          documents: true,
          assignedUser: {
            select: {
              id: true,
              name: true,
              email: true
            }
          },
          createdByUser: {
            select: {
              id: true,
              name: true,
              email: true
            }
          }
        }
      });

      // Send notifications if enabled
      if (notifyAssignee && assignedTo) {
        try {
          await locals.notifications.send({
            userId: assignedTo,
            type: 'case_assigned',
            title: 'New Case Assigned',
            message: `You have been assigned to case: ${title}`,
            data: {
              caseId: newCase.id,
              caseNumber: newCase.caseNumber,
              priority: newCase.priority
            }
          });
        } catch (notificationError) {
          console.error('Failed to send notification:', notificationError);
          // Don't fail the entire operation for notification failures
        }
      }

      // Log case creation for audit trail
      await locals.audit.log({
        action: 'case_created',
        userId: locals.user.id,
        resourceType: 'case',
        resourceId: newCase.id,
        details: {
          caseNumber: newCase.caseNumber,
          title: newCase.title,
          priority: newCase.priority,
          documentsCount: uploadedFiles.length
        }
      });

      // Return success with case data
      return message(form, {
        type: 'success',
        text: `Case ${caseNumber} created successfully`,
        data: {
          case: newCase,
          redirectUrl: `/cases/${newCase.id}`
        }
      });

    } catch (error) {
      console.error('Case creation failed:', error);
      
      // Database constraint violation
      if (error.code === 'P2002') {
        return fail(409, {
          form,
          message: 'A case with this number already exists'
        });
      }
      
      // Generic server error
      return fail(500, {
        form,
        message: 'Failed to create case. Please try again.'
      });
    }
  },

  updateCase: async ({ request, locals, url }) => {
    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      return fail(400, { message: 'Case ID is required' });
    }

    const form = await superValidate(request, zod(testCaseSchema));

    if (!form.valid) {
      return fail(400, { form });
    }

    try {
      // Check if case exists and user has permission
      const existingCase = await locals.db.case.findUnique({
        where: { id: caseId },
        include: { documents: true }
      });

      if (!existingCase) {
        return fail(404, {
          form,
          message: 'Case not found'
        });
      }

      // Check permissions (owner or assigned user)
      if (existingCase.createdBy !== locals.user.id && 
          existingCase.assignedTo !== locals.user.id) {
        return fail(403, {
          form,
          message: 'You do not have permission to edit this case'
        });
      }

      const { 
        caseNumber, 
        title, 
        description, 
        priority, 
        status, 
        assignedTo, 
        dueDate,
        tags,
        isConfidential,
        notifyAssignee 
      } = form.data;

      // Update case
      const updatedCase = await locals.db.case.update({
        where: { id: caseId },
        data: {
          caseNumber,
          title,
          description,
          priority,
          status,
          assignedTo,
          dueDate: dueDate ? new Date(dueDate) : null,
          tags,
          isConfidential,
          notifyAssignee,
          updatedAt: new Date()
        },
        include: {
          documents: true,
          assignedUser: {
            select: { id: true, name: true, email: true }
          }
        }
      });

      // Log update
      await locals.audit.log({
        action: 'case_updated',
        userId: locals.user.id,
        resourceType: 'case',
        resourceId: updatedCase.id,
        details: {
          changes: {
            // Compare with existing case to log specific changes
            title: existingCase.title !== title ? { from: existingCase.title, to: title } : undefined,
            priority: existingCase.priority !== priority ? { from: existingCase.priority, to: priority } : undefined,
            status: existingCase.status !== status ? { from: existingCase.status, to: status } : undefined
          }
        }
      });

      return message(form, {
        type: 'success',
        text: 'Case updated successfully',
        data: { case: updatedCase }
      });

    } catch (error) {
      console.error('Case update failed:', error);
      return fail(500, {
        form,
        message: 'Failed to update case. Please try again.'
      });
    }
  },

  saveDraft: async ({ request, locals }) => {
    const form = await superValidate(request, zod(caseFormSchema.partial()));

    try {
      // Save partial form data as draft
      const draft = await locals.db.caseDraft.upsert({
        where: {
          userId_draftKey: {
            userId: locals.user.id,
            draftKey: 'case_creation'
          }
        },
        update: {
          data: form.data,
          updatedAt: new Date()
        },
        create: {
          userId: locals.user.id,
          draftKey: 'case_creation',
          data: form.data
        }
      });

      return message(form, {
        type: 'success',
        text: 'Draft saved successfully',
        data: { draft }
      });

    } catch (error) {
      console.error('Draft save failed:', error);
      return fail(500, {
        form,
        message: 'Failed to save draft'
      });
    }
  }
};