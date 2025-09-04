import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { superValidate, message } from 'sveltekit-superforms/server';
import { zod } from 'sveltekit-superforms/adapters';
import { z } from 'zod';

const caseFormSchema = z.object({
  title: z.string().min(1, 'Title is required'),
  description: z.string().min(1, 'Description is required'),
  priority: z.enum(['low', 'medium', 'high'] as const)
});

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
  //   } catch (error: any) {
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
      caseNumber: formData.get('caseNumber')?.toString() || '',
      title: formData.get('title')?.toString() || '',
      description: formData.get('description')?.toString() || '',
      priority: formData.get('priority')?.toString() || 'medium',
      assignedTo: formData.get('assignedTo')?.toString() || null,
      dueDate: formData.get('dueDate')?.toString() || null,
      tags: formData.get('tags')?.toString()?.split(',').map(t => t.trim()).filter(Boolean) || [],
      isConfidential: formData.get('isConfidential') === 'true',
      notifyAssignee: formData.get('notifyAssignee') !== 'false'
    };

    // Basic validation
    const errors: Record<string, string> = {};
    if (!data.title.trim()) {
      errors.title = 'Title is required';
    }
    if (!data.caseNumber.trim()) {
      errors.caseNumber = 'Case number is required';
    }

    // Destructure the data for easier access - only destructure properties that exist
    const { title, description, priority } = data;
    const { caseNumber, assignedTo, dueDate, tags, isConfidential, notifyAssignee } = data;

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
      const uploadFormData = await request.formData();
      const attachments = [];

      // Extract all uploaded files
      for (const [key, value] of uploadFormData.entries()) {
        if (key.startsWith('attachments[') && value instanceof File && (value as File).size > 0) {
          const file = value as File;
          attachments.push({
            file,
            originalName: file.name,
            size: file.size,
            type: file.type
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

      // Skip duplicate check for now - this would require proper Drizzle setup
      // TODO: Implement duplicate case number check with Drizzle ORM
      // const existingCase = await db.select().from(cases)
      //   .where(eq(cases.caseNumber, caseNumber)).limit(1);
      // if (existingCase.length > 0) {
      //   return fail(409, {
      //     form,
      //     message: 'A case with this number already exists'
      //   });
      // }

      // Prepare a lightweight ingestion payload for our enhanced RAG pipeline.
      // The pipeline (orchestrator / microservice) will:
      //  - persist text & metadata to Postgres (with pgvector columns)
      //  - create embeddings (nomic / embed service)
      //  - push vectors to Qdrant and pgvector
      //  - run auto-tagging in Qdrant
      //  - upload files to MinIO (bucket) and attach metadata links to Postgres
      //  - cache frequent queries / indexes, schedule GPU jobs if needed
      const ingestionPayload = {
        type: 'case_creation',
        case: {
          caseNumber,
          title,
          description,
          priority,
          assignedTo,
          dueDate,
          tags,
          isConfidential,
          notifyAssignee,
          createdBy: locals.user?.id,
          createdAt: new Date().toISOString()
        },
        attachments: attachments.map(a => ({
          name: a.originalName,
          size: a.size,
          mimeType: a.type
        })),
        storage: {
          bucket: 'case-documents',
          basePath: `cases/${caseNumber}/documents/`
        },
        featureFlags: {
          embedWith: 'nomic',     // instruct orchestrator to use Nomic / configured embedder
          persistVectorTo: ['pgvector', 'qdrant'],
          autoTagWith: 'qdrant',
          cacheHits: true,
          scheduleGpu: true
        }
      };

      // Fire-and-forget: prefer orchestrator enqueue API if available.
      try {
        if (locals.orchestrator?.enqueue) {
          // enqueue a job in our GPU/orchestration system (recommended)
          await locals.orchestrator.enqueue('ingest-case', ingestionPayload);
        } else if (locals.orchestrator?.ingestUrl) {
          // fallback: HTTP call to ingestion microservice
          fetch(`${locals.orchestrator.ingestUrl.replace(/\/$/, '')}/ingest/case`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              ...(locals.orchestrator.apiKey ? { Authorization: `Bearer ${locals.orchestrator.apiKey}` } : {})
            },
            body: JSON.stringify(ingestionPayload)
          }).catch((err) => console.error('Ingestion service request failed:', err));
        } else {
          // If no orchestrator is configured, attempt a minimal local best-effort:
          //  - upload attachments to MinIO (if locals.minio available) and return metadata links
          if (locals.minio?.putObject && attachments.length) {
            for (const att of attachments) {
              try {
                const key = `cases/${caseNumber}/documents/${Date.now()}_${att.originalName}`;
                // putObject should accept (bucket, key, file)
                const url = await locals.minio.putObject('case-documents', key, att.file);
                // attach the returned URL into the ingestion payload (so downstream DB create can reference it)
                ingestionPayload.attachments = ingestionPayload.attachments.map(x =>
                  x.name === att.originalName ? { ...x, url } : x
                );
              } catch (putErr) {
                console.error('MinIO upload failed for', att.originalName, putErr);
                // tolerate upload errors here; orchestrator can retry later
              }
            }
          }

          // If we have an embed helper locally, create a small embedding now and store it in a lightweight table
          if (locals.embed?.embedText) {
            try {
              const text = [title, description].filter(Boolean).join('\n\n');
              const vector = await locals.embed.embedText(text); // returns number[]
              // store the vector payload in a lightweight 'pending_vectors' table for later ingestion
              await locals.db.pendingVector.create({
                data: {
                  sourceType: 'case',
                  sourceKey: caseNumber,
                  vector: vector as any,
                  metadata: {
                    title,
                    priority,
                    createdBy: locals.user?.id
                  }
                }
              });
            } catch (embedErr) {
              console.error('Local embedding (best-effort) failed:', embedErr);
            }
          }

          // As a final best-effort, persist the ingestionPayload to a queue table so a worker can pick it up
          try {
            await locals.db.ingestQueue.create({
              data: {
                jobType: 'case_creation',
                payload: ingestionPayload
              }
            });
          } catch (qErr) {
            console.error('Failed to enqueue ingestion job to DB queue:', qErr);
          }
        }
      } catch (err: any) {
        console.error('Failed to start ingestion pipeline:', err);
        // Do not fail the main request; ingestion should be best-effort/asynchronous.
      }
      //   where: { caseNumber }
      // });
      // if (existingCase) {
      //   return fail(409, {
      //     form,
      //     message: 'A case with this number already exists'
      //   });
      // }

      // Process file uploads to storage
      const uploadedFiles = [];
      for (const attachment of attachments) {
        // File upload commented out for now - requires proper storage setup
        // try {
        //   const fileUrl = await locals.storage.upload({
        //     file: attachment.file,
        //     bucket: 'case-documents',
        //     path: `cases/${caseNumber}/documents/`
        //   });
        //   uploadedFiles.push({
        //     originalName: attachment.originalName,
        //     fileName: fileUrl.split('/').pop(),
        //     url: fileUrl,
        //     size: attachment.size,
        //     mimeType: attachment.type,
        //     uploadedAt: new Date()
        //   });
        // } catch (uploadError) {
        //   console.error('File upload failed:', uploadError);
        //   return fail(500, {
        //     form,
        //     message: `Failed to upload file: ${attachment.originalName}`
        //   });
        // }
      }

      // Create case in database and persist embedding/vector (Drizzle/Prisma compatible best-effort)
      try {
        // 1) generate embedding (nomic preferred, fallback to local embed / llama wasm)
        const textForEmbedding = [title, description].filter(Boolean).join('\n\n');
        let vector: number[] | null = null;
        try {
          if (typeof (globalThis as any).nomicEmbedText === 'function') {
            vector = await (globalThis as any).nomicEmbedText(textForEmbedding);
          } else if (locals?.embed?.embedText) {
            vector = await locals.embed.embedText(textForEmbedding);
          } else if (locals?.llamaWasm?.embedText) {
            vector = await locals.llamaWasm.embedText(textForEmbedding);
          } else if (locals?.llama?.embed) {
            vector = await locals.llama.embed(textForEmbedding);
          }
        } catch (embedErr) {
          console.error('Embedding generation failed (best-effort):', embedErr);
          vector = null;
        }

        // 2) Case creation temporarily simplified - would need proper Drizzle setup
        const mockCase = {
          id: `mock-${Date.now()}`,
          caseNumber,
          title,
          description,
          priority,
          assignedTo,
          createdBy: locals.user?.id,
          createdAt: new Date()
        };

        // 3) Vector persistence temporarily disabled - would need proper Drizzle setup
        if (vector) {
          console.log('Vector generated but not persisted (mock mode):', vector.length, 'dimensions');
        }

          // pgvector operations temporarily disabled - would need proper setup
          console.log('pgvector operations skipped (mock mode)');

      } catch (dbErr) {
        console.error('Case creation DB error:', dbErr);
        if (dbErr?.code === 'P2002' || dbErr?.code === '23505') {
          return fail(409, { form, message: 'A case with this number already exists' });
        }
        return fail(500, { form, message: 'Failed to create case in DB' });
      }
      // const newCase = await locals.db.case.create({
      //   data: {
      //     caseNumber,
      //     title,
      //     description: description || null,
      //     priority,
      //     assignedTo: assignedTo || null,
      //     dueDate: dueDate ? new Date(dueDate) : null,
      //     tags: tags || [],
      //     isConfidential: isConfidential || false,
      //     notifyAssignee: notifyAssignee ?? true,
      //     createdBy: locals.user?.id,
      //     documents: {
      //       create: uploadedFiles.map(file => ({
      //         originalName: file.originalName,
      //         fileName: file.fileName,
      //         url: file.url,
      //         size: file.size,
      //         mimeType: file.mimeType,
      //         uploadedBy: locals.user?.id,
      //         uploadedAt: file.uploadedAt
      //       }))
      //     },
      //     // If pgvector column exists, persist the vector
      //     ...(vector ? { embedding: vector as any } : {})
      //     title,
      //     description: description || null,
      //     priority,
      //     assignedTo: assignedTo || null,
      //     dueDate: dueDate ? new Date(dueDate) : null,
      //     tags: tags || [],
      //     isConfidential: isConfidential || false,
      //     notifyAssignee: notifyAssignee ?? true,
      //     createdBy: locals.user?.id,
      //     documents: {
      //       create: uploadedFiles.map(file => ({
      //         originalName: file.originalName,
      //         fileName: file.fileName,
      //         url: file.url,
      //         size: file.size,
      //         mimeType: file.mimeType,
      //         uploadedBy: locals.user?.id,
      //         uploadedAt: file.uploadedAt
      //       }))
      //     }
      //   },
      //   include: {
      //     documents: true,
      //     assignedUser: {
      //       select: {
      //         id: true,
      //         name: true,
      //         email: true
      //       }
      //     },
      //     createdByUser: {
      //       select: {
      //         id: true,
      //         name: true,
      //         email: true
      //       }
      //     }
      //   }
      // });

      // Send notifications if enabled (commented out)
      // if (notifyAssignee && assignedTo) {
      //   try {
      //     await locals.notifications?.send({
      //       userId: assignedTo,
      //       type: 'case_assigned',
      //       title: 'New Case Assigned',
      //       message: `You have been assigned to case: ${title}`,
      //       data: {
      //         caseId: 'mock-id',
      //         caseNumber: caseNumber,
      //         priority: priority
      //       }
      //     });
      //   } catch (notificationError) {
      //     console.error('Failed to send notification:', notificationError);
      //     // Don't fail the entire operation for notification failures
      //   }
      // }

      // Log case creation for audit trail (commented out)
      // await locals.audit?.log({
      //   action: 'case_created',
      //   userId: locals.user?.id,
      //   resourceType: 'case',
      //   resourceId: 'mock-id',
      //   details: {
      //     caseNumber: caseNumber,
      //     title: title,
      //     priority: priority,
      //     documentsCount: uploadedFiles.length
      //   }
      // });

      // Return success with case data (simplified for testing)
      return {
        form: {
          ...form,
          valid: true
        },
        success: true,
        message: `Case ${caseNumber} created successfully`,
        data: {
          caseNumber,
          title,
          priority
        }
      };

    } catch (error: any) {
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

    const form = await superValidate(request, zod(caseFormSchema));

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
        title,
        description,
        priority
      } = form.data;

      // Update case
      const updatedCase = await locals.db.case.update({
        where: { id: caseId },
        data: {
          title,
          description,
          priority,
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
            // status: existingCase.status !== status ? { from: existingCase.status, to: status } : undefined
          }
        }
      });

      return message(form, {
        type: 'success',
        text: 'Case updated successfully',
        data: { case: updatedCase }
      });

    } catch (error: any) {
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

    } catch (error: any) {
      console.error('Draft save failed:', error);
      return fail(500, {
        form,
        message: 'Failed to save draft'
      });
    }
  }
};