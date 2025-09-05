

  import { superForm } from 'sveltekit-superforms/client';
  import { zod } from 'sveltekit-superforms/adapters';
  import { z } from 'zod';
  import { createMachine, assign } from 'xstate';
  import { useMachine } from '@xstate/svelte';
  import { Button } from 'bits-ui';
  import { Input } from 'bits-ui';
  import { Textarea } from 'bits-ui';
  import { Select } from 'bits-ui';
  import { Checkbox } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { createEventDispatcher } from 'svelte';
  
  // Props
  let { 
    initialData = {},
    onSubmit,
    class: className = ''
  } = $props();

  // Zod schema for form validation
  const documentMetadataSchema = z.object({
    title: z.string().min(1, 'Title is required').max(500, 'Title too long'),
    description: z.string().max(2000, 'Description too long').optional(),
    documentType: z.enum([
      'contract', 'motion', 'evidence', 'correspondence', 'brief', 'regulation', 'case_law'
    ]),
    practiceArea: z.enum([
      'corporate', 'litigation', 'intellectual_property', 'employment', 
      'real_estate', 'criminal', 'family', 'tax', 'immigration', 'environmental'
    ]).optional(),
    jurisdiction: z.enum(['federal', 'state', 'local']).default('federal'),
    priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium'),
    isConfidential: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    customFields: z.record(z.string()).optional(),
    retentionDate: z.string().optional(),
    assignedAttorney: z.string().optional(),
    clientName: z.string().optional(),
    caseNumber: z.string().optional(),
    
    // Analysis preferences
    analysisOptions: z.object({
      includeEmbeddings: z.boolean().default(true),
      includeSummary: z.boolean().default(true),
      includeEntities: z.boolean().default(true),
      includeRiskAnalysis: z.boolean().default(true),
      includeCompliance: z.boolean().default(false),
      priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium')
    }).default({})
  });

  type FormData = z.infer;

  // XState machine for form state management
  const formMachine = createMachine({
    id: 'documentMetadataForm',
    initial: 'editing',
    context: {
      data: initialData,
      errors: {},
      isValid: false,
      isDirty: false,
      submitCount: 0
    },
    states: {
      editing: {
        on: {
          VALIDATE: {
            target: 'validating',
            actions: assign({
              data: ({ event }) => event.data
            })
          },
          SUBMIT: {
            target: 'submitting',
            guard: ({ context }) => context.isValid
          },
          RESET: {
            target: 'editing',
            actions: assign({
              data: initialData,
              errors: {},
              isDirty: false
            })
          }
        }
      },
      validating: {
        invoke: {
          src: 'validateForm',
          onDone: {
            target: 'editing',
            actions: assign({
              errors: ({ event }) => event.output.errors,
              isValid: ({ event }) => event.output.isValid,
              isDirty: true
            })
          },
          onError: {
            target: 'editing',
            actions: assign({
              errors: ({ event }) => ({ form: event.error.message }),
              isValid: false
            })
          }
        }
      },
      submitting: {
        invoke: {
          src: 'submitForm',
          onDone: {
            target: 'success',
            actions: assign({
              submitCount: ({ context }) => context.submitCount + 1
            })
          },
          onError: {
            target: 'error',
            actions: assign({
              errors: ({ event }) => ({ form: event.error.message })
            })
          }
        }
      },
      success: {
        after: {
          3000: 'editing'
        }
      },
      error: {
        on: {
          RETRY: 'submitting',
          EDIT: 'editing'
        }
      }
    }
  }, {
    actors: {
      validateForm: async ({ input }) => {
        try {
          const result = documentMetadataSchema.safeParse(input.data);
          return {
            isValid: result.success,
            errors: result.success ? {} : result.error.flatten().fieldErrors
          };
        } catch (error) {
          throw new Error('Validation failed');
        }
      },
      submitForm: async ({ input }) => {
        if (onSubmit) {
          return await onSubmit(input.data);
        }
        throw new Error('No submit handler provided');
      }
    }
  });

  const [state, send] = useMachine(formMachine);

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    submit: FormData;
    change: FormData;
    reset: void;
  }>();

  // Form setup with Superforms
  const { form, errors, constraints, enhance } = superForm({}, {
    SPA: true,
    validators: zod(documentMetadataSchema),
  });

  // Dropdown options
  const documentTypes = [
    { value: 'contract', label: 'Contract' },
    { value: 'motion', label: 'Motion' },
    { value: 'evidence', label: 'Evidence' },
    { value: 'correspondence', label: 'Correspondence' },
    { value: 'brief', label: 'Brief' },
    { value: 'regulation', label: 'Regulation' },
    { value: 'case_law', label: 'Case Law' }
  ];

  const practiceAreas = [
    { value: 'corporate', label: 'Corporate' },
    { value: 'litigation', label: 'Litigation' },
    { value: 'intellectual_property', label: 'Intellectual Property' },
    { value: 'employment', label: 'Employment' },
    { value: 'real_estate', label: 'Real Estate' },
    { value: 'criminal', label: 'Criminal' },
    { value: 'family', label: 'Family' },
    { value: 'tax', label: 'Tax' },
    { value: 'immigration', label: 'Immigration' },
    { value: 'environmental', label: 'Environmental' }
  ];

  const jurisdictions = [
    { value: 'federal', label: 'Federal' },
    { value: 'state', label: 'State' },
    { value: 'local', label: 'Local' }
  ];

  const priorities = [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'critical', label: 'Critical' }
  ];

  // Local state for form data
  let formData = $state({
    title: '',
    documentType: 'evidence',
    jurisdiction: 'federal',
    priority: 'medium',
    isConfidential: false,
    tags: [],
    analysisOptions: {
      includeEmbeddings: true,
      includeSummary: true,
      includeEntities: true,
      includeRiskAnalysis: true,
      includeCompliance: false,
      priority: 'medium'
    },
    ...initialData
  });

  // Tag management
  let newTag = $state('');

  function addTag() {
    if (newTag.trim() && !formData.tags.includes(newTag.trim())) {
      formData.tags = [...formData.tags, newTag.trim()];
      newTag = '';
      validateForm();
    }
  }

  function removeTag(tagToRemove: string) {
    formData.tags = formData.tags.filter(tag => tag !== tagToRemove);
    validateForm();
  }

  // Form validation
  async function validateForm() {
    send({ type: 'VALIDATE', data: formData });
  }

  // Form submission
  async function handleSubmit() {
    if ($state.matches('editing') && $state.context.isValid) {
      send({ type: 'SUBMIT' });
      dispatch('submit', formData);
    }
  }

  // Reset form
  function resetForm() {
    send({ type: 'RESET' });
    formData = {
      title: '',
      documentType: 'evidence',
      jurisdiction: 'federal',
      priority: 'medium',
      isConfidential: false,
      tags: [],
      analysisOptions: {
        includeEmbeddings: true,
        includeSummary: true,
        includeEntities: true,
        includeRiskAnalysis: true,
        includeCompliance: false,
        priority: 'medium'
      },
      ...initialData
    };
    dispatch('reset');
  }

  // Watch for changes
  $effect(() => {
    dispatch('change', formData);
    validateForm();
  });

  // Helper functions
  function getStateMessage(): string {
    if ($state.matches('validating')) return 'Validating...';
    if ($state.matches('submitting')) return 'Submitting...';
    if ($state.matches('success')) return 'Successfully submitted!';
    if ($state.matches('error')) return 'Submission failed';
    return '';
  }

  function getStateColor(): string {
    if ($state.matches('validating') || $state.matches('submitting')) return 'text-blue-600';
    if ($state.matches('success')) return 'text-green-600';
    if ($state.matches('error')) return 'text-red-600';
    return 'text-gray-600';
  }

  function isFieldError(field: string): boolean {
    return !!$state.context.errors[field];
  }

  function getFieldError(field: string): string {
    const error = $state.context.errors[field];
    return Array.isArray(error) ? error[0] : error || '';
  }



  
    
    
      
        Document Metadata
        
          Provide information about the document for better AI analysis and organization
        
      
      
      
      {#if getStateMessage()}
        
          {getStateMessage()}
        
      {/if}
    

    
      
      
        
        
          
            Document Title *
          
          <Input.Root
            bind:value={formData.title}
            placeholder="Enter document title..."
            class={isFieldError('title') ? 'border-red-500' : ''}
            {...$constraints.title}
          />
          {#if isFieldError('title')}
            {getFieldError('title')}
          {/if}
        

        
        
          
            Document Type *
          
          
            
              
            
            
              {#each documentTypes as type}
                
                  {type.label}
                
              {/each}
            
          
          {#if isFieldError('documentType')}
            {getFieldError('documentType')}
          {/if}
        

        
        
          
            Practice Area
          
          
            
              
            
            
              {#each practiceAreas as area}
                
                  {area.label}
                
              {/each}
            
          
        

        
        
          
            Jurisdiction
          
          
            
              
            
            
              {#each jurisdictions as jurisdiction}
                
                  {jurisdiction.label}
                
              {/each}
            
          
        

        
        
          
            Priority
          
          
            
              
            
            
              {#each priorities as priority}
                
                  {priority.label}
                
              {/each}
            
          
        
      

      
      
        
          Description
        
        <Textarea.Root
          bind:value={formData.description}
          placeholder="Optional description or notes about the document..."
          class={`min-h-[100px] ${isFieldError('description') ? 'border-red-500' : ''}`}
          {...$constraints.description}
        />
        {#if isFieldError('description')}
          {getFieldError('description')}
        {/if}
      

      
      
        
          
            Client Name
          
          <Input.Root
            bind:value={formData.clientName}
            placeholder="Client or party name..."
          />
        

        
          
            Case Number
          
          <Input.Root
            bind:value={formData.caseNumber}
            placeholder="Associated case number..."
          />
        

        
          
            Assigned Attorney
          
          <Input.Root
            bind:value={formData.assignedAttorney}
            placeholder="Attorney name..."
          />
        
      

      
      
        
          Tags
        
        
          <Input.Root
            bind:value={newTag}
            placeholder="Add a tag..."
            class="flex-1"
            on:keydown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
          />
          <Button.Root
            type="button"
            variant="outline"
            on:click={addTag}
            disabled={!newTag.trim()}
          >
            Add Tag
          
        
        {#if formData.tags.length > 0}
          
            {#each formData.tags as tag}
              <Badge.Root 
                variant="secondary" 
                class="flex items-center space-x-1 cursor-pointer hover:bg-gray-200"
                on:click={() => removeTag(tag)}
              >
                {tag}
                
                  
                
              
            {/each}
          
        {/if}
      

      
      
        AI Analysis Options
        
        
          
            
              
                
                Generate vector embeddings
              
            

            
              
                
                Generate AI summary
              
            

            
              
                
                Extract entities and key terms
              
            
          

          
            
              
                
                Perform risk analysis
              
            

            
              
                
                Check compliance requirements
              
            

            
              
                
                Mark as confidential
              
            
          
        

        
        
          
            Processing Priority
          
          
            
              
            
            
              {#each priorities as priority}
                
                  {priority.label}
                
              {/each}
            
          
        
      

      
      
        
          Retention Date (Optional)
        
        <Input.Root
          type="date"
          bind:value={formData.retentionDate}
          class="w-full md:w-48"
        />
        
          Date when this document should be reviewed for retention/deletion
        
      

      
      
        
          {#if $state.context.isDirty}
            Form has unsaved changes
          {:else if $state.context.submitCount > 0}
            Form submitted {$state.context.submitCount} time{$state.context.submitCount === 1 ? '' : 's'}
          {/if}
        

        
          <Button.Root
            type="button"
            variant="outline"
            on:click={resetForm}
            disabled={$state.matches('submitting')}
          >
            Reset
          

          <Button.Root
            type="submit"
            disabled={!$state.context.isValid || $state.matches('submitting')}
            class="min-w-[100px]"
          >
            {#if $state.matches('submitting')}
              Submitting...
            {:else}
              Submit
            {/if}
          
        
      
    
  



  .document-metadata-form {
    @apply max-w-4xl mx-auto;
  }
