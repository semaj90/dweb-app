<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
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

  type FormData = z.infer<typeof documentMetadataSchema>;

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
  let formData = $state<FormData>({
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
</script>

<Card.Root class={`document-metadata-form ${className}`}>
  <div class="p-6">
    <!-- Form Header -->
    <div class="flex items-center justify-between mb-6">
      <div>
        <h3 class="text-lg font-semibold text-gray-900">Document Metadata</h3>
        <p class="text-sm text-gray-600 mt-1">
          Provide information about the document for better AI analysis and organization
        </p>
      </div>
      
      <!-- State Indicator -->
      {#if getStateMessage()}
        <div class={`text-sm font-medium ${getStateColor()}`}>
          {getStateMessage()}
        </div>
      {/if}
    </div>

    <form use:enhance on:submit|preventDefault={handleSubmit} class="space-y-6">
      <!-- Basic Information -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Title -->
        <div class="md:col-span-2">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Document Title *
          </label>
          <Input.Root
            bind:value={formData.title}
            placeholder="Enter document title..."
            class={isFieldError('title') ? 'border-red-500' : ''}
            {...$constraints.title}
          />
          {#if isFieldError('title')}
            <p class="text-sm text-red-600 mt-1">{getFieldError('title')}</p>
          {/if}
        </div>

        <!-- Document Type -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Document Type *
          </label>
          <Select.Root bind:selected={formData.documentType}>
            <Select.Trigger class={isFieldError('documentType') ? 'border-red-500' : ''}>
              <Select.Value placeholder="Select document type..." />
            </Select.Trigger>
            <Select.Content>
              {#each documentTypes as type}
                <Select.Item value={type.value}>
                  {type.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
          {#if isFieldError('documentType')}
            <p class="text-sm text-red-600 mt-1">{getFieldError('documentType')}</p>
          {/if}
        </div>

        <!-- Practice Area -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Practice Area
          </label>
          <Select.Root bind:selected={formData.practiceArea}>
            <Select.Trigger>
              <Select.Value placeholder="Select practice area..." />
            </Select.Trigger>
            <Select.Content>
              {#each practiceAreas as area}
                <Select.Item value={area.value}>
                  {area.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
        </div>

        <!-- Jurisdiction -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Jurisdiction
          </label>
          <Select.Root bind:selected={formData.jurisdiction}>
            <Select.Trigger>
              <Select.Value />
            </Select.Trigger>
            <Select.Content>
              {#each jurisdictions as jurisdiction}
                <Select.Item value={jurisdiction.value}>
                  {jurisdiction.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
        </div>

        <!-- Priority -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Priority
          </label>
          <Select.Root bind:selected={formData.priority}>
            <Select.Trigger>
              <Select.Value />
            </Select.Trigger>
            <Select.Content>
              {#each priorities as priority}
                <Select.Item value={priority.value}>
                  {priority.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
        </div>
      </div>

      <!-- Description -->
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Description
        </label>
        <Textarea.Root
          bind:value={formData.description}
          placeholder="Optional description or notes about the document..."
          class={`min-h-[100px] ${isFieldError('description') ? 'border-red-500' : ''}`}
          {...$constraints.description}
        />
        {#if isFieldError('description')}
          <p class="text-sm text-red-600 mt-1">{getFieldError('description')}</p>
        {/if}
      </div>

      <!-- Case Information -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Client Name
          </label>
          <Input.Root
            bind:value={formData.clientName}
            placeholder="Client or party name..."
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Case Number
          </label>
          <Input.Root
            bind:value={formData.caseNumber}
            placeholder="Associated case number..."
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Assigned Attorney
          </label>
          <Input.Root
            bind:value={formData.assignedAttorney}
            placeholder="Attorney name..."
          />
        </div>
      </div>

      <!-- Tags -->
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Tags
        </label>
        <div class="flex space-x-2 mb-3">
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
          </Button.Root>
        </div>
        {#if formData.tags.length > 0}
          <div class="flex flex-wrap gap-2">
            {#each formData.tags as tag}
              <Badge.Root 
                variant="secondary" 
                class="flex items-center space-x-1 cursor-pointer hover:bg-gray-200"
                on:click={() => removeTag(tag)}
              >
                <span>{tag}</span>
                <svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </Badge.Root>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Advanced Options -->
      <div class="border-t pt-6">
        <h4 class="text-md font-medium text-gray-900 mb-4">AI Analysis Options</h4>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="space-y-3">
            <Checkbox.Root bind:checked={formData.analysisOptions.includeEmbeddings}>
              <div class="flex items-center space-x-2">
                <Checkbox.Indicator />
                <span class="text-sm">Generate vector embeddings</span>
              </div>
            </Checkbox.Root>

            <Checkbox.Root bind:checked={formData.analysisOptions.includeSummary}>
              <div class="flex items-center space-x-2">
                <Checkbox.Indicator />
                <span class="text-sm">Generate AI summary</span>
              </div>
            </Checkbox.Root>

            <Checkbox.Root bind:checked={formData.analysisOptions.includeEntities}>
              <div class="flex items-center space-x-2">
                <Checkbox.Indicator />
                <span class="text-sm">Extract entities and key terms</span>
              </div>
            </Checkbox.Root>
          </div>

          <div class="space-y-3">
            <Checkbox.Root bind:checked={formData.analysisOptions.includeRiskAnalysis}>
              <div class="flex items-center space-x-2">
                <Checkbox.Indicator />
                <span class="text-sm">Perform risk analysis</span>
              </div>
            </Checkbox.Root>

            <Checkbox.Root bind:checked={formData.analysisOptions.includeCompliance}>
              <div class="flex items-center space-x-2">
                <Checkbox.Indicator />
                <span class="text-sm">Check compliance requirements</span>
              </div>
            </Checkbox.Root>

            <Checkbox.Root bind:checked={formData.isConfidential}>
              <div class="flex items-center space-x-2">
                <Checkbox.Indicator />
                <span class="text-sm">Mark as confidential</span>
              </div>
            </Checkbox.Root>
          </div>
        </div>

        <!-- Processing Priority -->
        <div class="mt-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Processing Priority
          </label>
          <Select.Root bind:selected={formData.analysisOptions.priority}>
            <Select.Trigger class="w-full md:w-48">
              <Select.Value />
            </Select.Trigger>
            <Select.Content>
              {#each priorities as priority}
                <Select.Item value={priority.value}>
                  {priority.label}
                </Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
        </div>
      </div>

      <!-- Retention Date -->
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Retention Date (Optional)
        </label>
        <Input.Root
          type="date"
          bind:value={formData.retentionDate}
          class="w-full md:w-48"
        />
        <p class="text-xs text-gray-500 mt-1">
          Date when this document should be reviewed for retention/deletion
        </p>
      </div>

      <!-- Form Actions -->
      <div class="flex items-center justify-between pt-6 border-t">
        <div class="text-sm text-gray-500">
          {#if $state.context.isDirty}
            Form has unsaved changes
          {:else if $state.context.submitCount > 0}
            Form submitted {$state.context.submitCount} time{$state.context.submitCount === 1 ? '' : 's'}
          {/if}
        </div>

        <div class="flex space-x-3">
          <Button.Root
            type="button"
            variant="outline"
            on:click={resetForm}
            disabled={$state.matches('submitting')}
          >
            Reset
          </Button.Root>

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
          </Button.Root>
        </div>
      </div>
    </form>
  </div>
</Card.Root>

<style>
  .document-metadata-form {
    @apply max-w-4xl mx-auto;
  }
</style>