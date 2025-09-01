<!--
  Simple Case Creation Test Page - Testing bits-ui buttons and database integration
-->
<script lang="ts">
  import Button from '$lib/components/ui/Button.svelte';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Textarea } from '$lib/components/ui/textarea';
  import * as Card from '$lib/components/ui/card';
  import { 
    AlertCircle, 
    Loader2, 
    Save, 
    CheckCircle,
    FileText,
    Scale
  } from 'lucide-svelte';
let isSubmitting = $state(false);
let submitResult = $state('');
let formData = $state({
    caseNumber: '',
    title: '',
    description: '',
    priority: 'medium'
  });

  async function handleSubmit(event: Event) {
    event.preventDefault();
    isSubmitting = true;
    submitResult = '';

    try {
      const response = await fetch('/api/test-case', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      const result = await response.json();
      
      if (response.ok) {
        submitResult = `✅ Success: Case created with ID ${result.id || 'unknown'}`;
        // Reset form
        formData = {
          caseNumber: '',
          title: '',
          description: '',
          priority: 'medium'
        };
      } else {
        submitResult = `❌ Error: ${result.error || 'Unknown error'}`;
      }
    } catch (error) {
      submitResult = `❌ Network Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    } finally {
      isSubmitting = false;
    }
  }
</script>

<svelte:head>
  <title>Test Case Creation - Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto px-4 py-8 max-w-4xl">
  <!-- Header -->
  <div class="mb-8">
    <div class="flex items-center space-x-3">
      <Scale class="h-6 w-6 text-primary" />
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Test Case Creation</h1>
        <p class="text-muted-foreground mt-1">
          Testing bits-ui buttons, database save, and API integration
        </p>
      </div>
    </div>
  </div>

  <!-- Result Display -->
  {#if submitResult}
    <div class="mb-6 p-4 rounded-lg border {submitResult.includes('✅') ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}">
      <p class="font-medium">{submitResult}</p>
    </div>
  {/if}

  <Card.Root>
    <Card.Header>
      <Card.Title class="flex items-center space-x-2">
        <FileText class="h-5 w-5" />
        <span>Case Information</span>
      </Card.Title>
      <Card.Description>
        Fill out the basic case information to test the system integration
      </Card.Description>
    </Card.Header>

    <Card.Content>
      <form onsubmit={handleSubmit} class="space-y-6">
        <!-- Case Number -->
        <div class="space-y-2">
          <Label for="caseNumber">
            <span class="flex items-center space-x-2">
              <FileText class="h-4 w-4" />
              <span>Case Number *</span>
            </span>
          </Label>
          <Input
            id="caseNumber"
            name="caseNumber"
            placeholder="ABC-2024-123456"
            bind:value={formData.caseNumber}
            required
          />
        </div>

        <!-- Title -->
        <div class="space-y-2">
          <Label for="title">Case Title *</Label>
          <Input
            id="title"
            name="title"
            placeholder="Enter a descriptive case title"
            bind:value={formData.title}
            required
          />
        </div>

        <!-- Description -->
        <div class="space-y-2">
          <Label for="description">Description</Label>
          <Textarea
            id="description"
            name="description"
            placeholder="Provide detailed case description"
            bind:value={formData.description}
            rows="4"
          />
        </div>

        <!-- Priority -->
        <div class="space-y-2">
          <Label for="priority">Priority Level</Label>
          <select 
            id="priority"
            bind:value={formData.priority}
            class="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          >
            <option value="low">Low Priority</option>
            <option value="medium">Medium Priority</option>
            <option value="high">High Priority</option>
          </select>
        </div>

        <!-- Submit Button (bits-ui Button) -->
        <div class="flex justify-end space-x-3 pt-6 border-t">
          <Button 
            type="button" 
            variant="outline"
            on:on:click={() => {
              formData = {
                caseNumber: '',
                title: '',
                description: '',
                priority: 'medium'
              };
              submitResult = '';
            }}
          >
            Clear Form
          </Button>
          
          <Button 
            type="submit" 
            disabled={isSubmitting || !formData.caseNumber || !formData.title}
            class="min-w-[120px]"
          >
            {#if isSubmitting}
              <Loader2 class="mr-2 h-4 w-4 animate-spin" />
              Creating...
            {:else}
              <Save class="mr-2 h-4 w-4" />
              Create Case
            {/if}
          </Button>
        </div>
      </form>
    </Card.Content>
  </Card.Root>

  <!-- API Test Info -->
  <div class="mt-8 p-6 bg-muted/30 rounded-lg border">
    <h3 class="font-semibold mb-3 flex items-center space-x-2">
      <AlertCircle class="h-5 w-5 text-primary" />
      <span>Integration Test Features</span>
    </h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-muted-foreground">
      <div>
        <h4 class="font-medium text-foreground mb-2">bits-ui Button Integration</h4>
        <ul class="space-y-1">
          <li>• Button component from bits-ui library</li>
          <li>• Loading states and animations</li>
          <li>• Disabled state handling</li>
          <li>• Icon integration with Lucide</li>
        </ul>
      </div>
      
      <div>
        <h4 class="font-medium text-foreground mb-2">Database & API Testing</h4>
        <ul class="space-y-1">
          <li>• POST request to /api/test-case</li>
          <li>• Database save operation test</li>
          <li>• Error handling and validation</li>
          <li>• Success/failure feedback</li>
        </ul>
      </div>
    </div>
  </div>
</div>