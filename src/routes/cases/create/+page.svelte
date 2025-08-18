<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Label } from '$lib/components/ui/label/index.js';
  import { enhance } from '$app/forms';
  import { goto } from '$app/navigation';
  import type { ActionData } from './$types';

  interface Props {
    form?: ActionData;
  }

  let { form }: Props = $props();

  let isSubmitting = $state(false);
  let formData = $state({
    title: '',
    description: '',
    caseType: 'criminal',
    priority: 'medium',
    jurisdiction: '',
    assignedDetective: '',
    assignedProsecutor: '',
    estimatedDuration: '',
    budget: ''
  });

  const caseTypes = [
    { value: 'criminal', label: 'Criminal Case' },
    { value: 'civil', label: 'Civil Case' },
    { value: 'family', label: 'Family Law' },
    { value: 'corporate', label: 'Corporate Law' },
    { value: 'intellectual_property', label: 'Intellectual Property' },
    { value: 'environmental', label: 'Environmental Law' }
  ];

  const priorityLevels = [
    { value: 'low', label: 'Low Priority', color: '#68d391' },
    { value: 'medium', label: 'Medium Priority', color: '#ed8936' },
    { value: 'high', label: 'High Priority', color: '#f56565' },
    { value: 'urgent', label: 'Urgent', color: '#9f7aea' }
  ];

  const jurisdictions = [
    'Federal Court',
    'State Court - California',
    'State Court - New York',
    'State Court - Texas',
    'State Court - Florida',
    'Municipal Court',
    'International Court'
  ];

  function handleSubmit() {
    isSubmitting = true;
  }

  function goBack() {
    goto('/yorha-demo');
  }

  function generateCaseNumber() {
    const prefix = formData.caseType.substring(0, 3).toUpperCase();
    const year = new Date().getFullYear();
    const random = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
    return `${prefix}-${year}-${random}`;
  }
</script>

<svelte:head>
  <title>Create New Case - Legal AI System</title>
  <meta name="description" content="Create a new legal case in the AI case management system" />
</svelte:head>

<div class="case-create-container">
  <div class="case-background"></div>
  
  <div class="case-content">
    <!-- Header -->
    <div class="page-header">
      <Button variant="ghost" class="back-button" onclick={goBack}>
        ← Back to Demo Navigation
      </Button>
      
      <div class="header-info">
        <h1 class="page-title">Create New Legal Case</h1>
        <p class="page-subtitle">Initialize a new case in the AI-powered legal management system</p>
      </div>
    </div>

    <form method="POST" use:enhance={() => {
      handleSubmit();
      return async ({ result, update }) => {
        isSubmitting = false;
        if (result.type === 'success') {
          goto('/cases?created=true');
        }
        await update();
      };
    }}>
      <div class="form-layout">
        <!-- Case Overview -->
        <Card class="form-card primary">
          <CardHeader>
            <CardTitle class="card-title">Case Overview</CardTitle>
            <CardDescription>Basic information about the legal case</CardDescription>
          </CardHeader>
          <CardContent class="card-content">
            <div class="form-grid">
              <div class="input-group">
                <Label for="title" class="yorha-label">Case Title *</Label>
                <Input 
                  id="title" 
                  name="title" 
                  type="text" 
                  required 
                  class="yorha-input"
                  bind:value={formData.title}
                  placeholder="State vs. John Doe - Armed Robbery"
                />
                {#if form?.fieldErrors?.title}
                  <span class="error-message">{form.fieldErrors.title}</span>
                {/if}
              </div>

              <div class="input-group">
                <Label for="caseType" class="yorha-label">Case Type *</Label>
                <select 
                  id="caseType" 
                  name="caseType" 
                  class="yorha-select"
                  bind:value={formData.caseType}
                  required
                >
                  {#each caseTypes as type}
                    <option value={type.value}>{type.label}</option>
                  {/each}
                </select>
              </div>

              <div class="input-group">
                <Label for="priority" class="yorha-label">Priority Level *</Label>
                <select 
                  id="priority" 
                  name="priority" 
                  class="yorha-select"
                  bind:value={formData.priority}
                  required
                >
                  {#each priorityLevels as priority}
                    <option value={priority.value}>{priority.label}</option>
                  {/each}
                </select>
              </div>

              <div class="input-group">
                <Label for="jurisdiction" class="yorha-label">Jurisdiction *</Label>
                <select 
                  id="jurisdiction" 
                  name="jurisdiction" 
                  class="yorha-select"
                  bind:value={formData.jurisdiction}
                  required
                >
                  <option value="">Select Jurisdiction</option>
                  {#each jurisdictions as jurisdiction}
                    <option value={jurisdiction}>{jurisdiction}</option>
                  {/each}
                </select>
              </div>
            </div>
          </CardContent>
        </Card>

        <!-- Case Description -->
        <Card class="form-card secondary">
          <CardHeader>
            <CardTitle class="card-title">Case Description</CardTitle>
            <CardDescription>Detailed description of the case circumstances</CardDescription>
          </CardHeader>
          <CardContent class="card-content">
            <div class="input-group">
              <Label for="description" class="yorha-label">Description *</Label>
              <textarea 
                id="description" 
                name="description" 
                required 
                class="yorha-textarea"
                bind:value={formData.description}
                placeholder="Provide a detailed description of the case, including relevant facts, timeline, and key issues..."
                rows="6"
              ></textarea>
              {#if form?.fieldErrors?.description}
                <span class="error-message">{form.fieldErrors.description}</span>
              {/if}
            </div>
          </CardContent>
        </Card>

        <!-- Assignment & Resources -->
        <Card class="form-card accent">
          <CardHeader>
            <CardTitle class="card-title">Assignment & Resources</CardTitle>
            <CardDescription>Assign team members and allocate resources</CardDescription>
          </CardHeader>
          <CardContent class="card-content">
            <div class="form-grid">
              <div class="input-group">
                <Label for="assignedDetective" class="yorha-label">Assigned Detective</Label>
                <Input 
                  id="assignedDetective" 
                  name="assignedDetective" 
                  type="text" 
                  class="yorha-input"
                  bind:value={formData.assignedDetective}
                  placeholder="Detective Jane Smith"
                />
              </div>

              <div class="input-group">
                <Label for="assignedProsecutor" class="yorha-label">Assigned Prosecutor</Label>
                <Input 
                  id="assignedProsecutor" 
                  name="assignedProsecutor" 
                  type="text" 
                  class="yorha-input"
                  bind:value={formData.assignedProsecutor}
                  placeholder="ADA John Williams"
                />
              </div>

              <div class="input-group">
                <Label for="estimatedDuration" class="yorha-label">Estimated Duration (months)</Label>
                <Input 
                  id="estimatedDuration" 
                  name="estimatedDuration" 
                  type="number" 
                  class="yorha-input"
                  bind:value={formData.estimatedDuration}
                  placeholder="6"
                  min="1"
                  max="120"
                />
              </div>

              <div class="input-group">
                <Label for="budget" class="yorha-label">Estimated Budget ($)</Label>
                <Input 
                  id="budget" 
                  name="budget" 
                  type="number" 
                  class="yorha-input"
                  bind:value={formData.budget}
                  placeholder="50000"
                  min="0"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <!-- AI Recommendations -->
        <Card class="form-card info">
          <CardHeader>
            <CardTitle class="card-title">🤖 AI Recommendations</CardTitle>
            <CardDescription>AI-powered suggestions based on case details</CardDescription>
          </CardHeader>
          <CardContent class="card-content">
            <div class="ai-recommendations">
              <div class="recommendation-item">
                <span class="rec-icon">📋</span>
                <div class="rec-content">
                  <h4 class="rec-title">Suggested Case Number</h4>
                  <p class="rec-value">{generateCaseNumber()}</p>
                </div>
              </div>
              
              <div class="recommendation-item">
                <span class="rec-icon">⏱️</span>
                <div class="rec-content">
                  <h4 class="rec-title">Recommended Timeline</h4>
                  <p class="rec-value">Based on case type: 8-12 months</p>
                </div>
              </div>
              
              <div class="recommendation-item">
                <span class="rec-icon">👥</span>
                <div class="rec-content">
                  <h4 class="rec-title">Required Team Size</h4>
                  <p class="rec-value">1 Detective, 1 Prosecutor, 2 Paralegals</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <!-- Error Display -->
      {#if form?.message}
        <div class="alert {form.type === 'error' ? 'alert-error' : 'alert-success'}">
          {form.message}
        </div>
      {/if}

      <!-- Submit Actions -->
      <div class="form-actions">
        <Button 
          type="button" 
          variant="outline" 
          class="cancel-button"
          onclick={() => goto('/cases')}
        >
          Cancel
        </Button>
        
        <Button 
          type="submit" 
          class="create-button" 
          disabled={isSubmitting}
          size="lg"
        >
          {#if isSubmitting}
            <span class="loading-spinner"></span>
            Creating Case...
          {:else}
            Create Case
          {/if}
        </Button>
      </div>
    </form>
  </div>
</div>

<style>
  .case-create-container {
    min-height: 100vh;
    position: relative;
    font-family: 'Rajdhani', 'Roboto Mono', monospace;
  }

  .case-background {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    z-index: -2;
  }

  .case-background::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, rgba(214, 158, 46, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(68, 200, 245, 0.03) 0%, transparent 50%);
    z-index: -1;
  }

  .case-content {
    position: relative;
    z-index: 10;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  /* Header */
  .page-header {
    margin-bottom: 2rem;
  }

  .back-button {
    margin-bottom: 1rem;
    color: #a0aec0;
    border: 1px solid #4a5568;
    background: rgba(26, 32, 44, 0.8);
    backdrop-filter: blur(10px);
  }

  .back-button:hover {
    color: #d69e2e;
    border-color: #d69e2e;
  }

  .header-info {
    text-align: center;
    padding: 2rem 0;
  }

  .page-title {
    color: #d69e2e;
    font-size: 2.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(214, 158, 46, 0.3);
  }

  .page-subtitle {
    color: #a0aec0;
    font-size: 1.125rem;
    line-height: 1.6;
    margin: 0;
  }

  /* Form Layout */
  .form-layout {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
  }

  /* Form Cards */
  :global(.form-card) {
    background: rgba(45, 55, 72, 0.8);
    border: 1px solid rgba(214, 158, 46, 0.3);
    border-radius: 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
  }

  :global(.form-card:hover) {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  }

  .form-card.primary {
    border-left: 4px solid #4299e1;
  }

  .form-card.secondary {
    border-left: 4px solid #9f7aea;
  }

  .form-card.accent {
    border-left: 4px solid #ed8936;
  }

  .form-card.info {
    border-left: 4px solid #68d391;
  }

  :global(.card-title) {
    color: #d69e2e;
    font-size: 1.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .card-content {
    padding: 1.5rem;
  }

  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
  }

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  /* YoRHa Form Elements */
  :global(.yorha-label) {
    color: #e2e8f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.875rem;
  }

  :global(.yorha-input) {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 0.75rem 1rem;
    font-family: inherit;
    transition: all 0.3s ease;
  }

  :global(.yorha-input:focus) {
    border-color: #d69e2e;
    box-shadow: 0 0 0 2px rgba(214, 158, 46, 0.2);
  }

  :global(.yorha-input::placeholder) {
    color: #6b7280;
  }

  .yorha-select {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 0.75rem 1rem;
    font-family: inherit;
    transition: all 0.3s ease;
  }

  .yorha-select:focus {
    border-color: #d69e2e;
    box-shadow: 0 0 0 2px rgba(214, 158, 46, 0.2);
    outline: none;
  }

  .yorha-select option {
    background: #1a202c;
    color: #e2e8f0;
  }

  .yorha-textarea {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 0.75rem 1rem;
    font-family: inherit;
    transition: all 0.3s ease;
    resize: vertical;
    min-height: 120px;
  }

  .yorha-textarea:focus {
    border-color: #d69e2e;
    box-shadow: 0 0 0 2px rgba(214, 158, 46, 0.2);
    outline: none;
  }

  .yorha-textarea::placeholder {
    color: #6b7280;
  }

  /* AI Recommendations */
  .ai-recommendations {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .recommendation-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    background: rgba(104, 211, 145, 0.1);
    border: 1px solid rgba(104, 211, 145, 0.3);
    border-radius: 0;
  }

  .rec-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
  }

  .rec-content {
    flex: 1;
  }

  .rec-title {
    color: #68d391;
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 0.25rem 0;
  }

  .rec-value {
    color: #e2e8f0;
    font-size: 0.95rem;
    margin: 0;
  }

  /* Error Messages */
  .error-message {
    color: #f56565;
    font-size: 0.875rem;
    font-weight: 500;
  }

  /* Alerts */
  .alert {
    padding: 1rem;
    border-radius: 0;
    margin-bottom: 1rem;
    font-weight: 500;
  }

  .alert-error {
    background: rgba(245, 101, 101, 0.1);
    border: 1px solid #f56565;
    color: #f56565;
  }

  .alert-success {
    background: rgba(104, 211, 145, 0.1);
    border: 1px solid #68d391;
    color: #68d391;
  }

  /* Form Actions */
  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 2rem 0;
  }

  :global(.cancel-button) {
    background: transparent;
    border: 1px solid #4a5568;
    color: #a0aec0;
    padding: 1rem 2rem;
    border-radius: 0;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 1px;
  }

  :global(.cancel-button:hover) {
    border-color: #a0aec0;
    color: #e2e8f0;
  }

  :global(.create-button) {
    background: linear-gradient(135deg, #d69e2e 0%, #ed8936 100%);
    border: none;
    color: #1a202c;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 2rem;
    min-height: 3rem;
    border-radius: 0;
    transition: all 0.3s ease;
  }

  :global(.create-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(214, 158, 46, 0.3);
  }

  :global(.create-button:disabled) {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .loading-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #1a202c;
    border-radius: 50%;
    border-top-color: transparent;
    animation: spin 1s linear infinite;
    margin-right: 0.5rem;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .case-content {
      padding: 1rem;
    }

    .form-layout {
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .form-grid {
      grid-template-columns: 1fr;
    }

    .form-actions {
      flex-direction: column;
    }

    .page-title {
      font-size: 2rem;
    }
  }
</style>