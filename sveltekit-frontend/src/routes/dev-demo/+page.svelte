<script lang="ts">
  import { onMount } from 'svelte';
  import { superForm } from 'sveltekit-superforms';
  import { zod } from 'sveltekit-superforms/adapters';
  import { z } from 'zod';
  import Button from '$lib/components/ui/enhanced/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/enhanced/Card.svelte';
  import Input from '$lib/components/ui/enhanced/Input.svelte';
  import { Badge } from '$lib/components/ui/modular/Badge.svelte';
  import { cn } from '$lib/utils';

  // Svelte 5 runes for state management
  let authStatus = $state<any>(null);
  let cases = $state<any[]>([]);
  let apiStatus = $state<Record<string, any>>({});
  let chatMessages = $state<any[]>([]);
  let currentMessage = $state('');
  let isStreaming = $state(false);

  // Zod schema for case creation (matching API schema)
  const caseSchema = z.object({
    title: z.string().min(1, "Case title is required").max(500, "Case title too long"),
    description: z.string().optional(),
    priority: z.enum(["low", "medium", "high", "critical"]).default("medium"),
    status: z.enum(["open", "investigating", "pending", "closed", "archived"]).default("open"),
    location: z.string().optional(),
    jurisdiction: z.string().optional()
  });

  // Superforms setup
  const { form, errors, constraints, enhance } = superForm({
    title: '',
    description: '',
    priority: 'medium',
    status: 'open',
    location: '',
    jurisdiction: ''
  }, {
    SPA: true,
    validators: zod(caseSchema),
    onUpdate: async ({ form }) => {
      if (form.valid) {
        // Handle form submission
        await createCase(form.data);
      }
    }
  });

  async function checkSystemStatus() {
    try {
      // Check authentication status
      const authResponse = await fetch('/api/auth/debug');
      authStatus = await authResponse.json();

      // Check API endpoints
      const endpoints = [
        { name: 'Cases API', url: '/api/cases', method: 'GET' },
        { name: 'Enhanced RAG', url: 'http://localhost:8094/health', method: 'GET' },
        { name: 'Upload Service', url: 'http://localhost:8093/health', method: 'GET' },
        { name: 'Ollama', url: 'http://localhost:11434/api/tags', method: 'GET' }
      ];

      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint.url);
          const data = await response.json();
          apiStatus[endpoint.name] = {
            status: response.ok ? 'healthy' : 'error',
            data: response.ok ? data : null,
            error: response.ok ? null : data
          };
        } catch (error) {
          apiStatus[endpoint.name] = {
            status: 'unreachable',
            error: error instanceof Error ? error.message : 'Unknown error'
          };
        }
      }

      // Trigger reactivity
      apiStatus = { ...apiStatus };

      // Load cases
      await loadCases();
    } catch (error) {
      console.error('System status check failed:', error);
    }
  }

  async function loadCases() {
    try {
      const response = await fetch('/api/cases');
      const result = await response.json();
      if (result.success) {
        cases = result.data.cases || [];
      } else {
        console.error('Failed to load cases:', result.error);
      }
    } catch (error) {
      console.error('Error loading cases:', error);
    }
  }

  async function createCase(caseData: any) {
    try {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(caseData)
      });

      const result = await response.json();
      
      if (result.success) {
        console.log('Case created successfully:', result.data);
        await loadCases(); // Refresh case list
        // Reset form
        $form = {
          title: '',
          description: '',
          priority: 'medium',
          status: 'open',
          location: '',
          jurisdiction: ''
        };
      } else {
        console.error('Failed to create case:', result.error);
      }
    } catch (error) {
      console.error('Error creating case:', error);
    }
  }

  async function sendChatMessage() {
    if (!currentMessage.trim() || isStreaming) return;

    const userMessage = {
      role: 'user',
      content: currentMessage,
      timestamp: new Date()
    };

    chatMessages = [...chatMessages, userMessage];
    const messageToSend = currentMessage;
    currentMessage = '';
    isStreaming = true;

    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: messageToSend,
          model: 'gemma3-legal:latest',
          stream: false
        })
      });

      if (response.ok) {
        const result = await response.json();
        const aiMessage = {
          role: 'assistant',
          content: result.response || 'No response received',
          timestamp: new Date()
        };
        chatMessages = [...chatMessages, aiMessage];
      } else {
        const errorMessage = {
          role: 'system',
          content: 'Error: AI service unavailable',
          timestamp: new Date()
        };
        chatMessages = [...chatMessages, errorMessage];
      }
    } catch (error) {
      const errorMessage = {
        role: 'system',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      };
      chatMessages = [...chatMessages, errorMessage];
    } finally {
      isStreaming = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  }

  onMount(() => {
    checkSystemStatus();
  });
</script>

<div class="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent mb-2">
        Legal AI Platform Development Demo
      </h1>
      <p class="text-gray-600">Superforms + Zod + SvelteKit 2 + Go Microservices + Database Integration</p>
    </div>

    <!-- System Status Overview -->
    <Card class="mb-6 p-6">
      {#snippet children()}
        <CardHeader>
          {#snippet children()}
            <CardTitle>System Status</CardTitle>
          {/snippet}
        </CardHeader>
        <CardContent>
          {#snippet children()}
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <!-- Authentication Status -->
              <div class="p-4 bg-white rounded-lg border">
                <h3 class="font-semibold mb-2">Authentication</h3>
                {#if authStatus}
                  <Badge variant={authStatus.hasUser ? 'success' : 'secondary'}>
                    {#snippet children()}
                      {authStatus.hasUser ? 'Authenticated' : 'Development Mode'}
                    {/snippet}
                  </Badge>
                  <p class="text-sm text-gray-600 mt-1">
                    {authStatus.hint}
                  </p>
                {:else}
                  <Badge variant="secondary">
                    {#snippet children()}Loading...{/snippet}
                  </Badge>
                {/if}
              </div>

              <!-- API Status -->
              {#each Object.entries(apiStatus) as [name, status]}
                <div class="p-4 bg-white rounded-lg border">
                  <h3 class="font-semibold mb-2">{name}</h3>
                  <Badge variant={status.status === 'healthy' ? 'success' : status.status === 'error' ? 'destructive' : 'secondary'}>
                    {#snippet children()}
                      {status.status.toUpperCase()}
                    {/snippet}
                  </Badge>
                  {#if status.error}
                    <p class="text-xs text-red-600 mt-1">{status.error}</p>
                  {/if}
                </div>
              {/each}
            </div>
          {/snippet}
        </CardContent>
      {/snippet}
    </Card>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Case Management with Superforms -->
      <Card class="p-6">
        {#snippet children()}
          <CardHeader class="mb-4">
            {#snippet children()}
              <CardTitle>Create New Case (Superforms + Zod)</CardTitle>
            {/snippet}
          </CardHeader>
          <CardContent>
            {#snippet children()}
              <form method="POST" use:enhance class="space-y-4">
                <div>
                  <label for="title" class="block text-sm font-medium mb-1">Case Title *</label>
                  <Input
                    type="text"
                    id="title"
                    name="title"
                    bind:value={$form.title}
                    class={cn("w-full", $errors.title && "border-red-500")}
                    {...$constraints.title}
                  />
                  {#if $errors.title}
                    <p class="text-sm text-red-600 mt-1">{$errors.title}</p>
                  {/if}
                </div>

                <div>
                  <label for="description" class="block text-sm font-medium mb-1">Description</label>
                  <textarea
                    id="description"
                    name="description"
                    bind:value={$form.description}
                    rows="3"
                    class={cn("w-full px-3 py-2 border border-gray-300 rounded-md", $errors.description && "border-red-500")}
                    {...$constraints.description}
                  ></textarea>
                  {#if $errors.description}
                    <p class="text-sm text-red-600 mt-1">{$errors.description}</p>
                  {/if}
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label for="priority" class="block text-sm font-medium mb-1">Priority</label>
                    <select
                      id="priority"
                      name="priority"
                      bind:value={$form.priority}
                      class="w-full px-3 py-2 border border-gray-300 rounded-md"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                      <option value="critical">Critical</option>
                    </select>
                  </div>

                  <div>
                    <label for="status" class="block text-sm font-medium mb-1">Status</label>
                    <select
                      id="status"
                      name="status"
                      bind:value={$form.status}
                      class="w-full px-3 py-2 border border-gray-300 rounded-md"
                    >
                      <option value="open">Open</option>
                      <option value="investigating">Investigating</option>
                      <option value="pending">Pending</option>
                      <option value="closed">Closed</option>
                    </select>
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label for="location" class="block text-sm font-medium mb-1">Location</label>
                    <Input
                      type="text"
                      id="location"
                      name="location"
                      bind:value={$form.location}
                      class="w-full"
                    />
                  </div>

                  <div>
                    <label for="jurisdiction" class="block text-sm font-medium mb-1">Jurisdiction</label>
                    <Input
                      type="text"
                      id="jurisdiction"
                      name="jurisdiction"
                      bind:value={$form.jurisdiction}
                      class="w-full"
                    />
                  </div>
                </div>

                <Button type="submit" class="w-full">
                  {#snippet children()}Create Case{/snippet}
                </Button>
              </form>
            {/snippet}
          </CardContent}
        {/snippet}
      </Card>

      <!-- AI Chat Interface -->
      <Card class="p-6">
        {#snippet children()}
          <CardHeader class="mb-4">
            {#snippet children()}
              <CardTitle>Legal AI Assistant</CardTitle>
            {/snippet}
          </CardHeader>
          <CardContent>
            {#snippet children()}
              <div class="space-y-4">
                <!-- Chat Messages -->
                <div class="h-80 overflow-y-auto border rounded-lg p-4 bg-gray-50">
                  {#if chatMessages.length === 0}
                    <p class="text-gray-500 text-center">Start a conversation with the Legal AI...</p>
                  {:else}
                    {#each chatMessages as message}
                      <div class={cn("mb-3 p-2 rounded", 
                        message.role === 'user' ? 'bg-blue-100 ml-8' : 
                        message.role === 'assistant' ? 'bg-green-100 mr-8' : 
                        'bg-red-100 mr-8'
                      )}>
                        <div class="flex items-start gap-2">
                          <Badge variant={message.role === 'user' ? 'default' : message.role === 'assistant' ? 'success' : 'destructive'}>
                            {#snippet children()}
                              {message.role.toUpperCase()}
                            {/snippet}
                          </Badge>
                          <div class="flex-1">
                            <p class="text-sm">{message.content}</p>
                            <p class="text-xs text-gray-500 mt-1">
                              {message.timestamp.toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    {/each}
                  {/if}
                </div>

                <!-- Chat Input -->
                <div class="flex gap-2">
                  <Input
                    type="text"
                    bind:value={currentMessage}
                    onkeydown={handleKeydown}
                    placeholder="Ask a legal question..."
                    disabled={isStreaming}
                    class="flex-1"
                  />
                  <Button onclick={sendChatMessage} disabled={isStreaming || !currentMessage.trim()}>
                    {#snippet children()}
                      {isStreaming ? 'Sending...' : 'Send'}
                    {/snippet}
                  </Button>
                </div>
              </div>
            {/snippet}
          </CardContent>
        {/snippet}
      </Card>
    </div>

    <!-- Cases List -->
    <Card class="mt-6 p-6">
      {#snippet children()}
        <CardHeader class="mb-4">
          {#snippet children()}
            <CardTitle>Recent Cases</CardTitle>
          {/snippet}
        </CardHeader>
        <CardContent>
          {#snippet children()}
            {#if cases.length === 0}
              <p class="text-gray-500 text-center py-8">No cases found. Create your first case above!</p>
            {:else}
              <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {#each cases as case}
                  <div class="border rounded-lg p-4 bg-white">
                    <h3 class="font-semibold mb-2">{case.title}</h3>
                    <p class="text-sm text-gray-600 mb-2">{case.description || 'No description'}</p>
                    <div class="flex items-center gap-2 mb-2">
                      <Badge variant={case.priority === 'critical' ? 'destructive' : case.priority === 'high' ? 'default' : 'secondary'}>
                        {#snippet children()}
                          {case.priority.toUpperCase()}
                        {/snippet}
                      </Badge>
                      <Badge variant="outline">
                        {#snippet children()}
                          {case.status.toUpperCase()}
                        {/snippet}
                      </Badge>
                    </div>
                    <p class="text-xs text-gray-500">
                      Case #{case.caseNumber || case.id?.substring(0, 8)}
                    </p>
                  </div>
                {/each}
              </div>
            {/if}
          {/snippet}
        </CardContent>
      {/snippet}
    </Card>
  </div>
</div>