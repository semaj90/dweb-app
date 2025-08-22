<script lang="ts">
  import { $state } from 'svelte';

  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Input } from '$lib/components/ui/input';
  
  let prompt = $state('');
  let response = $state('');
  let isLoading = $state(false);

  async function queryAI() {
    if (!prompt.trim()) return;
    
    isLoading = true;
    try {
      const res = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal:latest',
          prompt,
          stream: false,
          options: {
            temperature: 0.7,
            max_tokens: 1024
          }
        })
      });
      const data = await res.json();
      response = data.response;
    } catch (error) {
      response = 'Error: Could not connect to AI service';
    } finally {
      isLoading = false;
    }
  }
</script>

<Card class="w-full max-w-2xl">
  <CardHeader>
    <CardTitle>Legal AI Assistant</CardTitle>
  </CardHeader>
  <CardContent class="space-y-4">
    <Textarea 
      bind:value={prompt}
      placeholder="Ask a legal question..."
      rows={3}
    />
    <Button onclick={queryAI} disabled={isLoading}>
      {isLoading ? 'Thinking...' : 'Ask AI'}
    </Button>
    {#if response}
      <div class="p-4 bg-muted rounded-lg">
        <p class="whitespace-pre-wrap">{response}</p>
      </div>
    {/if}
  </CardContent>
</Card>


