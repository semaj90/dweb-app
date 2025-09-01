<!-- Updated AI Chat for GPU Ollama -->
<script lang="ts">
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Input } from '$lib/components/ui/input';
  
  let messages = $state<{role: string, content: string}[]>([]);
  let input = $state('');
  let isLoading = $state(false);

  async function sendMessage() {
    if (!input.trim()) return;
    
    messages.push({ role: 'user', content: input });
    const userMessage = input;
    input = '';
    isLoading = true;

    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: userMessage,
          stream: false,
          options: {
            temperature: 0.3,
            num_ctx: 2048
          }
        })
      });
      
      if (!response.ok) throw new Error('AI service unavailable');
      
      const data = await response.json();
      messages.push({ role: 'assistant', content: data.response });
    } catch (error) {
      messages.push({ role: 'error', content: 'AI service error - check GPU setup' });
    } finally {
      isLoading = false;
    }
  }
</script>

<Card class="h-96 flex flex-col">
  <CardHeader>
    <CardTitle>Legal AI Assistant (GPU)</CardTitle>
  </CardHeader>
  <CardContent class="flex-1 flex flex-col space-y-4">
    <div class="flex-1 overflow-y-auto space-y-2 p-2 border rounded">
      {#each messages as message}
        <div class="p-2 rounded {message.role === 'user' ? 'bg-blue-100 ml-8' : message.role === 'error' ? 'bg-red-100' : 'bg-gray-100 mr-8'}">
          <strong>{message.role}:</strong> {message.content}
        </div>
      {/each}
      {#if isLoading}
        <div class="text-center p-2">⚡ GPU AI processing...</div>
      {/if}
    </div>
    <div class="flex space-x-2">
      <Input bind:value={input} placeholder="Legal question..." keydown={(e) => e.key === 'Enter' && sendMessage()} />
      <Button on:on:click={sendMessage} disabled={isLoading}>Send</Button>
    </div>
  </CardContent>
</Card>


