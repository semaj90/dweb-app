<script lang="ts">
  interface Props {
    user?: any;
    isDarkMode?: boolean;
  }
  
  let { user = {}, isDarkMode = $bindable(false) } = $props();
  
  let messages: any[] = [];
  let currentMessage = '';
  
  function sendMessage() {
    if (!currentMessage.trim()) return;
    
    messages = [...messages, {
      id: Date.now(),
      text: currentMessage,
      sender: 'user',
      timestamp: new Date()
    }];
    
    // Mock AI response
    setTimeout(() => {
      messages = [...messages, {
        id: Date.now() + 1,
        text: "I understand your request. Let me analyze that for you.",
        sender: 'ai',
        timestamp: new Date()
      }];
    }, 1000);
    
    currentMessage = '';
  }
</script>

<div class="nier-ai-assistant p-6 bg-black text-green-400 font-mono rounded-lg">
  <div class="mb-4">
    <h3 class="text-xl font-bold">NieR AI Assistant</h3>
    <p class="text-sm opacity-75">Androids are prohibited from removing their visors</p>
  </div>
  
  <div class="messages space-y-3 max-h-60 overflow-y-auto mb-4">
    {#each messages as message}
      <div class="message" class:text-yellow-400={message.sender === 'ai'}>
        <span class="font-bold">[{message.sender.toUpperCase()}]</span>
        {message.text}
      </div>
    {/each}
  </div>
  
  <div class="flex gap-2">
    <input
      bind:value={currentMessage}
      onkeydown={(e) => e.key === 'Enter' && sendMessage()}
      placeholder="Enter command..."
      class="flex-1 bg-gray-900 border border-green-400 text-green-400 px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-green-400"
    />
    <button
      onclick={sendMessage}
      class="bg-green-400 text-black px-4 py-2 rounded hover:bg-green-300 transition-colors"
    >
      SEND
    </button>
  </div>
</div>

<style>
  .nier-ai-assistant {
    background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
    border: 1px solid #00ff00;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
  }
  
  .messages {
    scrollbar-width: thin
    scrollbar-color: #00ff00 #000000;
  }
  
  .messages::-webkit-scrollbar {
    width: 8px;
  }
  
  .messages::-webkit-scrollbar-track {
    background: #000000;
  }
  
  .messages::-webkit-scrollbar-thumb {
    background-color: #00ff00;
    border-radius: 4px;
  }
</style>