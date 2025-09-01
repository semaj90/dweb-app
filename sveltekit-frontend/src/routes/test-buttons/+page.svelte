<script lang="ts">
  import { onMount } from 'svelte';
  
  let clickCount = $state(0);
  let message = $state('');
  let isLoading = $state(false);
  
  function handleClick() {
    clickCount++;
    message = `Button clicked ${clickCount} times`;
  }
  
  async function handleAsyncClick() {
    isLoading = true;
    message = 'Loading...';
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    message = 'Async operation complete!';
    isLoading = false;
  }
  
  function handleNESClick(type: string) {
    message = `NES ${type} button clicked!`;
  }
  
  onMount(() => {
    console.log('✅ Test Buttons page mounted');
  });
</script>

<div class="container mx-auto p-6 max-w-4xl">
  <h1 class="text-3xl font-bold mb-6">Button Event Test Page</h1>
  
  <!-- Test Results Display -->
  {#if message}
    <div class="nes-container with-title is-centered mb-6">
      <p class="title">Test Results</p>
      <p class="text-lg">{message}</p>
      <p class="text-sm text-gray-500">Click count: {clickCount}</p>
    </div>
  {/if}
  
  <!-- Basic Button Tests -->
  <div class="nes-container with-title mb-6">
    <p class="title">Basic Button Tests</p>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <button 
        class="nes-btn"
        on:on:onclick={handleClick}
      >
        Basic Click
      </button>
      
      <button 
        class="nes-btn is-primary"
        on:on:onclick={() => handleNESClick('primary')}
      >
        Primary
      </button>
      
      <button 
        class="nes-btn is-success"
        on:on:onclick={() => handleNESClick('success')}
      >
        Success
      </button>
      
      <button 
        class="nes-btn is-warning"
        on:on:onclick={() => handleNESClick('warning')}
      >
        Warning
      </button>
      
      <button 
        class="nes-btn is-error"
        on:on:onclick={() => handleNESClick('error')}
      >
        Error
      </button>
      
      <button 
        class="nes-btn is-disabled"
        disabled
        on:on:onclick={() => handleNESClick('disabled')}
      >
        Disabled
      </button>
      
      <button 
        class="nes-btn"
        on:on:onclick={handleAsyncClick}
        disabled={isLoading}
      >
        {isLoading ? 'Loading...' : 'Async Click'}
      </button>
      
      <button 
        class="nes-btn"
        on:on:onclick={() => {
          message = 'Inline handler works!';
        }}
      >
        Inline Handler
      </button>
    </div>
  </div>
  
  <!-- YoRHa Style Buttons -->
  <div class="nes-container with-title mb-6">
    <p class="title">YoRHa Style Buttons</p>
    <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
      <button 
        class="yorha-button px-6 py-3 rounded"
        on:on:onclick={() => message = 'YoRHa Command Executed'}
      >
        <span class="uppercase tracking-widest">Execute</span>
      </button>
      
      <button 
        class="yorha-button-secondary px-6 py-3 rounded"
        on:on:onclick={() => message = 'YoRHa Analysis Complete'}
      >
        <span class="uppercase tracking-widest">Analyze</span>
      </button>
      
      <button 
        class="yorha-button-danger px-6 py-3 rounded"
        on:on:onclick={() => message = 'YoRHa Emergency Protocol'}
      >
        <span class="uppercase tracking-widest">Abort</span>
      </button>
    </div>
  </div>
  
  <!-- Mouse Event Tests -->
  <div class="nes-container with-title mb-6">
    <p class="title">Mouse Event Tests</p>
    <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
      <button 
        class="nes-btn"
        on:on:on:mouseenter={() => message = 'Mouse entered!'}
        on:on:on:mouseleave={() => message = 'Mouse left!'}
      >
        Hover Me
      </button>
      
      <button 
        class="nes-btn"
        on:dblclick={() => message = 'Double clicked!'}
      >
        Double Click
      </button>
      
      <button 
        class="nes-btn"
        on:contextmenu|preventDefault={() => message = 'Right clicked!'}
      >
        Right Click
      </button>
    </div>
  </div>
  
  <!-- Keyboard Event Tests -->
  <div class="nes-container with-title mb-6">
    <p class="title">Keyboard Event Tests</p>
    <input 
      type="text" 
      class="nes-input"
      placeholder="Type something..."
      onkeydown={(e) => {
        if (e.key === 'Enter') {
          message = `Enter pressed: ${e.currentTarget.value}`;
        }
      }}
      on:input={(e) => {
        message = `Typing: ${e.currentTarget.value}`;
      }}
    />
  </div>
  
  <!-- Form Event Tests -->
  <div class="nes-container with-title mb-6">
    <p class="title">Form Event Tests</p>
    <div class="space-y-4">
      <div>
        <label>
          <input 
            type="checkbox" 
            class="nes-checkbox"
            onchange={(e) => {
              message = `Checkbox ${e.currentTarget.checked ? 'checked' : 'unchecked'}`;
            }}
          />
          <span>Toggle Me</span>
        </label>
      </div>
      
      <div>
        <label>
          <input 
            type="radio" 
            class="nes-radio" 
            name="test"
            onchange={() => message = 'Radio 1 selected'}
          />
          <span>Option 1</span>
        </label>
        <label>
          <input 
            type="radio" 
            class="nes-radio" 
            name="test"
            onchange={() => message = 'Radio 2 selected'}
          />
          <span>Option 2</span>
        </label>
      </div>
      
      <div class="nes-select">
        <select onchange={(e) => message = `Selected: ${e.currentTarget.value}`}>
          <option value="">Select an option</option>
          <option value="option1">Option 1</option>
          <option value="option2">Option 2</option>
          <option value="option3">Option 3</option>
        </select>
      </div>
    </div>
  </div>
  
  <!-- GPU-Accelerated Buttons -->
  <div class="nes-container with-title">
    <p class="title">GPU-Accelerated Effects</p>
    <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
      <button 
        class="nes-btn gpu-accelerated glow-effect"
        on:on:onclick={() => message = 'GPU Glow effect!'}
      >
        Glow Effect
      </button>
      
      <button 
        class="nes-btn gpu-accelerated pulse-effect"
        on:on:onclick={() => message = 'GPU Pulse effect!'}
      >
        Pulse Effect
      </button>
      
      <button 
        class="nes-btn gpu-accelerated scan-effect"
        on:on:onclick={() => message = 'GPU Scan effect!'}
      >
        Scan Lines
      </button>
    </div>
  </div>
</div>

<style>
  /* YoRHa Button Styles */
  .yorha-button {
    background: linear-gradient(135deg, #4a4a4a 0%, #6b6b6b 100%);
    color: #ffd700;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    position: relative;
    overflow: hidden;
    transform: translateZ(0);
    will-change: transform;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  .yorha-button:hover {
    transform: translateZ(0) scale(1.05);
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
  }
  
  .yorha-button-secondary {
    background: linear-gradient(135deg, #2a2a2a 0%, #4a4a4a 100%);
    color: #92cc41;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    transform: translateZ(0);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  .yorha-button-danger {
    background: linear-gradient(135deg, #6b2020 0%, #8b3030 100%);
    color: #ff4081;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    transform: translateZ(0);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  /* GPU Effects */
  .gpu-accelerated {
    transform: translateZ(0);
    will-change: transform, opacity;
    backface-visibility: hidden;
    -webkit-font-smoothing: antialiased;
  }
  
  .glow-effect {
    animation: glow 2s ease-in-out infinite;
  }
  
  @keyframes glow {
    0%, 100% {
      box-shadow: 0 0 5px rgba(146, 204, 65, 0.5),
                  0 0 10px rgba(146, 204, 65, 0.3),
                  0 0 15px rgba(146, 204, 65, 0.1);
    }
    50% {
      box-shadow: 0 0 20px rgba(146, 204, 65, 0.8),
                  0 0 30px rgba(146, 204, 65, 0.6),
                  0 0 40px rgba(146, 204, 65, 0.4);
    }
  }
  
  .pulse-effect {
    animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
  
  @keyframes pulse {
    0%, 100% {
      transform: translateZ(0) scale(1);
    }
    50% {
      transform: translateZ(0) scale(1.05);
    }
  }
  
  .scan-effect {
    position: relative;
    overflow: hidden;
  }
  
  .scan-effect::after {
    content: '';
    position: absolute;
    top: -100%;
    left: 0;
    right: 0;
    height: 100%;
    background: linear-gradient(
      180deg,
      transparent 0%,
      rgba(146, 204, 65, 0.1) 50%,
      transparent 100%
    );
    animation: scan 3s linear infinite;
  }
  
  @keyframes scan {
    0% {
      top: -100%;
    }
    100% {
      top: 100%;
    }
  }
</style>
