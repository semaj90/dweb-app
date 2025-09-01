<script lang="ts">
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
let cookiesEnabled = $state(false);
let localStorageEnabled = $state(false);
let sessionStorageEnabled = $state(false);
let devMode = $state(false);
let sessionInfo = $state('');
  
  onMount(() => {
    if (!browser) return;
    
    devMode = window.location.hostname === 'localhost' || 
              window.location.hostname === '127.0.0.1';
    
    // Test cookie support
    try {
      document.cookie = 'test=1; SameSite=Strict';
      cookiesEnabled = document.cookie.includes('test=1');
      document.cookie = 'test=; expires=Thu, 01 Jan 1970 00:00:00 GMT';
    } catch (e) {
      cookiesEnabled = false;
    }
    
    // Test localStorage support
    try {
      localStorage.setItem('test', '1');
      localStorageEnabled = localStorage.getItem('test') === '1';
      localStorage.removeItem('test');
    } catch (e) {
      localStorageEnabled = false;
    }
    
    // Test sessionStorage support
    try {
      sessionStorage.setItem('test', '1');
      sessionStorageEnabled = sessionStorage.getItem('test') === '1';
      sessionStorage.removeItem('test');
    } catch (e) {
      sessionStorageEnabled = false;
    }
    
    // Check for existing session
    const sessionCookie = document.cookie
      .split('; ')
      .find(row => row.startsWith('session_id='));
    
    if (sessionCookie) {
      sessionInfo = `Active session: ${sessionCookie.split('=')[1].substring(0, 8)}...`;
    } else if (localStorageEnabled) {
      const devSession = localStorage.getItem('dev_session') || 
                        (() => {
                          const id = `dev_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                          localStorage.setItem('dev_session', id);
                          return id;
                        })();
      sessionInfo = `Dev session: ${devSession.substring(0, 16)}...`;
    } else {
      sessionInfo = 'No session available';
    }
  });
</script>

{#if devMode}
  <div class="fixed bottom-4 right-4 bg-gray-900 text-white text-xs p-3 rounded-lg shadow-lg max-w-xs z-50">
    <div class="font-bold mb-2 flex items-center">
      🧪 Development Session Status
    </div>
    
    <div class="space-y-1">
      <div class="flex justify-between">
        <span>Cookies:</span>
        <span class={cookiesEnabled ? 'text-green-400' : 'text-red-400'}>
          {cookiesEnabled ? '✓' : '✗'}
        </span>
      </div>
      
      <div class="flex justify-between">
        <span>localStorage:</span>
        <span class={localStorageEnabled ? 'text-green-400' : 'text-red-400'}>
          {localStorageEnabled ? '✓' : '✗'}
        </span>
      </div>
      
      <div class="flex justify-between">
        <span>sessionStorage:</span>
        <span class={sessionStorageEnabled ? 'text-green-400' : 'text-red-400'}>
          {sessionStorageEnabled ? '✓' : '✗'}
        </span>
      </div>
      
      <hr class="border-gray-600 my-2">
      
      <div class="text-xs text-gray-300">
        {sessionInfo}
      </div>
      
      {#if !cookiesEnabled && (localStorageEnabled || sessionStorageEnabled)}
        <div class="text-yellow-400 text-xs mt-2">
          ⚠️ Using storage fallback
        </div>
      {/if}
      
      {#if !cookiesEnabled && !localStorageEnabled && !sessionStorageEnabled}
        <div class="text-red-400 text-xs mt-2">
          ❌ No session storage available
        </div>
      {/if}
    </div>
  </div>
{/if}