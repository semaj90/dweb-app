<script lang="ts">
  import BitsUnoDemo from "$lib/components/ui/BitsUnoDemo.svelte";
  import { goto } from '$app/navigation';
</script>

<svelte:head>
  <title>Bits UI + UnoCSS Demo | Legal Case Management</title>
  <meta name="description" content="Demonstration of Bits UI and UnoCSS integration in the legal case management system." />
</svelte:head>

<!-- Navigation Header -->
<div class="space-y-4">
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <button 
          onclick={() => goto('/')}
          class="space-y-4"
        >
          ← Back to Dashboard
        </button>
      </div>
      <h1 class="space-y-4">UI Integration Demo</h1>
      <div class="space-y-4"></div> <!-- Spacer for centering -->
    </div>
  </div>
</div>

<!-- Demo Component -->
<BitsUnoDemo />

<!-- Additional Information -->
<div class="space-y-4">
  <div class="space-y-4">
    <div class="space-y-4">
      <h2 class="space-y-4">Technical Implementation</h2>
      
      <div class="space-y-4">
        <div>
          <h3 class="space-y-4">Installed Packages</h3>
          <div class="space-y-4">
            <div class="space-y-4">
              <div>bits-ui@2.8.10</div>
              <div>unocss@66.3.2</div>
              <div>@unocss/vite@66.3.2</div>
              <div>@unocss/preset-uno@66.3.2</div>
              <div>@unocss/preset-attributify@66.3.2</div>
              <div>@unocss/preset-icons@66.3.2</div>
              <div>@unocss/preset-typography@66.3.2</div>
              <div>@unocss/preset-web-fonts@66.3.2</div>
            </div>
          </div>
        </div>
        
        <div>
          <h3 class="space-y-4">Configuration Files</h3>
          <div class="space-y-4">
            <div class="space-y-4">
              <span class="space-y-4">✓</span>
              <code class="space-y-4">unocss.config.ts</code>
            </div>
            <div class="space-y-4">
              <span class="space-y-4">✓</span>
              <code class="space-y-4">vite.config.ts</code>
            </div>
            <div class="space-y-4">
              <span class="space-y-4">✓</span>
              <code class="space-y-4">+layout.svelte</code>
            </div>
            <div class="space-y-4">
              <span class="space-y-4">✓</span>
              <code class="space-y-4">package.json</code>
            </div>
          </div>
        </div>
      </div>
      
      <div class="space-y-4">
        <h4 class="space-y-4">Integration Status</h4>
        <p class="space-y-4">
          ✅ Both Bits UI and UnoCSS are successfully integrated and working together in both the web-app and desktop-app (which shares the same frontend).
          The desktop app uses the web-app's SvelteKit frontend as specified in the Tauri configuration.
        </p>
      </div>
    </div>
  </div>
</div>
