<script lang="ts">
  import GoldenRatioGrid from '$lib/components/ui/layout/GoldenRatioGrid.svelte';
  import { Button } from '$lib/components/ui/button';
  import { Card } from '$lib/components/ui/enhanced-bits';

  let currentVariant = $state<'sidebar' | 'dashboard' | 'split' | 'content' | 'legal-document'>('dashboard');
  let currentDirection = $state<'horizontal' | 'vertical' | 'both'>('horizontal');
  let showDebugGrid = $state(false);

  const variants = ['sidebar', 'dashboard', 'split', 'content', 'legal-document'] as const;
  const directions = ['horizontal', 'vertical', 'both'] as const;

  function toggleDebugGrid() {
    showDebugGrid = !showDebugGrid;
    if (showDebugGrid) {
      document.body.classList.add('debug-golden-grid');
    } else {
      document.body.classList.remove('debug-golden-grid');
    }
  }
</script>

<svelte:head>
  <title>Golden Ratio Grid System Demo - Legal AI</title>
  <meta name="description" content="Interactive demonstration of the golden ratio CSS grid system for legal AI applications" />
</svelte:head>

<div class="golden-container">
  <div class="mb-phi-lg">
    <h1 class="text-phi-3xl font-bold text-nier-golden-primary mb-phi-md">
      Golden Ratio Grid System Demo
    </h1>
    <p class="text-phi-lg text-nier-golden-secondary mb-phi-lg leading-phi">
      Interactive demonstration of the mathematical golden ratio (φ ≈ 1.618) applied to CSS Grid layouts for legal AI applications.
    </p>

    <!-- Controls -->
    <div class="golden-section mb-phi-xl">
      <div class="golden-card">
        <h3 class="text-phi-lg font-semibold mb-phi-md">Grid Variant</h3>
        <div class="flex gap-phi-sm flex-wrap">
          {#each variants as variant}
            <Button
              variant={currentVariant === variant ? 'primary' : 'outline'}
              size="sm"
              onclick={() => currentVariant = variant}
              class="capitalize"
            >
              {variant.replace('-', ' ')}
            </Button>
          {/each}
        </div>
      </div>

      <div class="golden-card">
        <h3 class="text-phi-lg font-semibold mb-phi-md">Golden Direction</h3>
        <div class="flex gap-phi-sm flex-wrap">
          {#each directions as direction}
            <Button
              variant={currentDirection === direction ? 'primary' : 'outline'}
              size="sm"
              onclick={() => currentDirection = direction}
              class="capitalize"
            >
              {direction}
            </Button>
          {/each}
        </div>
      </div>
    </div>

    <div class="flex gap-phi-md mb-phi-lg">
      <Button
        variant="outline"
        onclick={toggleDebugGrid}
        class="text-phi-sm"
      >
        {showDebugGrid ? 'Hide' : 'Show'} Debug Grid
      </Button>
    </div>
  </div>

  <!-- Golden Ratio Grid Demo -->
  <div class="border-2 border-nier-golden-accent rounded-phi-md overflow-hidden" style="height: 600px;">
    <GoldenRatioGrid 
      variant={currentVariant}
      direction={currentDirection}
      size="full"
      legal={true}
      evidenceLayout={currentVariant === 'legal-document'}
      caseLayout={currentVariant === 'dashboard'}
      aiPanels={true}
      responsive={true}
      gap="md"
      class="h-full"
    >
      {#snippet header()}
        <div class="w-full bg-nier-golden-primary text-white p-phi-md">
          <h2 class="text-phi-lg font-semibold">
            Legal AI Header (φ⁻¹ × 5rem ≈ {(0.618 * 5).toFixed(2)}rem height)
          </h2>
          <p class="text-phi-sm opacity-80">
            Golden ratio proportioned header section
          </p>
        </div>
      {/snippet}

      {#snippet sidebar()}
        <div class="h-full bg-nier-golden-secondary p-phi-md overflow-y-auto">
          <h3 class="text-phi-lg font-semibold text-nier-golden-primary mb-phi-md">
            Sidebar (φ⁻¹ proportion)
          </h3>
          <div class="space-y-phi-sm">
            <div class="evidence-card-phi">
              <h4 class="text-phi-md font-medium mb-phi-sm">Case Files</h4>
              <p class="text-phi-sm text-gray-600">Evidence organized by golden ratio proportions</p>
            </div>
            <div class="evidence-card-phi">
              <h4 class="text-phi-md font-medium mb-phi-sm">AI Analysis</h4>
              <p class="text-phi-sm text-gray-600">Machine learning insights</p>
            </div>
            <div class="evidence-card-phi">
              <h4 class="text-phi-md font-medium mb-phi-sm">Legal Research</h4>
              <p class="text-phi-sm text-gray-600">Case law and precedents</p>
            </div>
          </div>
        </div>
      {/snippet}

      {#snippet children()}
        <div class="h-full bg-white p-phi-lg overflow-y-auto">
          <h3 class="text-phi-xl font-bold text-nier-golden-primary mb-phi-lg">
            Main Content Area (φ proportion)
          </h3>
          
          <div class="grid gap-phi-md mb-phi-lg">
            <div class="ai-panel-phi ai-confidence-phi-high">
              <h4 class="text-phi-lg font-semibold text-green-800">High Confidence AI Analysis</h4>
              <p class="text-phi-md">This legal document analysis shows strong correlations with established precedents.</p>
              <div class="text-phi-sm text-green-600 font-medium">Confidence: 94% • φ-based UI proportions</div>
            </div>

            <div class="ai-panel-phi ai-confidence-phi-medium">
              <h4 class="text-phi-lg font-semibold text-yellow-800">Medium Confidence Recommendation</h4>
              <p class="text-phi-md">Additional evidence may be required to strengthen this case argument.</p>
              <div class="text-phi-sm text-yellow-600 font-medium">Confidence: 72% • Golden ratio spacing</div>
            </div>

            <div class="ai-panel-phi ai-confidence-phi-low">
              <h4 class="text-phi-lg font-semibold text-red-800">Low Confidence Warning</h4>
              <p class="text-phi-md">This analysis requires human review due to conflicting legal interpretations.</p>
              <div class="text-phi-sm text-red-600 font-medium">Confidence: 34% • Requires attention</div>
            </div>
          </div>

          <div class="golden-section mb-phi-lg">
            <Card class="p-phi-lg">
              <h4 class="text-phi-lg font-semibold mb-phi-md">Evidence Timeline</h4>
              <div class="space-y-phi-sm">
                <div class="border-l-phi border-nier-golden-accent pl-phi-md">
                  <div class="text-phi-sm text-gray-500">2024-07-30 14:30</div>
                  <div class="text-phi-md">Document uploaded and processed</div>
                </div>
                <div class="border-l-phi border-nier-golden-accent pl-phi-md">
                  <div class="text-phi-sm text-gray-500">2024-07-30 14:32</div>
                  <div class="text-phi-md">AI analysis completed</div>
                </div>
              </div>
            </Card>

            <Card class="p-phi-lg">
              <h4 class="text-phi-lg font-semibold mb-phi-md">Case Statistics</h4>
              <div class="space-y-phi-sm">
                <div class="flex justify-between">
                  <span class="text-phi-sm">Total Evidence</span>
                  <span class="text-phi-sm font-semibold">24</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-phi-sm">AI Processed</span>
                  <span class="text-phi-sm font-semibold">22</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-phi-sm">Confidence Avg</span>
                  <span class="text-phi-sm font-semibold">78%</span>
                </div>
              </div>
            </Card>
          </div>

          <p class="text-phi-sm text-gray-600 leading-phi">
            This main content area uses the golden ratio (φ ≈ 1.618) for proportional spacing, typography, and layout. 
            All margins, padding, and font sizes are mathematically related to create visual harmony and optimal readability for legal professionals.
          </p>
        </div>
      {/snippet}

      {#snippet secondary()}
        <div class="h-full bg-gray-50 p-phi-md overflow-y-auto">
          <h3 class="text-phi-lg font-semibold text-nier-golden-primary mb-phi-md">
            Secondary Panel (1 proportion)
          </h3>
          
          <div class="space-y-phi-md">
            <div class="nier-golden-panel">
              <h4 class="text-phi-md font-semibold mb-phi-sm">Quick Actions</h4>
              <div class="space-y-phi-sm">
                <Button variant="outline" size="sm" class="w-full">Export Report</Button>
                <Button variant="outline" size="sm" class="w-full">Share Case</Button>
                <Button variant="outline" size="sm" class="w-full">Schedule Review</Button>
              </div>
            </div>

            <div class="nier-golden-panel">
              <h4 class="text-phi-md font-semibold mb-phi-sm">Related Cases</h4>
              <div class="space-y-phi-xs">
                <div class="text-phi-sm p-phi-sm bg-white rounded border">Case #2024-001</div>
                <div class="text-phi-sm p-phi-sm bg-white rounded border">Case #2024-003</div>
                <div class="text-phi-sm p-phi-sm bg-white rounded border">Case #2024-007</div>
              </div>
            </div>

            <div class="nier-golden-panel">
              <h4 class="text-phi-md font-semibold mb-phi-sm">AI Insights</h4>
              <p class="text-phi-sm text-gray-600">
                Pattern recognition suggests similar cases have 87% success rate with current evidence strength.
              </p>
            </div>
          </div>
        </div>
      {/snippet}

      {#snippet footer()}
        <div class="w-full bg-nier-golden-tertiary text-white p-phi-md">
          <div class="flex justify-between items-center">
            <div>
              <span class="text-phi-sm">Legal AI System • Golden Ratio Layout</span>
            </div>
            <div class="text-phi-sm">
              φ = {1.618033988749} • Current: {currentVariant} / {currentDirection}
            </div>
          </div>
        </div>
      {/snippet}
    </GoldenRatioGrid>
  </div>

  <!-- Golden Ratio Explanation -->
  <div class="mt-phi-xl">
    <Card class="p-phi-xl">
      <h2 class="text-phi-2xl font-bold text-nier-golden-primary mb-phi-lg">
        Golden Ratio in Legal AI Design
      </h2>
      
      <div class="golden-section gap-phi-xl">
        <div>
          <h3 class="text-phi-lg font-semibold mb-phi-md">Mathematical Foundation</h3>
          <p class="text-phi-md leading-phi mb-phi-md">
            The golden ratio (φ ≈ 1.618) is a mathematical constant found throughout nature and art. 
            When applied to user interface design, it creates visually pleasing and harmonious proportions.
          </p>
          <ul class="space-y-phi-sm text-phi-sm">
            <li><strong>φ = 1.618033988749...</strong> - The golden ratio</li>
            <li><strong>1/φ ≈ 0.618</strong> - The reciprocal (shorter segment)</li>
            <li><strong>φ² ≈ 2.618</strong> - Golden ratio squared</li>
            <li><strong>φ³ ≈ 4.236</strong> - Golden ratio cubed</li>
          </ul>
        </div>

        <div>
          <h3 class="text-phi-lg font-semibold mb-phi-md">Legal AI Benefits</h3>
          <ul class="space-y-phi-sm text-phi-sm">
            <li><strong>Visual Hierarchy:</strong> Important legal information gets φ proportion space</li>
            <li><strong>Reading Flow:</strong> Natural eye movement patterns for document review</li>
            <li><strong>Professional Appearance:</strong> Mathematical precision conveys reliability</li>
            <li><strong>Reduced Cognitive Load:</strong> Harmonious proportions reduce mental fatigue</li>
            <li><strong>Responsive Design:</strong> Maintains proportions across all device sizes</li>
            <li><strong>Accessibility:</strong> Optimal spacing for users with visual impairments</li>
          </ul>
        </div>
      </div>

      <div class="mt-phi-lg p-phi-md bg-gray-50 rounded-phi-sm">
        <h4 class="text-phi-md font-semibold mb-phi-sm">CSS Custom Properties Available:</h4>
        <div class="grid grid-cols-3 gap-phi-md text-phi-sm font-mono">
          <div>
            <strong>Spacing:</strong><br>
            --space-phi-xs<br>
            --space-phi-sm<br>
            --space-phi-md<br>
            --space-phi-lg<br>
            --space-phi-xl
          </div>
          <div>
            <strong>Typography:</strong><br>
            --text-phi-xs<br>
            --text-phi-sm<br>
            --text-phi-base<br>
            --text-phi-lg<br>
            --text-phi-xl
          </div>
          <div>
            <strong>Layout:</strong><br>
            --layout-phi-sidebar<br>
            --layout-phi-main<br>
            --aspect-phi-landscape<br>
            --aspect-phi-portrait
          </div>
        </div>
      </div>
    </Card>
  </div>
</div>

<style>
  /* Component-specific golden ratio styles */
  .border-phi {
    border-width: calc(1px * var(--golden-ratio));
  }
  
  .rounded-phi-sm {
    border-radius: var(--space-phi-sm);
  }
  
  .rounded-phi-md {
    border-radius: var(--space-phi-md);
  }
  
  /* Demo-specific styling */
  :global(.debug-golden-grid) {
    position: relative
  }
  
  :global(.debug-golden-grid::before) {
    content: '';
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
      linear-gradient(to right, rgba(255, 0, 0, 0.1) 0, rgba(255, 0, 0, 0.1) 61.8%, transparent 61.8%),
      linear-gradient(to bottom, rgba(0, 255, 0, 0.1) 0, rgba(0, 255, 0, 0.1) 61.8%, transparent 61.8%);
    pointer-events: none
    z-index: 1000;
  }
</style>