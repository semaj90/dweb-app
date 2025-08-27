<script lang="ts">
  import type { LayoutData } from "./$types";

  let { data = $bindable(), children } = $props(); // LayoutData;
</script>

<div class="min-h-screen bg-gray-50">
  <div class="max-w-7xl mx-auto p-6">
    <div class="bg-white rounded-lg shadow-sm p-6">
      <h1 class="text-2xl font-bold text-gray-900 mb-6">Cases</h1>
      {@render children()}
    </div>
  </div>
</div>