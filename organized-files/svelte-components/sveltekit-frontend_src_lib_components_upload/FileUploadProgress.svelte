<script lang="ts">
  interface Props {
    progress?: any;
    fileName?: any;
    label?: any;
  }
  let {
    progress = 0,
    fileName = '',
    label = ''
  } = $props();



  import { Progress } from 'bits-ui';
      </script>

<div class="w-full p-4 rounded-lg bg-white shadow flex flex-col gap-2 uno-border uno-border-gray-200">
  <div class="flex items-center justify-between mb-2">
    <span class="font-medium text-gray-700">{label || 'Uploading file'}:</span>
    <span class="text-sm text-gray-500">{fileName}</span>
  </div>
  <Progress value={progress} class="h-2 w-full bg-green-500 rounded" />
  <div class="text-xs text-gray-400 mt-1">{progress}%</div>
</div>

<style>
  /* @unocss-include */
  .uno-border {
    border: 1px solid #e5e7eb;
  }
</style>
