<script lang="ts">
  import { Progress, Card, Badge } from '$lib/components/ui/modular';

  interface Props {
    progress?: number;
    fileName?: string;
    label?: string;
    variant?: 'default' | 'success' | 'warning' | 'error' | 'yorha' | 'legal' | 'evidence';
    status?: 'uploading' | 'completed' | 'error' | 'paused';
    showPercentage?: boolean;
  }

  let {
    progress = 0,
    fileName = '',
    label = 'Uploading file',
    variant = 'default',
    status = 'uploading',
    showPercentage = true
  }: Props = $props();

  // Computed values
  let progressVariant = $derived(() => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'paused': return 'warning';
      default: return variant === 'yorha' ? 'yorha' : variant === 'legal' ? 'legal' : 'info';
    }
  });

  let badgeVariant = $derived(() => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'destructive';
      case 'paused': return 'warning';
      default: return 'info';
    }
  });

  let statusText = $derived(() => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'error': return 'Failed';
      case 'paused': return 'Paused';
      default: return 'Uploading';
    }
  });
</script>

<Card variant={variant} class="w-full">
  <!-- File info header -->
  <div class="flex items-center justify-between mb-4">
    <div class="flex items-center gap-3">
      <div class="i-lucide-file w-5 h-5 text-muted-foreground" aria-hidden="true"></div>
      <div>
        <p class="font-medium text-sm">{fileName || 'Unknown file'}</p>
        <p class="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
    
    <!-- Status Badge -->
    <Badge variant={badgeVariant()} size="sm">
      {statusText()}
    </Badge>
  </div>

  <!-- Progress Bar -->
  <Progress
    value={progress}
    variant={progressVariant()}
    showPercentage={showPercentage}
    size="default"
    class="mb-2"
  />

  <!-- Additional Info -->
  {#if status === 'error'}
    <p class="text-xs text-red-600 mt-2">
      Upload failed. Please try again.
    </p>
  {:else if status === 'completed'}
    <p class="text-xs text-green-600 mt-2">
      Upload completed successfully!
    </p>
  {:else if status === 'paused'}
    <p class="text-xs text-yellow-600 mt-2">
      Upload paused. Click to resume.
    </p>
  {/if}
</Card>
