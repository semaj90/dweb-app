<script lang="ts">
  let className: string = '';
  export { className as class };
</script>

<div class="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground {className}">
  <slot />
</div>