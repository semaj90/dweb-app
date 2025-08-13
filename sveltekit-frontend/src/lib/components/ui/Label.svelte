<script lang="ts">
  interface Props {
    for_?: string;
    class_?: string;
  }
  
  let {
    for_ = '',
    class_ = '',
    children
  }: Props & { children?: any } = $props();
</script>

<label for={for_} class="text-sm font-medium text-gray-700 dark:text-gray-300 {class_}">
  {#if children}
    {@render children()}
  {/if}
</label>
