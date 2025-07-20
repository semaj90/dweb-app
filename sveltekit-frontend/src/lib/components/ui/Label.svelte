<script lang="ts">
  // Basic Label component
  export let for_: string = '';
  export let class_: string = '';
</script>

<label for={for_} class="container mx-auto px-4">
  <slot />
</label>
