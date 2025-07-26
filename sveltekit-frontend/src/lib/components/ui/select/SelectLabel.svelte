<script lang="ts">
  export let class_: string = '';
</script>

<span class="container mx-auto px-4">
  <slot />
</span>

<style>
  /* @unocss-include */
  .select-label {
    font-size: 12px;
    font-weight: 500;
    color: #6b7280;
    margin-bottom: 4px;
    display: block;
}
</style>
