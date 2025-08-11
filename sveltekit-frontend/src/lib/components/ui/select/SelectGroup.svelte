<script lang="ts">
  interface Props {
    class_: string ;
  }
  let {
    class_ = ''
  }: Props = $props();



  </script>

<div class="space-y-4">
  <slot />
</div>

<style>
  /* @unocss-include */
  .select-group {
    padding: 4px 0;
}
</style>
