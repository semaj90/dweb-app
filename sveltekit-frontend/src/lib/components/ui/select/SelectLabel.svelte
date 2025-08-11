<script lang="ts">
  interface Props {
    class_: string ;
  }
  let {
    class_ = ''
  }: Props = $props();



  </script>

<span class="space-y-4">
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
