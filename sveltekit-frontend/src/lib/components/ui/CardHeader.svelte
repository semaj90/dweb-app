<script lang="ts">
  interface Props {
    className: string ;
  }
  let {
    className = ''
  }: Props = $props();



    $: classes = className;
</script>

<div class="space-y-4">
  <slot />
</div>
