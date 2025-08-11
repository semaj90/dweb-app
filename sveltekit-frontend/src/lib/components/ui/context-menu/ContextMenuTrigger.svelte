<script lang="ts">
  interface Props {
    trigger: any;;
  }
  let {
    trigger
  }: Props = $props();



  </script>

<div use:trigger {...$$restProps}>
  <slot />
</div>
