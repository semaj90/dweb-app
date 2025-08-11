<script lang="ts">
  interface Props {
    trigger: any;;
  }
  let {
    trigger
  }: Props = $props();



  </script>

<button use:trigger {...$$restProps}>
  <slot />
</button>
