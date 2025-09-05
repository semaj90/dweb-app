<script lang="ts">
  interface Props {
    trigger: any
  }
  let {
    trigger
  } = $props();



  </script>

<div use:trigger >
  <slot></slot>
</div>
