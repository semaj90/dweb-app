<script lang="ts">
  interface Props {
    trigger: any
  }
  let {
    trigger
  } = $props();



  </script>

<button use:trigger >
  <slot></slot>
</button>
