<script lang="ts">
  interface Props {
    class_: string
  }
  let {
    class_ = ''
  } = $props();



  </script>

<div class="my-1 h-px bg-gray-200 dark:bg-gray-700 {class_}" role="separator"></div>
