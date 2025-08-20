<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  interface Props extends CommonProps {
    class_: string ;
  }
  let {
    class_ = ''
  }: Props = $props();



  </script>

<div class="my-1 h-px bg-gray-200 dark:bg-gray-700 {class_}" role="separator"></div>
