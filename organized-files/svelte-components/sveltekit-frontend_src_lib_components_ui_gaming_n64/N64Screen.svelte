<script lang="ts">
  export let content: string = 'Screen content';
</script>

<style>
  .screen { width:320px; height:240px; background:#000; color:#0f0; padding:8px; font-family:monospace }
</style>

<div class="screen">{content}</div>
