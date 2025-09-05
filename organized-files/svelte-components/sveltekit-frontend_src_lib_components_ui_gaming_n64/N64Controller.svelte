<script lang="ts">
  export let onPress: (button: string) => void = () => {};
</script>

<style>
  .controller { display: flex gap:8px; padding:8px; background:#222; color:#fff; border-radius:6px }
  button { padding:6px 8px; background:#444; color:#fff; border:0; border-radius:4px }
</style>

<div class="controller">
  <button on:click={() => onPress('A')}>A</button>
  <button on:click={() => onPress('B')}>B</button>
  <button on:click={() => onPress('Start')}>Start</button>
</div>
