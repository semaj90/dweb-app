<script lang="ts">
  import Button from './Button.svelte';

  let underline = $state(false);
  let bold = $state(false);
  let italic = $state(false);
  let red = $state(false);
</script>

<p class={['p-3', { underline, 'font-bold': bold }, italic && 'italic']} class:c-red={red}>
  Demo text using clsx classes and <code>`class:`</code> directive
</p>

<Button on:click={() => (underline = !underline)}>toggle underline</Button>
<Button on:click={() => (bold = !bold)}>toggle bold</Button>
<Button on:click={() => (italic = !italic)}>toggle italic</Button>
<Button on:click={() => (red = !red)}>toggle red</Button>
