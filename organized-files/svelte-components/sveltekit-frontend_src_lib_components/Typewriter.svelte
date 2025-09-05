<script lang="ts">
  interface Props {
    text: string
    speed: number
  }
  let {
    text = '',
    speed = 50
  } = $props();



      
  let output = '';
  let i = 0;
  let intervalId: NodeJS.Timeout | null = null;
  
  $effect(() => { if (text && i === 0) {
    output = '';
    i = 0;
    
    if (intervalId) {
      clearInterval(intervalId);
}
    intervalId = setInterval(() => {
      if (i < text.length) {
        output += text[i];
        i++;
      } else {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
}}
    }, speed);
}
</script>

<div class="space-y-4">
  {output}
</div>

<style>
  /* @unocss-include */
  .typewriter {
    font-size: 1.2rem;
    color: #6c757d;
    min-height: 1.5rem;
    border-right: 2px solid #007bff;
    padding-right: 5px;
    animation: blink 1s infinite;
}
  @keyframes blink {
    0%, 50% { 
      border-color: #007bff; 
}
    51%, 100% { 
      border-color: transparent 
}}
</style>
