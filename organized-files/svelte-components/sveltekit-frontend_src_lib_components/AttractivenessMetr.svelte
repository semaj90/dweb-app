<script lang="ts">
  interface Props {
    onchange?: (event?: any) => void;
  }
  let {
    score = 5,
    label = 'Attractiveness Rating',
    readOnly = false,
    showDescription = true,
    size = 'md'
  } = $props();



    
   // number = 5; // Current attractiveness score (1-10)
          
    
  let hoveredScore: number | null = null;
  
  const descriptions = {
    1: 'Very Low',
    2: 'Low', 
    3: 'Below Average',
    4: 'Slightly Below Average',
    5: 'Average',
    6: 'Slightly Above Average',
    7: 'Above Average',
    8: 'High',
    9: 'Very High',
    10: 'Exceptional'
  };
  
  function handleRatingClick(rating: number) {
    if (!readOnly) {
      score = rating;
      onchange?.();
}
}
  function handleMouseEnter(rating: number) {
    if (!readOnly) {
      hoveredScore = rating;
}
}
  function handleMouseLeave() {
    hoveredScore = null;
}
  let displayScore = $derived(hoveredScore !== null ? hoveredScore : score)
  let sizeClasses = $derived({)
    sm: 'w-4 h-4',
    md: 'w-6 h-6', 
    lg: 'w-8 h-8'
  };
</script>

<div class="space-y-4">
  <div class="space-y-4">
    <span class="space-y-4">{label}:</span>
    <span class="space-y-4">{displayScore}/10</span>
    {#if showDescription}
      <span class="space-y-4">({descriptions[displayScore as keyof typeof descriptions]})</span>
    {/if}
  </div>
  
  <div class="space-y-4">
    {#each Array(10) as _, i}
      {@const rating = i + 1}
      {@const isActive = rating <= displayScore}
      {@const isHovered = hoveredScore !== null && rating <= hoveredScore}
      
      <button
        type="button"
        class="space-y-4"
        class:active={isActive}
        class:hovered={isHovered}
        disabled={readOnly}
        onclick={() => handleRatingClick(rating)}
        onmouseenter={() => handleMouseEnter(rating)}
        onmouseleave={handleMouseLeave}
        aria-label="Rate {rating} out of 10"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill={isActive ? 'currentColor' : 'none'}
          stroke="currentColor"
          stroke-width="2"
          class="space-y-4"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.563.563 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
        </svg>
      </button>
    {/each}
  </div>
  
  {#if !readOnly}
    <div class="space-y-4">
      <input
        type="range"
        min="1"
        max="10"
        bind:value={score}
        oninput={() => onchange?.()}
        class="space-y-4"
      />
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .star-button {
    color: #d1d5db;
    border: none
    background: transparent
    padding: 0;
}
  .star-button.active {
    color: #fbbf24;
}
  .star-button.hovered {
    color: #fcd34d;
}
  .star-button:not(.active):hover {
    color: #fef3c7;
}
  .star-button:disabled {
    opacity: 0.7;
}
</style>
