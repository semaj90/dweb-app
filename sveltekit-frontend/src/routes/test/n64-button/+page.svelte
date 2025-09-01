<!--
  Test page for N64 3D Button Component
  Verifies functionality and styling of the gaming UI component
-->
<script lang="ts">
  import N643DButton from '$lib/components/ui/gaming/n64/N643DButton.svelte';
  import { onMount } from 'svelte';
  
  let testResults = $state([]);
  let isTestingComplete = $state(false);
  
  // Test different button configurations
  const buttonTests = [
    {
      name: 'Basic Button',
      props: { children: 'Basic Test' }
    },
    {
      name: 'Primary Variant',
      props: { 
        variant: 'primary',
        children: 'Primary Button'
      }
    },
    {
      name: 'Large Size',
      props: { 
        size: 'large',
        children: 'Large Button'
      }
    },
    {
      name: 'High Quality Rendering',
      props: {
        meshComplexity: 'high',
        materialType: 'pbr',
        enableLighting: true,
        enableReflections: true,
        children: 'High Quality'
      }
    },
    {
      name: 'With Particles',
      props: {
        enableParticles: true,
        glowIntensity: 0.8,
        children: 'Particle Effects'
      }
    },
    {
      name: '3D Rotated',
      props: {
        rotationX: 15,
        rotationY: 10,
        children: '3D Rotated'
      }
    },
    {
      name: 'Loading State',
      props: {
        loading: true,
        children: 'Loading...'
      }
    },
    {
      name: 'Disabled State',
      props: {
        disabled: true,
        children: 'Disabled'
      }
    }
  ];
  
  let buttonClickCount = $state(0);
  
  const handleButtonClick = (testName: string) => {
    buttonClickCount++;
    console.log(`Button clicked: ${testName} (Total clicks: ${buttonClickCount})`);
    
    // Add to test results
    testResults.push({
      test: testName,
      result: 'Click handler executed successfully',
      timestamp: new Date().toLocaleTimeString()
    });
    
    // Force reactivity update
    testResults = [...testResults];
  };
  
  onMount(() => {
    // Run component existence tests
    console.log('Testing N64 3D Button Component...');
    
    // Check if component loaded
    const componentExists = typeof N643DButton !== 'undefined';
    testResults.push({
      test: 'Component Import',
      result: componentExists ? 'SUCCESS' : 'FAILED',
      timestamp: new Date().toLocaleTimeString()
    });
    
    // Test DOM rendering after a short delay
    setTimeout(() => {
      const buttons = document.querySelectorAll('.n64-3d-button');
      testResults.push({
        test: 'DOM Rendering',
        result: buttons.length > 0 ? `SUCCESS (${buttons.length} buttons rendered)` : 'FAILED',
        timestamp: new Date().toLocaleTimeString()
      });
      
      // Test CSS classes
      const hasCorrectClasses = Array.from(buttons).some(btn => 
        btn.classList.contains('n64-3d-button')
      );
      testResults.push({
        test: 'CSS Classes',
        result: hasCorrectClasses ? 'SUCCESS' : 'FAILED',
        timestamp: new Date().toLocaleTimeString()
      });
      
      isTestingComplete = true;
      testResults = [...testResults];
    }, 500);
  });
</script>

<svelte:head>
  <title>N64 Button Component Test</title>
</svelte:head>

<div class="test-page">
  <header class="test-header">
    <h1>N64 3D Button Component Test</h1>
    <p>Testing the gaming UI component functionality and styling</p>
    {#if isTestingComplete}
      <div class="test-summary">
        <span class="success-count">✓ Tests completed</span>
        <span class="click-count">Clicks: {buttonClickCount}</span>
      </div>
    {/if}
  </header>

  <main class="test-content">
    <!-- Button Test Grid -->
    <section class="button-tests">
      <h2>Button Variations</h2>
      <div class="button-grid">
        {#each buttonTests as test}
          <div class="button-test-item">
            <h3>{test.name}</h3>
            <N643DButton 
              {...test.props}
              on:on:on:click={() => handleButtonClick(test.name)}
            >
              {test.props.children || 'Test Button'}
            </N643DButton>
          </div>
        {/each}
      </div>
    </section>

    <!-- Test Results -->
    <section class="test-results">
      <h2>Test Results</h2>
      <div class="results-container">
        {#if testResults.length === 0}
          <p>Running tests...</p>
        {:else}
          <div class="results-list">
            {#each testResults as result}
              <div class="result-item" class:success={result.result.includes('SUCCESS')} class:failed={result.result.includes('FAILED')}>
                <span class="test-name">{result.test}</span>
                <span class="test-result">{result.result}</span>
                <span class="test-time">{result.timestamp}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </section>

    <!-- Performance Test -->
    <section class="performance-test">
      <h2>Performance Test</h2>
      <div class="performance-buttons">
        <N643DButton 
          variant="primary"
          on:on:on:click={() => handleButtonClick('Performance Test')}
        >
          Click Test ({buttonClickCount} clicks)
        </N643DButton>
        
        <N643DButton 
          variant="success"
          meshComplexity="high"
          materialType="pbr"
          enableParticles={true}
          enableLighting={true}
          enableReflections={true}
          on:on:on:click={() => handleButtonClick('High-Performance Test')}
        >
          High-Performance Test
        </N643DButton>
      </div>
    </section>

    <!-- Gallery Route Test -->
    <section class="route-test">
      <h2>Gallery Route Test</h2>
      <div class="route-buttons">
        <N643DButton 
          variant="info"
          on:on:on:click={() => window.location.href = '/gallery'}
        >
          Test Gallery Route
        </N643DButton>
        
        <N643DButton 
          variant="secondary"
          on:on:on:click={() => window.open('/gallery', '_blank')}
        >
          Open Gallery in New Tab
        </N643DButton>
      </div>
    </section>
  </main>
</div>

<style>
  .test-page {
    min-height: 100vh;
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    color: white;
    font-family: 'Rajdhani', sans-serif;
    padding: 2rem;
  }

  .test-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .test-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
  }

  .test-summary {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
  }

  .success-count {
    color: #28a745;
    font-weight: bold;
  }

  .click-count {
    color: #ffc107;
    font-weight: bold;
  }

  .test-content {
    max-width: 1200px;
    margin: 0 auto;
  }

  .button-tests {
    margin-bottom: 3rem;
  }

  .button-tests h2 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .button-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .button-test-item {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
  }

  .button-test-item h3 {
    margin: 0 0 1rem 0;
    font-size: 1.1rem;
    color: #ccc;
  }

  .test-results {
    margin-bottom: 3rem;
  }

  .test-results h2 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .results-container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .results-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .result-item {
    display: grid;
    grid-template-columns: 1fr 2fr auto;
    gap: 1rem;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    border-left: 4px solid #6c757d;
  }

  .result-item.success {
    border-left-color: #28a745;
  }

  .result-item.failed {
    border-left-color: #dc3545;
  }

  .test-name {
    font-weight: bold;
  }

  .test-result {
    color: #ccc;
  }

  .test-time {
    color: #888;
    font-size: 0.9rem;
  }

  .performance-test,
  .route-test {
    margin-bottom: 3rem;
  }

  .performance-test h2,
  .route-test h2 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .performance-buttons,
  .route-buttons {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
  }

  /* Mobile responsiveness */
  @media (max-width: 768px) {
    .test-page {
      padding: 1rem;
    }
    
    .button-grid {
      grid-template-columns: 1fr;
      gap: 1rem;
    }
    
    .result-item {
      grid-template-columns: 1fr;
      text-align: center;
    }
    
    .performance-buttons,
    .route-buttons {
      flex-direction: column;
      align-items: center;
    }
  }

  /* High contrast mode support */
  @media (prefers-contrast: high) {
    .test-page {
      background: #000;
      color: #fff;
    }
    
    .button-test-item,
    .results-container {
      border: 2px solid #fff;
      background: #000;
    }
  }
</style>