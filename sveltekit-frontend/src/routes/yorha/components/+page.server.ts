import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url, fetch }) => {
  try {
    // Define YoRHa 3D component specifications for SSR
    const componentSpecs = {
      button: {
        id: 'yorha-button-3d',
        name: 'YoRHa Button 3D',
        description: 'Cyberpunk-themed interactive button with 3D effects',
        category: 'input',
        variants: ['primary', 'secondary', 'danger', 'success', 'warning'],
        sizes: ['small', 'medium', 'large', 'xl'],
        features: ['glowEffect', 'hoverAnimation', 'clickFeedback', 'loadingState'],
        defaultConfig: {
          text: 'YoRHa Button',
          variant: 'primary',
          size: 'medium',
          icon: 'terminal',
          loading: false,
          disabled: false,
          glowEffect: true,
          hoverAnimation: true
        }
      },
      panel: {
        id: 'yorha-panel-3d',
        name: 'YoRHa Panel 3D',
        description: 'Futuristic panel with holographic effects',
        category: 'layout',
        variants: ['default', 'terminal', 'status', 'data'],
        features: ['borderGlow', 'glitchEffect', 'collapsible', 'scrollable'],
        defaultConfig: {
          title: 'YoRHa Panel',
          variant: 'default',
          width: 400,
          height: 300,
          scrollable: true,
          collapsible: true,
          glitchEffect: false,
          borderGlow: true
        }
      },
      input: {
        id: 'yorha-input-3d',
        name: 'YoRHa Input 3D',
        description: 'Advanced input field with neural interface styling',
        category: 'input',
        types: ['text', 'email', 'password', 'search', 'number'],
        variants: ['default', 'terminal', 'secure', 'scanner'],
        features: ['autoComplete', 'validation', 'neuralGlow', 'scanlineEffect'],
        defaultConfig: {
          placeholder: 'Enter command...',
          type: 'text',
          variant: 'terminal',
          value: '',
          error: false,
          autoComplete: true,
          neuralGlow: true
        }
      },
      modal: {
        id: 'yorha-modal-3d',
        name: 'YoRHa Modal 3D',
        description: 'Immersive modal dialog with depth effects',
        category: 'overlay',
        variants: ['default', 'alert', 'confirm', 'fullscreen'],
        sizes: ['small', 'medium', 'large', 'xl', 'fullscreen'],
        features: ['backdropBlur', 'depthEffect', 'enterAnimation', 'exitAnimation'],
        defaultConfig: {
          title: 'YoRHa Modal',
          variant: 'default',
          size: 'medium',
          closable: true,
          backdropBlur: true,
          depthEffect: true
        }
      },
      terminal: {
        id: 'yorha-terminal-3d',
        name: 'YoRHa Terminal 3D',
        description: 'Advanced terminal interface with command processing',
        category: 'utility',
        themes: ['dark', 'amber', 'green', 'blue'],
        features: ['commandHistory', 'autoComplete', 'syntaxHighlight', 'typewriter'],
        defaultConfig: {
          theme: 'amber',
          prompt: 'YORHA:~$',
          history: [],
          autoComplete: true,
          typewriterEffect: true
        }
      },
      dataviz: {
        id: 'yorha-dataviz-3d',
        name: 'YoRHa Data Visualization',
        description: 'Real-time data visualization with 3D charts',
        category: 'display',
        chartTypes: ['line', 'bar', 'pie', 'scatter', 'heatmap', 'network'],
        features: ['realTime', 'animation', 'interaction', 'export'],
        defaultConfig: {
          type: 'line',
          animated: true,
          realTime: true,
          theme: 'cyberpunk'
        }
      }
    };

    // Generate component gallery data
    const galleryData = {
      categories: [
        { id: 'input', name: 'Input Components', count: 2 },
        { id: 'layout', name: 'Layout Components', count: 1 },
        { id: 'overlay', name: 'Overlay Components', count: 1 },
        { id: 'utility', name: 'Utility Components', count: 1 },
        { id: 'display', name: 'Display Components', count: 1 }
      ],
      components: Object.values(componentSpecs),
      totalComponents: Object.keys(componentSpecs).length,
      previewModes: [
        { id: '3d', label: '3D Preview', icon: 'Layers' },
        { id: 'code', label: 'Code View', icon: 'Code' },
        { id: 'config', label: 'Configuration', icon: 'Settings' }
      ]
    };

    // Generate demo code snippets
    const codeSnippets = {
      button: `<script>
  import { YoRHaButton3D } from '$lib/components/three/yorha-ui';
  
  let buttonConfig = {
    text: 'Execute Command',
    variant: 'primary',
    size: 'medium',
    glowEffect: true
  };
</script>

<YoRHaButton3D 
  {...buttonConfig}
  onclick={() => console.log('YoRHa button clicked')}
/>`,
      panel: `<script>
  import { YoRHaPanel3D } from '$lib/components/three/yorha-ui';
</script>

<YoRHaPanel3D
  title="System Status"
  variant="terminal"
  borderGlow={true}
>
  <div slot="content">
    Panel content goes here...
  </div>
</YoRHaPanel3D>`,
      input: `<script>
  import { YoRHaInput3D } from '$lib/components/three/yorha-ui';
  
  let inputValue = '';
</script>

<YoRHaInput3D
  bind:value={inputValue}
  placeholder="Enter neural link..."
  variant="terminal"
  neuralGlow={true}
/>`,
      modal: `<script>
  import { YoRHaModal3D } from '$lib/components/three/yorha-ui';
  
  let showModal = false;
</script>

<YoRHaModal3D
  bind:open={showModal}
  title="System Alert"
  variant="alert"
  size="medium"
>
  <div slot="content">
    Modal content...
  </div>
</YoRHaModal3D>`
    };

    return {
      specs: componentSpecs,
      gallery: galleryData,
      codeSnippets,
      initialLoad: true,
      timestamp: new Date().toISOString()
    };
  } catch (error: any) {
    console.error('Error loading YoRHa components data:', error);
    
    return {
      specs: {},
      gallery: { categories: [], components: [], totalComponents: 0, previewModes: [] },
      codeSnippets: {},
      initialLoad: false,
      timestamp: new Date().toISOString(),
      error: 'Failed to load components data'
    };
  }
};