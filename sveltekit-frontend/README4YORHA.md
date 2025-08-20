# YoRHa 3D UI & Brain Visualization

> **Production-ready Three.js UI components with Square Enix NieR: Automata gothic aesthetic**

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Three.js](https://img.shields.io/badge/Three.js-000000?style=flat&logo=three.js&logoColor=white)](https://threejs.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-FF6B00?style=flat&logo=webgl&logoColor=white)](https://gpuweb.github.io/gpuweb/)
[![SvelteKit](https://img.shields.io/badge/SvelteKit-FF3E00?style=flat&logo=svelte&logoColor=white)](https://kit.svelte.dev/)

## 🎯 **Overview**

YoRHa 3D UI is a comprehensive component library that brings **CSS-like styling capabilities** to Three.js while implementing the distinctive **gothic aesthetic** from Square Enix's NieR: Automata YoRHa interface design. Perfect for legal AI platforms, data visualization, and immersive applications.

---

## ✨ **Features**

### 🎨 **Advanced Styling System**
- **CSS-like Properties**: Padding, margins, borders, shadows, gradients in 3D space
- **Interactive States**: Hover, active, disabled, focus with smooth transitions
- **YoRHa Color Palette**: Authentic gothic colors (black, white, beige, gold)
- **Material Effects**: Glow, scan lines, glitch animations, terminal aesthetics

### 🧩 **Core Components**
- **YoRHaButton3D**: Interactive buttons with variants, icons, loading states
- **YoRHaPanel3D**: Containers with headers, scrolling, resizing capabilities
- **YoRHaInput3D**: Text fields with validation, icons, different input types
- **YoRHaModal3D**: Dialogs with backdrop effects, animations, auto-close

### 📐 **Layout System**
- **Flexbox**: Row/column layouts with justify, align, wrap properties
- **CSS Grid**: Template areas, spanning, auto-placement
- **Absolute**: Direct 3D positioning with coordinates
- **Stack**: Z-depth layering with automatic spacing
- **Flow**: Automatic wrapping for responsive layouts

### ⚡ **Performance Features**
- **WebGPU Integration**: GPU-accelerated computations and physics
- **Real-time Data**: WebSocket and SSE integration for live updates
- **Memory Management**: Automatic cleanup and resource disposal
- **Modular Architecture**: Tree-shakable imports and lazy loading

---

## 🚀 **Quick Start**

### Installation

```bash
# Install dependencies
npm install three @types/three

# Copy YoRHa UI library to your project
cp -r src/lib/components/three/yorha-ui your-project/src/lib/components/three/
```

### Basic Usage

```typescript
import {
  YoRHaButton3D,
  YoRHaPanel3D,
  YoRHaLayoutPresets,
  createYoRHaUIDemo
} from '$lib/components/three/yorha-ui';

// Quick demo setup
const demo = createYoRHaUIDemo(document.getElementById('ui-container'));

// Custom component creation
const button = new YoRHaButton3D({
  text: 'Execute Command',
  variant: 'primary',
  icon: 'play',
  size: 'large'
});

const panel = new YoRHaPanel3D({
  title: 'Control Panel',
  variant: 'terminal',
  width: 6,
  height: 4
});

// Layout management
const layout = YoRHaLayoutPresets.createFlexColumn(0.3);
layout.addChild(button);
layout.addChild(panel);

// Add to Three.js scene
scene.add(layout);
```

---

## 📚 **Component Reference**

### YoRHaButton3D

Interactive 3D buttons with YoRHa styling.

```typescript
const button = new YoRHaButton3D({
  text: 'Action Button',
  variant: 'primary' | 'secondary' | 'accent' | 'ghost' | 'danger',
  size: 'small' | 'medium' | 'large',
  icon: 'play' | 'pause' | 'stop' | 'arrow-right' | 'plus',
  iconPosition: 'left' | 'right' | 'top' | 'bottom',
  rounded: true,
  loading: false,
  // CSS-like styling
  width: 3,
  height: 0.6,
  backgroundColor: 0xffffff,
  borderColor: 0x000000,
  borderWidth: 0.02,
  borderRadius: 0.1,
  shadow: {
    enabled: true,
    color: 0x000000,
    blur: 0.3,
    intensity: 0.4
  },
  glow: {
    enabled: true,
    color: 0xd4af37,
    intensity: 0.3
  }
});

// Event handling
button.addEventListener('click', () => {
  console.log('Button clicked!');
});

button.addEventListener('hover', () => {
  console.log('Button hovered!');
});
```

**Button Variants:**
- `primary`: Gold accent, main action buttons
- `secondary`: Grey styling, secondary actions
- `accent`: Bright gold with glow effect
- `ghost`: Transparent with border outline
- `danger`: Red styling for destructive actions

### YoRHaPanel3D

Container components with advanced features.

```typescript
const panel = new YoRHaPanel3D({
  title: 'Data Terminal',
  variant: 'default' | 'outlined' | 'filled' | 'glass' | 'terminal',
  width: 5,
  height: 4,
  showCloseButton: true,
  resizable: true,
  minimizable: true,
  scrollable: true,
  headerHeight: 0.6
});

// Add content to panel
const childButton = new YoRHaButton3D({ text: 'Child Action' });
panel.addContent(childButton);

// Event handling
panel.addEventListener('close', () => {
  console.log('Panel closed');
});

panel.addEventListener('resize', (event) => {
  console.log('Panel resized:', event.data);
});
```

### YoRHaInput3D

Advanced text input fields with validation.

```typescript
const input = new YoRHaInput3D({
  placeholder: 'Enter data...',
  type: 'text' | 'password' | 'email' | 'search' | 'number',
  variant: 'default' | 'outlined' | 'filled' | 'ghost' | 'terminal',
  value: 'Initial value',
  maxLength: 255,
  clearable: true,
  icon: 'user' | 'lock' | 'email' | 'search',
  iconPosition: 'left' | 'right',
  error: false,
  success: false,
  multiline: false,
  rows: 1
});

// Input handling
input.addEventListener('input', (event) => {
  console.log('Input changed:', event.data.value);
});

input.addEventListener('submit', (event) => {
  console.log('Form submitted:', event.data.value);
});

// Methods
input.focus();
input.blur();
input.setValue('New value');
input.setError(true);
input.setSuccess(true);
input.clear();
```

### YoRHaModal3D

Modal dialogs with animations and backdrop effects.

```typescript
const modal = new YoRHaModal3D({
  title: 'Confirmation Dialog',
  variant: 'default' | 'alert' | 'confirm' | 'fullscreen' | 'terminal',
  size: 'small' | 'medium' | 'large' | 'fullscreen',
  backdrop: 'blur' | 'dark' | 'transparent' | 'none',
  closable: true,
  persistent: false,
  showHeader: true,
  showFooter: false
});

// Modal content
const modalLayout = YoRHaLayoutPresets.createDialog();
const confirmButton = new YoRHaButton3D({ text: 'Confirm', variant: 'primary' });
modalLayout.addChild(confirmButton);
modal.addContent(modalLayout);

// Modal control
modal.open();
modal.close();
modal.toggle();

// Events
modal.addEventListener('open', () => console.log('Modal opened'));
modal.addEventListener('closed', () => console.log('Modal closed'));
```

---

## 🏗️ **Layout System**

### Flexbox Layout

```typescript
const flexLayout = new YoRHaLayout3D({
  type: 'flex',
  direction: 'row' | 'column' | 'row-reverse' | 'column-reverse',
  justify: 'start' | 'end' | 'center' | 'space-between' | 'space-around',
  align: 'start' | 'end' | 'center' | 'stretch' | 'baseline',
  wrap: 'nowrap' | 'wrap' | 'wrap-reverse',
  gap: 0.3,
  padding: { top: 0.2, right: 0.2, bottom: 0.2, left: 0.2, front: 0, back: 0 }
});

// Add children with individual layout properties
flexLayout.addChild(component, {
  flex: 1,
  grow: 1,
  shrink: 0,
  basis: 'auto',
  alignSelf: 'center',
  order: 1
});
```

### Grid Layout

```typescript
const gridLayout = new YoRHaLayout3D({
  type: 'grid',
  gridColumns: 3,
  gridRows: 2,
  gap: 0.2
});

// Add children with grid positioning
gridLayout.addChild(component, {
  gridColumn: 1,    // or "1 / 3" for spanning
  gridRow: 2
});
```

### Layout Presets

```typescript
// Common layout patterns
const rowLayout = YoRHaLayoutPresets.createFlexRow(0.2);
const columnLayout = YoRHaLayoutPresets.createFlexColumn(0.2);
const gridLayout = YoRHaLayoutPresets.createGrid(3, 2, 0.2);
const dialogLayout = YoRHaLayoutPresets.createDialog();
const formLayout = YoRHaLayoutPresets.createForm();
const toolbarLayout = YoRHaLayoutPresets.createToolbar();
```

---

## 🌐 **API Integration**

### Real-time Data Binding

```typescript
import { yorhaAPI, YoRHaAPIClient } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';

// Initialize API client
const api = new YoRHaAPIClient({
  baseURL: 'http://localhost:5173/api/yorha',
  enableWebSocket: true,
  enableSSE: true
});

// Create components from API data
const button = await api.createButtonFromAPI('control-button-1');
const panel = await api.createPanelFromAPI('data-panel-1');

// Subscribe to real-time updates
api.subscribe('component:control-button-1:updated', (data) => {
  button.setText(data.text);
  button.setVariant(data.variant);
});

// System monitoring
const systemStatus = await api.getSystemStatus();
console.log('System health:', systemStatus);

// Event logging
api.logEvent({
  type: 'user_interaction',
  componentId: 'control-button-1',
  data: { action: 'click', timestamp: Date.now() }
});
```

### Component Configuration

```json
// API Response format for component configuration
{
  "id": "control-button-1",
  "type": "button",
  "config": {
    "text": "Execute Command",
    "variant": "primary",
    "size": "large",
    "icon": "play",
    "width": 3,
    "height": 0.6
  },
  "data": {
    "loading": false,
    "disabled": false,
    "clickCount": 42
  },
  "metrics": {
    "interactions": 156,
    "renderTime": 2.3,
    "lastUpdate": "2024-01-15T10:30:00Z",
    "performanceScore": 98.5
  }
}
```

---

## ⚡ **WebGPU Integration**

### GPU-Accelerated Computations

```typescript
import { yorhaWebGPU, YoRHaWebGPUMath } from '$lib/components/three/yorha-ui/webgpu/YoRHaWebGPUMath';

// Initialize WebGPU (automatic on module load)
const isSupported = await yorhaWebGPU.initialize();

// Vector operations
const vectorsA = [{ x: 1, y: 2, z: 3 }, { x: 4, y: 5, z: 6 }];
const vectorsB = [{ x: 7, y: 8, z: 9 }, { x: 10, y: 11, z: 12 }];

const result = await yorhaWebGPU.vectorAdd(vectorsA, vectorsB);
console.log('GPU computation time:', result.executionTime, 'ms');
console.log('Result data:', result.data);

// Layout computations
const layoutResult = await yorhaWebGPU.computeLayout(
  nodes,
  { x: 10, y: 8, z: 5 },
  'row'
);

// Physics simulation
const physicsResult = await yorhaWebGPU.simulatePhysics(
  particles,
  0.016, // 60 FPS delta time
  { x: 0, y: -9.81, z: 0 } // gravity
);

// Performance benchmarks
const benchmark = await yorhaWebGPU.getBenchmarkResults();
console.log('WebGPU performance:', benchmark);
```

---

## 🎨 **Theming & Customization**

### YoRHa Color Palette

```typescript
import { YORHA_COLORS, YoRHaThemes } from '$lib/components/three/yorha-ui';

// Available colors
const colors = {
  primary: {
    black: 0x0a0a0a,    // Deep black backgrounds
    white: 0xfaf6ed,    // Off-white text
    beige: 0xd4c5a9,    // Warm beige surfaces
    grey: 0x8b8680      // Medium grey borders
  },
  accent: {
    gold: 0xd4af37,     // Primary gold accent
    amber: 0xffc649,    // Bright amber highlights
    bronze: 0xcd7f32,   // Bronze secondary
    copper: 0xb87333    // Copper tertiary
  },
  status: {
    success: 0x90ee90,  // Light green success
    warning: 0xffa500,  // Orange warnings
    error: 0xff6b6b,    // Red errors
    info: 0x87ceeb      // Light blue info
  },
  interaction: {
    hover: 0xe8dcc0,    // Hover state
    active: 0xffd700,   // Active/pressed state
    disabled: 0x4a4a4a, // Disabled state
    focus: 0xf0e68c     // Focus state
  }
};

// Pre-defined themes
const terminalTheme = YoRHaThemes.TERMINAL;
const alertTheme = YoRHaThemes.ALERT;
```

### Custom Styling

```typescript
const customButton = new YoRHaButton3D({
  text: 'Custom Button',
  // Custom colors
  backgroundColor: 0x2a2a2a,
  borderColor: 0xffd700,
  textColor: 0xffffff,

  // Custom effects
  glow: {
    enabled: true,
    color: 0x00ffff,
    intensity: 0.5
  },

  // Custom animations
  animation: {
    type: 'pulse',
    speed: 2,
    intensity: 0.3
  },

  // Interactive states
  hover: {
    backgroundColor: 0x3a3a3a,
    borderColor: 0x00ffff,
    transform: {
      position: new THREE.Vector3(0, 0.02, 0)
    }
  }
});
```

---

## 🛠️ **Utilities & Helpers**

### Quick Component Creation

```typescript
import { YoRHaUtils, YoRHaQuickSetup } from '$lib/components/three/yorha-ui';

// Utility functions
const quickButton = YoRHaUtils.createButton('Click Me', 'primary');
const quickPanel = YoRHaUtils.createPanel('Data Panel', 5, 4);
const quickInput = YoRHaUtils.createInput('Enter text...', 'text');
const quickModal = YoRHaUtils.createModal('Alert Dialog', 'alert');

// Quick setup patterns
const loginForm = YoRHaQuickSetup.createLoginForm();
const confirmDialog = YoRHaQuickSetup.createConfirmDialog('Delete item?', 'This action cannot be undone');
const settingsPanel = YoRHaQuickSetup.createSettingsPanel();
const toolbar = YoRHaQuickSetup.createToolbar([
  { text: 'Save', icon: 'save', variant: 'primary' },
  { text: 'Load', icon: 'load', variant: 'secondary' },
  { text: 'Delete', icon: 'delete', variant: 'danger' }
]);
```

---

## 🔧 **Integration Examples**

### SvelteKit Integration

```svelte
<!-- src/routes/brain/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { createYoRHaUIDemo } from '$lib/components/three/yorha-ui';

  let container: HTMLElement;
  let uiDemo: any;

  onMount(async () => {
    uiDemo = createYoRHaUIDemo(container);
  });

  onDestroy(() => {
    uiDemo?.dispose();
  });
</script>

<div bind:this={container} class="yorha-container" />

<style>
  .yorha-container {
    width: 100%;
    height: 100vh;
    background: #0a0a0a;
  }
</style>
```

### Legal AI Platform Integration

```typescript
// Legal document analysis interface
const documentPanel = new YoRHaPanel3D({
  title: 'Legal Document Analysis',
  variant: 'terminal',
  width: 12,
  height: 8
});

const searchInput = new YoRHaInput3D({
  placeholder: 'Search legal documents...',
  type: 'search',
  icon: 'search',
  clearable: true
});

const analyzeButton = new YoRHaButton3D({
  text: 'Analyze with AI',
  variant: 'accent',
  icon: 'brain',
  loading: false
});

// Real-time legal data updates
api.subscribe('legal:document:analyzed', (data) => {
  const resultModal = new YoRHaModal3D({
    title: 'Analysis Complete',
    variant: 'default',
    size: 'large'
  });

  // Display analysis results
  resultModal.open();
});
```

---

## 📊 **Performance & Best Practices**

### Memory Management

```typescript
// Always dispose of components when done
component.dispose();
layout.dispose();
modal.dispose();

// Use object pooling for frequently created/destroyed components
class YoRHaComponentPool {
  private buttonPool: YoRHaButton3D[] = [];

  getButton(): YoRHaButton3D {
    return this.buttonPool.pop() || new YoRHaButton3D();
  }

  releaseButton(button: YoRHaButton3D): void {
    button.reset();
    this.buttonPool.push(button);
  }
}
```

### Performance Optimization

```typescript
// Use GPU acceleration when available
const useWebGPU = await yorhaWebGPU.initialize();

// Batch layout updates
layout.setLayoutType('flex');
layout.setDirection('column');
layout.setGap(0.3);
layout.updateLayout(); // Single update call

// Limit animation frame rate for background components
component.setAnimationFrameRate(30); // 30 FPS instead of 60

// Use LOD (Level of Detail) for distant components
component.setLOD({
  high: { distance: 5, quality: 1.0 },
  medium: { distance: 10, quality: 0.7 },
  low: { distance: 20, quality: 0.3 }
});
```

### Bundle Optimization

```typescript
// Tree-shake unused components
import { YoRHaButton3D } from '$lib/components/three/yorha-ui/components/YoRHaButton3D';

// Lazy load heavy components
const YoRHaModal3D = await import('$lib/components/three/yorha-ui/components/YoRHaModal3D');

// Use dynamic imports for optional features
if (needsWebGPU) {
  const { yorhaWebGPU } = await import('$lib/components/three/yorha-ui/webgpu/YoRHaWebGPUMath');
}
```

---

## 🔍 **Debugging & Development**

### Debug Mode

```typescript
// Enable debug mode
const component = new YoRHaButton3D({
  text: 'Debug Button',
  debug: true // Shows bounding boxes, performance metrics
});

// Performance monitoring
component.on('performance', (metrics) => {
  console.log('Render time:', metrics.renderTime);
  console.log('Memory usage:', metrics.memoryUsage);
  console.log('FPS:', metrics.fps);
});
```

### Development Tools

```typescript
// Component inspector
console.log('Component tree:', layout.getComponentTree());
console.log('Layout metrics:', layout.getLayoutMetrics());

// Style debugging
component.dumpStyles(); // Logs all applied styles
component.validateStyles(); // Checks for style conflicts

// Event debugging
component.enableEventLogging(true);
```

---

## 🚀 **Deployment**

### Production Build

```bash
# Build optimized bundle
npm run build:yorha

# Generate type definitions
npm run build:types

# Run performance tests
npm run test:performance
```

### CDN Integration

```html
<!-- Load from CDN -->
<script src="https://unpkg.com/three@latest/build/three.min.js"></script>
<script src="https://cdn.yorha-ui.com/latest/yorha-ui.min.js"></script>

<script>
  const { YoRHaButton3D, createYoRHaUIDemo } = window.YoRHaUI;
  const demo = createYoRHaUIDemo(document.getElementById('container'));
</script>
```

---

## 📈 **Roadmap**

### Version 1.1
- [ ] Advanced typography system with font loading
- [ ] Particle effects and shaders library
- [ ] Audio feedback integration
- [ ] Accessibility improvements (ARIA, keyboard navigation)

### Version 1.2
- [ ] VR/AR support with WebXR integration
- [ ] Advanced physics simulation
- [ ] Custom shader material system
- [ ] Component animation timeline editor

### Version 2.0
- [ ] React and Vue adapters
- [ ] Visual component editor
- [ ] Advanced theming system
- [ ] Plugin architecture

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-org/yorha-ui-3d.git
cd yorha-ui-3d
npm install
npm run dev
```

### Testing

```bash
npm run test          # Run all tests
npm run test:unit     # Unit tests
npm run test:visual   # Visual regression tests
npm run test:perf     # Performance tests
```

---

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 **Acknowledgments**

- **Square Enix** for the incredible NieR: Automata design inspiration
- **Three.js Team** for the amazing 3D library
- **WebGPU Working Group** for next-generation GPU compute
- **SvelteKit Team** for the excellent framework

---

**Built with ❤️ for the YoRHa resistance**

*Glory to mankind.*